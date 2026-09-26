//! ADNet's weight-free head logic, ported from `adnet/models/heads/SPGHead.py`
//! and `adnet/ops/nms/csrc/nms_kernel.cu` (github.com/Sephirex-X/ADNet).
//!
//! The network runs in two ONNX graphs because the start-point decode between
//! them (`decode_theta_xy`) is a data-dependent top-k that does not export:
//!
//!   backbone.onnx  img -> hm, shape_hm, batch_features
//!   [rust]         decode_starts + anchor_xs          (this file)
//!   roi_head.onnx  batch_features, prior_xs -> cls_logits, reg
//!   [rust]         build_proposals + postprocess       (this file)
//!
//! Function names mirror the Python ones so the two can be diffed.

/// Static shape of the head, fixed by the training config.
#[derive(Debug, Clone, Copy)]
pub struct HeadGeometry {
    /// Model input size (`cfg.img_w`, `cfg.img_h`).
    pub img_w: f32,
    pub img_h: f32,
    /// Rows sampled per lane (`S` / `n_offsets`, 72 in every ADNet config).
    pub n_offsets: usize,
}

impl HeadGeometry {
    fn n_strips(&self) -> f32 {
        (self.n_offsets - 1) as f32
    }

    /// `self.anchor_ys = torch.linspace(1, 0, steps=n_offsets)`: normalised
    /// image y of each sample row, bottom (1) to top (0).
    pub fn anchor_y(&self, i: usize) -> f32 {
        1.0 - i as f32 / self.n_strips()
    }
}

/// A start point picked off the heatmap (`decode_theta_xy`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Start {
    /// Normalised image coordinates of the lane's start point.
    pub x: f32,
    pub y: f32,
    /// Lane angle / 180, as regressed by `shape_hm` channel 0.
    pub theta: f32,
}

/// `decode_theta_xy`: the `k` highest heatmap cells become lane start points.
///
/// `hm` and `theta` are one image's `(hm_h, hm_w)` planes of `hm` and
/// `shape_hm[:, 0]`. Top-k on logits equals top-k on the clamped sigmoid the
/// Python takes, apart from tie order among saturated cells.
pub fn decode_starts(
    hm: &[f32],
    theta: &[f32],
    hm_h: usize,
    hm_w: usize,
    k: usize,
    geom: &HeadGeometry,
) -> Vec<Start> {
    assert_eq!(hm.len(), hm_h * hm_w);
    assert_eq!(theta.len(), hm_h * hm_w);
    let mut order: Vec<usize> = (0..hm.len()).collect();
    let k = k.min(order.len());
    let desc = |a: &usize, b: &usize| hm[*b].total_cmp(&hm[*a]);
    if k < order.len() {
        order.select_nth_unstable_by(k, desc);
        order.truncate(k);
    }
    order.sort_by(desc);

    let down_x = geom.img_w / hm_w as f32;
    let down_y = geom.img_h / hm_h as f32;
    order
        .into_iter()
        .map(|flat| Start {
            x: (flat % hm_w) as f32 * down_x / geom.img_w,
            y: (flat / hm_w) as f32 * down_y / geom.img_h,
            theta: theta[flat] / 180.0,
        })
        .collect()
}

/// `generate_anchors_in_one`: the straight line through a start point at its
/// predicted angle, as x (model-input pixels) at every anchor row. These are
/// both the `prior_xs` fed to the RoI head and the base the regression offsets
/// are added to.
pub fn anchor_xs(start: &Start, geom: &HeadGeometry) -> Vec<f32> {
    let angle = (1.0 - start.theta) * std::f32::consts::PI;
    let tan = (angle + 1e-4).tan();
    (0..geom.n_offsets)
        .map(|i| (geom.anchor_y(i) - start.y) * geom.img_h / tan + start.x * geom.img_w)
        .collect()
}

/// One row of `reg_proposals`: `[cls0, cls1, start_y, start_x, theta, length, xs...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub cls_logits: [f32; 2],
    pub start_y: f32,
    pub start_x: f32,
    pub theta: f32,
    pub length: f32,
    /// x in model-input pixels at each anchor row.
    pub xs: Vec<f32>,
}

impl Proposal {
    /// `softmax(cls_logits)[1]`.
    pub fn score(&self) -> f32 {
        let [bg, fg] = self.cls_logits;
        1.0 / (1.0 + (bg - fg).exp())
    }
}

/// `reg_proposals = anchors; [:, :2] = cls_logits; [:, 5:] += reg`.
///
/// `cls_logits` is `(k, 2)` and `reg` is `(k, n_offsets + 1)`, flattened.
pub fn build_proposals(
    starts: &[Start],
    anchors: &[Vec<f32>],
    cls_logits: &[f32],
    reg: &[f32],
    geom: &HeadGeometry,
) -> Vec<Proposal> {
    let reg_len = geom.n_offsets + 1;
    assert_eq!(cls_logits.len(), starts.len() * 2);
    assert_eq!(reg.len(), starts.len() * reg_len);
    starts
        .iter()
        .zip(anchors)
        .enumerate()
        .map(|(i, (start, anchor))| {
            let r = &reg[i * reg_len..(i + 1) * reg_len];
            Proposal {
                cls_logits: [cls_logits[2 * i], cls_logits[2 * i + 1]],
                start_y: start.y,
                start_x: start.x,
                theta: start.theta,
                length: r[0],
                xs: anchor.iter().zip(&r[1..]).map(|(a, d)| a + d).collect(),
            }
        })
        .collect()
}

/// `relocate_xy`: move `start_y` to the first row whose x is inside the image,
/// since that (not the heatmap peak) is what NMS and decoding measure from.
fn relocate_xy(proposal: &mut Proposal, geom: &HeadGeometry) {
    // Python zeroes out-of-image xs, then takes the first nonzero one.
    let first = proposal
        .xs
        .iter()
        .position(|&x| x >= 0.0 && x <= geom.img_w && x != 0.0)
        .unwrap_or(geom.n_offsets);
    proposal.start_y = first as f32 / geom.n_strips();
}

/// `devIoU` from the CUDA NMS kernel: true when two lanes' mean horizontal
/// distance over their shared rows is below `threshold` pixels.
fn lanes_overlap(a: &Proposal, b: &Proposal, threshold: f32, geom: &HeadGeometry) -> bool {
    let n_strips = geom.n_strips();
    // C float -> int conversions truncate toward zero; `as i32` does the same.
    let start_of = |p: &Proposal| (p.start_y * n_strips + 0.5) as i32;
    let end_of = |p: &Proposal, start: i32| {
        (start as f32 + p.length - 1.0 + 0.5 - f32::from(u8::from(p.length - 1.0 < 0.0))) as i32
    };
    let (start_a, start_b) = (start_of(a), start_of(b));
    let start = start_a.max(start_b);
    let end = end_of(a, start_a)
        .min(end_of(b, start_b))
        .min(geom.n_offsets as i32 - 1);
    if end < start {
        return false;
    }
    let dist: f32 = (start as usize..=end as usize)
        .map(|i| (a.xs[i] - b.xs[i]).abs())
        .sum();
    dist < threshold * (end - start + 1) as f32
}

/// A detected lane in normalised model-input coordinates (x over width, y over
/// the cropped image height), ordered from the bottom of the image upwards.
#[derive(Debug, Clone, PartialEq)]
pub struct Lane {
    pub score: f32,
    pub points: Vec<(f32, f32)>,
}

/// `SPGHead.nms` followed by `get_lanes` / `proposals_to_pred` at test time.
pub fn postprocess(
    mut proposals: Vec<Proposal>,
    conf_threshold: f32,
    nms_threshold: f32,
    max_lanes: usize,
    geom: &HeadGeometry,
) -> Vec<Lane> {
    proposals.retain(|p| p.score() > conf_threshold);
    for p in &mut proposals {
        relocate_xy(p, geom);
    }
    proposals.sort_by(|a, b| b.score().total_cmp(&a.score()));

    let mut kept: Vec<Proposal> = Vec::new();
    for candidate in proposals {
        if kept.len() == max_lanes {
            break;
        }
        if !kept
            .iter()
            .any(|k| lanes_overlap(k, &candidate, nms_threshold, geom))
        {
            kept.push(candidate);
        }
    }

    kept.iter()
        .filter_map(|p| proposal_to_lane(p, geom))
        .collect()
}

/// `proposals_to_pred`: keep rows `start..=start+length-1`, plus any unbroken
/// run of in-image rows below `start` (the lane is extended to the image edge).
fn proposal_to_lane(p: &Proposal, geom: &HeadGeometry) -> Option<Lane> {
    let n_strips = geom.n_offsets - 1;
    let length = p.length.round() as i64;
    let start = ((p.start_y * n_strips as f32).round().max(0.0) as usize).min(n_strips);
    // Python's negative-index slicing makes end < -1 clear only the tail; that
    // is a bug upstream, so treat any non-positive length as an empty lane.
    let end = (start as i64 + length - 1).min(n_strips as i64);
    if end < 0 {
        return None;
    }
    let end = end as usize;

    let xs: Vec<f32> = p.xs.iter().map(|x| x / geom.img_w).collect();
    let in_image = |x: f32| (0.0..=1.0).contains(&x);
    let mut extend_from = start;
    while extend_from > 0 && in_image(xs[extend_from - 1]) {
        extend_from -= 1;
    }

    let points: Vec<(f32, f32)> = (extend_from..=end)
        .filter(|&i| xs[i] >= 0.0)
        .map(|i| (xs[i], geom.anchor_y(i)))
        .collect();
    (points.len() > 1).then(|| Lane {
        score: p.score(),
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEOM: HeadGeometry = HeadGeometry {
        img_w: 800.0,
        img_h: 320.0,
        n_offsets: 72,
    };

    fn proposal(x: f32, score_logit: f32, length: f32) -> Proposal {
        Proposal {
            cls_logits: [0.0, score_logit],
            start_y: 0.0,
            start_x: x / GEOM.img_w,
            theta: 0.5,
            length,
            xs: vec![x; GEOM.n_offsets],
        }
    }

    #[test]
    fn decode_picks_top_k_in_score_order() {
        // 2x4 heatmap; hm_down_scale = 800/4 = 200 in x, 320/2 = 160 in y.
        let hm = [0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let theta = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 45.0, 0.0];
        let starts = decode_starts(&hm, &theta, 2, 4, 2, &GEOM);
        assert_eq!(
            starts,
            vec![
                Start { x: 0.25, y: 0.0, theta: 0.5 },
                Start { x: 0.5, y: 0.5, theta: 0.25 },
            ]
        );
    }

    #[test]
    fn vertical_anchor_stays_at_start_x() {
        // theta 0.5 -> 90 degrees -> a vertical line.
        let xs = anchor_xs(&Start { x: 0.5, y: 1.0, theta: 0.5 }, &GEOM);
        assert!(xs.iter().all(|x| (x - 400.0).abs() < 0.1), "{xs:?}");
    }

    #[test]
    fn slanted_anchor_matches_python() {
        // theta 0.25 -> angle 135 deg -> tan = -1, so x grows as y shrinks:
        // x_i = (anchor_y_i - start_y) * img_h / -1 + start_x * img_w.
        let xs = anchor_xs(&Start { x: 0.25, y: 1.0, theta: 0.25 }, &GEOM);
        assert!((xs[0] - 200.0).abs() < 0.1);
        assert!((xs[71] - 520.0).abs() < 0.5, "{}", xs[71]);
    }

    #[test]
    fn nms_merges_near_duplicates_and_keeps_distinct_lanes() {
        let lanes = postprocess(
            vec![
                proposal(200.0, 4.0, 72.0),
                proposal(210.0, 3.0, 72.0), // 10 px from the first: suppressed
                proposal(600.0, 2.0, 72.0),
                proposal(400.0, -4.0, 72.0), // below conf threshold
            ],
            0.3,
            45.0,
            5,
            &GEOM,
        );
        let xs: Vec<f32> = lanes.iter().map(|l| l.points[0].0 * GEOM.img_w).collect();
        assert_eq!(xs, vec![200.0, 600.0]);
    }

    #[test]
    fn max_lanes_caps_output() {
        let proposals = (0..6).map(|i| proposal(100.0 * i as f32 + 50.0, 1.0, 72.0)).collect();
        assert_eq!(postprocess(proposals, 0.3, 45.0, 3, &GEOM).len(), 3);
    }

    #[test]
    fn length_trims_the_far_end() {
        let lane = postprocess(vec![proposal(300.0, 4.0, 10.0)], 0.3, 45.0, 5, &GEOM);
        // Rows 0..=9, bottom of the image upwards.
        assert_eq!(lane[0].points.len(), 10);
        assert_eq!(lane[0].points[0].1, 1.0);
    }

    #[test]
    fn lane_extends_down_to_image_edge() {
        // Out of image for rows 0..5, then inside from row 5 on.
        let mut p = proposal(300.0, 4.0, 20.0);
        for x in &mut p.xs[..5] {
            *x = -50.0;
        }
        let lane = postprocess(vec![p], 0.3, 45.0, 5, &GEOM);
        // relocate_xy moves the start to row 5; rows 5..=24 are kept.
        assert_eq!(lane[0].points.len(), 20);
        assert!((lane[0].points[0].1 - GEOM.anchor_y(5)).abs() < 1e-6);
    }
}
