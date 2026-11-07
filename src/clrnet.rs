use anyhow::{Result, ensure};

pub const N_PRIORS: usize = 192;
pub const ROW_LEN: usize = 78;   // 0..2 logits | 2 start_y | 3 start_x | 4 theta | 5 length | 6..78 xs
pub const N_OFFSETS: usize = 72;
const N_STRIPS: usize = N_OFFSETS - 1;
const IMG_W: f64 = 800.0;

/// Points are (x, y) normalised to the *cropped* model input (y = 0 is the top of the
/// crop), ordered bottom of the image upwards.
#[derive(Debug, Clone, PartialEq)]
pub struct Lane {
    pub score: f32,
    pub points: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeParams {
    pub conf_threshold: f32,
    pub nms_threshold: f32,
    pub max_lanes: usize,
}

fn lane_score(row: &[f32]) -> f32 {
    let m = row[0].max(row[1]);
    let (e0, e1) = ((row[0] - m).exp(), (row[1] - m).exp());
    e1 / (e0 + e1) // softmax probability of "lane"
}

/// Port of devIoU in CLRNet's nms_kernel.cu. Rows: [l0, l1, start_y, start_x, length*71, xs*799 (72)].
fn overlaps(a: &[f64], b: &[f64], thr: f64) -> bool {
    let start_a = (a[2] * N_STRIPS as f64 + 0.5) as i64;
    let start_b = (b[2] * N_STRIPS as f64 + 0.5) as i64;
    let start = start_a.max(start_b);
    let end_of = |row: &[f64], s: i64| -> i64 {
        let l = row[4] - 1.0;
        (s as f64 + l + 0.5 - if l < 0.0 { 1.0 } else { 0.0 }) as i64
    };
    let end = end_of(a, start_a).min(end_of(b, start_b)).min(N_OFFSETS as i64 - 1);
    if end < start {
        return false;
    }
    let dist: f64 = (start..=end).map(|i| (a[5 + i as usize] - b[5 + i as usize]).abs()).sum();
    dist < thr * (end - start + 1) as f64
}

/// Greedy NMS, highest score first; returns the kept indices.
fn nms(rows: &[Vec<f64>], scores: &[f32], thr: f64, top_k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]));
    let mut suppressed = vec![false; rows.len()];
    let mut keep = Vec::new();
    for (oi, &i) in order.iter().enumerate() {
        if suppressed[oi] {
            continue;
        }
        keep.push(i);
        for (oj, &j) in order.iter().enumerate().skip(oi + 1) {
            if !suppressed[oj] && overlaps(&rows[i], &rows[j], thr) {
                suppressed[oj] = true;
            }
        }
        if keep.len() == top_k {
            break;
        }
    }
    keep
}

/// `pred` is the raw [192 * 78] model output.
pub fn decode(pred: &[f32], params: &DecodeParams) -> Result<Vec<Lane>> {
    ensure!(pred.len() == N_PRIORS * ROW_LEN, "expected {} values, got {}", N_PRIORS * ROW_LEN, pred.len());
    let candidates: Vec<(&[f32], f32)> = pred
        .chunks_exact(ROW_LEN)
        .map(|row| (row, lane_score(row)))
        .filter(|&(_, s)| s >= params.conf_threshold)
        .collect();
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // The layout the NMS kernel expects: drop theta (col 4), scale length and xs.
    let rows: Vec<Vec<f64>> = candidates
        .iter()
        .map(|(row, _)| {
            let mut r: Vec<f64> = row[..4].iter().chain(&row[5..]).map(|&v| v as f64).collect();
            r[4] *= N_STRIPS as f64;
            for v in &mut r[5..] {
                *v *= IMG_W - 1.0;
            }
            r
        })
        .collect();
    let scores: Vec<f32> = candidates.iter().map(|&(_, s)| s).collect();
    let keep = nms(&rows, &scores, params.nms_threshold as f64, params.max_lanes);

    let mut lanes = Vec::new();
    for i in keep {
        let row = candidates[i].0;
        let mut xs: Vec<f64> = row[6..].iter().map(|&v| v as f64).collect();
        let start = ((row[2] as f64 * N_STRIPS as f64).round().max(0.0) as usize).min(N_STRIPS);
        let length = (row[5] as f64 * N_STRIPS as f64).round() as i64;
        let end = (start as i64 + length - 1).min(N_OFFSETS as i64 - 1);

        // Before `start`: valid only as a contiguous in-range run ending at `start`.
        // From `start` on: valid. Past `end`: never valid.
        let mut valid = vec![false; N_OFFSETS];
        for k in (0..start).rev() {
            if (0.0..=1.0).contains(&xs[k]) { valid[k] = true } else { break }
        }
        for v in valid.iter_mut().skip(start) {
            *v = true;
        }
        for (k, v) in valid.iter_mut().enumerate() {
            if k as i64 > end {
                *v = false;
            }
        }
        for (k, x) in xs.iter_mut().enumerate() {
            if !valid[k] {
                *x = -2.0;
            }
        }

        // Row k sits at y = 1 - k/71, so row 0 is the bottom of the crop.
        let points: Vec<(f32, f32)> = (0..N_OFFSETS)
            .filter(|&k| xs[k] >= 0.0)
            .map(|k| (xs[k] as f32, 1.0 - k as f32 / N_STRIPS as f32))
            .collect();
        if points.len() > 1 {
            lanes.push(Lane { score: candidates[i].1, points });
        }
    }
    Ok(lanes)
}