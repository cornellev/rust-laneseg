//! ZED BGRA frame -> ADNet input tensor.
//!
//! Matches ADNet's test pipeline (`base_dataset.py` + `generate_lane_pts.py`):
//! crop `cut_height` rows off the top, resize to the model size, keep OpenCV's
//! BGR channel order, scale to [0, 1], no mean/std normalisation, CHW layout.

/// Where the model input sits in the original image, for mapping lanes back.
#[derive(Debug, Clone, Copy)]
pub struct Crop {
    pub src_w: usize,
    pub src_h: usize,
    /// Rows removed from the top of the source image.
    pub cut_top: usize,
}

impl Crop {
    pub fn new(src_w: usize, src_h: usize, cut_top_fraction: f32) -> Self {
        let cut_top = ((src_h as f32 * cut_top_fraction.clamp(0.0, 0.9)).round() as usize)
            .min(src_h - 1);
        Self {
            src_w,
            src_h,
            cut_top,
        }
    }

    /// Normalised model-input coordinates -> source-image pixel coordinates.
    pub fn to_source(&self, x_norm: f32, y_norm: f32) -> (f32, f32) {
        let crop_h = (self.src_h - self.cut_top) as f32;
        (
            x_norm * self.src_w as f32,
            self.cut_top as f32 + y_norm * crop_h,
        )
    }
}

/// Bilinear-resize the cropped BGRA image into `out` as `[3, dst_h, dst_w]` BGR / 255.
pub fn bgra_to_input(bgra: &[u8], crop: &Crop, dst_w: usize, dst_h: usize, out: &mut Vec<f32>) {
    let (src_w, src_h) = (crop.src_w, crop.src_h);
    assert_eq!(bgra.len(), src_w * src_h * 4);
    let crop_h = src_h - crop.cut_top;
    let plane = dst_w * dst_h;
    out.clear();
    out.resize(3 * plane, 0.0);

    // Half-pixel-centre sampling, same as cv2.resize(INTER_LINEAR).
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = crop_h as f32 / dst_h as f32;
    let axis = |dst: usize, scale: f32, len: usize| {
        let s = ((dst as f32 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (s as usize).min(len - 1);
        (i0, (i0 + 1).min(len - 1), s - i0 as f32)
    };
    let xs: Vec<_> = (0..dst_w).map(|x| axis(x, scale_x, src_w)).collect();

    for y in 0..dst_h {
        let (y0, y1, fy) = axis(y, scale_y, crop_h);
        let row0 = &bgra[(crop.cut_top + y0) * src_w * 4..];
        let row1 = &bgra[(crop.cut_top + y1) * src_w * 4..];
        for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
            for c in 0..3 {
                let p = |row: &[u8], xi: usize| row[xi * 4 + c] as f32;
                let top = p(row0, x0) + (p(row0, x1) - p(row0, x0)) * fx;
                let bottom = p(row1, x0) + (p(row1, x1) - p(row1, x0)) * fx;
                out[c * plane + y * dst_w + x] = (top + (bottom - top) * fy) / 255.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_maps_back_to_source() {
        let crop = Crop::new(672, 376, 0.25);
        assert_eq!(crop.cut_top, 94);
        assert_eq!(crop.to_source(0.5, 0.0), (336.0, 94.0));
        assert_eq!(crop.to_source(1.0, 1.0), (672.0, 376.0));
    }

    #[test]
    fn keeps_bgr_order_and_scales() {
        let (w, h) = (8, 8);
        // B=10, G=20, R=30, A=255 everywhere.
        let bgra: Vec<u8> = (0..w * h).flat_map(|_| [10, 20, 30, 255]).collect();
        let crop = Crop::new(w, h, 0.5);
        let mut out = Vec::new();
        bgra_to_input(&bgra, &crop, 4, 2, &mut out);
        assert_eq!(out.len(), 3 * 8);
        assert!((out[0] - 10.0 / 255.0).abs() < 1e-6);
        assert!((out[8] - 20.0 / 255.0).abs() < 1e-6);
        assert!((out[16] - 30.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn crop_drops_top_rows() {
        let (w, h) = (4, 4);
        // Top half black, bottom half white.
        let bgra: Vec<u8> = (0..w * h)
            .flat_map(|i| if i / w < 2 { [0, 0, 0, 255] } else { [255, 255, 255, 255] })
            .collect();
        let mut out = Vec::new();
        bgra_to_input(&bgra, &Crop::new(w, h, 0.5), 4, 2, &mut out);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }
}
