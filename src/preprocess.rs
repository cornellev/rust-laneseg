/// Where the model input sits in the original image, for mapping lanes back.
#[derive(Debug, Clone, Copy)]
pub struct Crop {
    pub src_w: usize,
    pub src_h: usize,
    pub cut_top: usize, // rows removed from the top
}

impl Crop {
    pub fn new(src_w: usize, src_h: usize, cut_top_fraction: f32) -> Self {
        let cut_top = ((src_h as f32 * cut_top_fraction.clamp(0.0, 0.9)).round() as usize).min(src_h - 1);
        Self { src_w, src_h, cut_top }
    }
    /// Normalised model-input coordinates -> source-image pixels.
    pub fn to_source(&self, x_norm: f32, y_norm: f32) -> (f32, f32) {
        let crop_h = (self.src_h - self.cut_top) as f32;
        (x_norm * self.src_w as f32, self.cut_top as f32 + y_norm * crop_h)
    }
}

/// OpenCV's bicubic weights (a = -0.75) for a fractional offset t.
fn cubic_weights(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let w0 = ((A * (t + 1.0) - 5.0 * A) * (t + 1.0) + 8.0 * A) * (t + 1.0) - 4.0 * A;
    let w1 = ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
    let u = 1.0 - t;
    let w2 = ((A + 2.0) * u - (A + 3.0)) * u * u + 1.0;
    [w0, w1, w2, 1.0 - w0 - w1 - w2]
}

/// For each destination index: four source taps (clamped at the edges) and their weights.
fn axis_taps(dst_len: usize, scale: f32, src_len: usize) -> Vec<([usize; 4], [f32; 4])> {
    (0..dst_len)
        .map(|d| {
            let s = (d as f32 + 0.5) * scale - 0.5;
            let i = s.floor();
            let w = cubic_weights(s - i);
            let i = i as isize;
            let tap = |k: isize| (i + k).clamp(0, src_len as isize - 1) as usize;
            ([tap(-1), tap(0), tap(1), tap(2)], w)
        })
        .collect()
}

/// Cubic-resize the cropped BGRA image into `out` as [3, dst_h, dst_w] BGR / 255.
pub fn bgra_to_input(bgra: &[u8], crop: &Crop, dst_w: usize, dst_h: usize, out: &mut Vec<f32>) {
    let (src_w, src_h) = (crop.src_w, crop.src_h);
    assert_eq!(bgra.len(), src_w * src_h * 4);
    let crop_h = src_h - crop.cut_top;
    let plane = dst_w * dst_h;
    out.clear();
    out.resize(3 * plane, 0.0);

    let xt = axis_taps(dst_w, src_w as f32 / dst_w as f32, src_w);
    let yt = axis_taps(dst_h, crop_h as f32 / dst_h as f32, crop_h);

    for (y, (yi, yw)) in yt.iter().enumerate() {
        for (x, (xi, xw)) in xt.iter().enumerate() {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (&ry, &wy) in yi.iter().zip(yw) {
                    let row = (crop.cut_top + ry) * src_w * 4;
                    let mut line = 0.0f32;
                    for (&rx, &wx) in xi.iter().zip(xw) {
                        line += bgra[row + rx * 4 + c] as f32 * wx;
                    }
                    acc += line * wy;
                }
                out[c * plane + y * dst_w + x] = (acc / 255.0).clamp(0.0, 1.0);
            }
        }
    }
}