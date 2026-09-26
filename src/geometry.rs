//! Image pixels -> ground-plane metres.
//!
//! Output frame follows REP-103 with its origin on the road directly below
//! the left camera lens: x forward, y left. Depth from the ZED is used where
//! it is valid; elsewhere the pixel ray is intersected with a flat road at the
//! configured camera height and pitch.
use crate::config::CameraConfig;

/// A lane-point lookup into the ZED depth image (`WIDTH * HEIGHT` metres,
/// NaN/inf where invalid).
pub struct DepthImage<'a> {
    pub width: usize,
    pub height: usize,
    pub values: &'a [f32],
}

impl DepthImage<'_> {
    /// Median of the valid depths in a small window: lane markings are thin
    /// and stereo depth is noisy right on their edges.
    fn sample(&self, u: f32, v: f32) -> Option<f32> {
        const R: isize = 2;
        let (cu, cv) = (u.round() as isize, v.round() as isize);
        let mut found = [0.0f32; ((2 * R + 1) * (2 * R + 1)) as usize];
        let mut n = 0;
        for dv in -R..=R {
            for du in -R..=R {
                let (x, y) = (cu + du, cv + dv);
                if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
                    continue;
                }
                let d = self.values[y as usize * self.width + x as usize];
                if d.is_finite() && d > 0.0 {
                    found[n] = d;
                    n += 1;
                }
            }
        }
        let found = &mut found[..n];
        (n > 0).then(|| {
            found.sort_by(f32::total_cmp);
            found[n / 2]
        })
    }
}

pub struct GroundProjector {
    camera: CameraConfig,
    max_range: f32,
}

impl GroundProjector {
    pub fn new(camera: CameraConfig, max_range: f32) -> Self {
        Self { camera, max_range }
    }

    /// Pixel `(u, v)` in the left image -> `(forward, left)` metres.
    pub fn project(&self, u: f32, v: f32, depth: Option<&DepthImage>) -> Option<(f32, f32)> {
        let c = &self.camera;
        let (rx, ry) = ((u - c.cx) / c.fx, (v - c.cy) / c.fy);
        let point = match depth.and_then(|d| d.sample(u, v)) {
            // ZED depth is Z along the optical axis. Camera frame is
            // x right, y down, z forward, so forward = z and left = -x.
            Some(z) => (z, -rx * z),
            None => self.flat_ground(rx, ry)?,
        };
        (point.0 > 0.0 && point.0.hypot(point.1) <= self.max_range).then_some(point)
    }

    /// Intersect the pixel ray with a flat road `height_m` below the lens.
    fn flat_ground(&self, rx: f32, ry: f32) -> Option<(f32, f32)> {
        let (sin, cos) = self.camera.pitch_deg.to_radians().sin_cos();
        // Ray (rx, ry, 1) rotated by the downward pitch into a level frame.
        let down = ry * cos + sin;
        let forward = cos - ry * sin;
        if down <= 1e-4 {
            return None; // at or above the horizon
        }
        let t = self.camera.height_m / down;
        Some((forward * t, -rx * t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projector(pitch_deg: f32) -> GroundProjector {
        GroundProjector::new(
            CameraConfig {
                fx: 100.0,
                fy: 100.0,
                cx: 50.0,
                cy: 50.0,
                height_m: 1.0,
                pitch_deg,
            },
            50.0,
        )
    }

    #[test]
    fn flat_ground_level_camera() {
        // 10 px below centre with f=100 -> 0.1 rad down -> 10 m ahead at 1 m high.
        let (fwd, left) = projector(0.0).project(50.0, 60.0, None).unwrap();
        assert!((fwd - 10.0).abs() < 1e-4 && left.abs() < 1e-6);
        // Right of centre is negative y (REP-103).
        let (_, left) = projector(0.0).project(60.0, 60.0, None).unwrap();
        assert!(left < 0.0);
        assert!(projector(0.0).project(50.0, 40.0, None).is_none(), "above horizon");
    }

    #[test]
    fn pitch_brings_ground_closer() {
        let (level, _) = projector(0.0).project(50.0, 60.0, None).unwrap();
        let (pitched, _) = projector(5.0).project(50.0, 60.0, None).unwrap();
        assert!(pitched < level);
    }

    #[test]
    fn depth_wins_over_flat_ground() {
        let values = vec![7.0f32; 100 * 100];
        let depth = DepthImage { width: 100, height: 100, values: &values };
        let (fwd, left) = projector(0.0).project(70.0, 60.0, Some(&depth)).unwrap();
        assert!((fwd - 7.0).abs() < 1e-6);
        assert!((left + 1.4).abs() < 1e-5);
    }

    #[test]
    fn invalid_depth_falls_back() {
        let values = vec![f32::NAN; 100 * 100];
        let depth = DepthImage { width: 100, height: 100, values: &values };
        let (fwd, _) = projector(0.0).project(50.0, 60.0, Some(&depth)).unwrap();
        assert!((fwd - 10.0).abs() < 1e-4);
    }
}
