//! Lane costmap, laid out like `nav2_msgs/Costmap`: `data[row * size_x + col]`, `col` along
//! +x (forward), `row` along +y (left), cell `(0, 0)` at `origin`. Costs follow nav2: 0 free,
//! 254 lethal, 255 no information; costmap-planner-rust treats >= 253 as impassable.
//!
//! The frame is REP-103 with its origin on the road below the left lens, and the grid is
//! centred there.
//!
//! Wire format (little-endian): the same 32-byte header virtual-rgbd-sensor's occupancy grid
//! uses and costmap-planner-rust's `decode_costmap` reads, then `size_x * size_y` cost bytes.
//!
//! | Offset | Size | Field |
//! |---:|---:|---|
//! | 0 | 4 | stamp seconds, `i32` (ZED image timestamp) |
//! | 4 | 4 | stamp nanoseconds, `u32` |
//! | 8 | 4 | `size_x` (cells forward), `u32` |
//! | 12 | 4 | `size_y` (cells lateral), `u32` |
//! | 16 | 4 | resolution in m/cell, `f32` |
//! | 20 | 4 | `origin_x` in m, `f32` |
//! | 24 | 4 | `origin_y` in m, `f32` |
//! | 28 | 4 | reserved, zero |
use anyhow::{Result, ensure};

use crate::config::CostmapConfig;

pub const COSTMAP_KEY_EXPR: &str = "cev/perception/lane_costmap";
pub const COSTMAP_ENCODING: &str = "application/vnd.cornellev.costmap";
pub const COSTMAP_HEADER_LEN: usize = 32;

pub const FREE_SPACE: u8 = 0;
pub const LETHAL_OBSTACLE: u8 = 254;
pub const NO_INFORMATION: u8 = 255;

/// A lane marking as a ground-frame polyline `(forward, left)` in metres,
/// ordered near to far.
pub type GroundLane = Vec<(f32, f32)>;

#[derive(Debug, Clone, PartialEq)]
pub struct Costmap {
    pub size_x: u32,
    pub size_y: u32,
    pub resolution: f32,
    pub origin_x: f32,
    pub origin_y: f32,
    pub data: Vec<u8>,
}

impl Costmap {
    pub fn at(&self, row: usize, col: usize) -> u8 {
        self.data[row * self.size_x as usize + col]
    }

    /// World `(x, y)` -> fractional `(col, row)`.
    fn to_cell(&self, x: f32, y: f32) -> (f32, f32) {
        (
            (x - self.origin_x) / self.resolution,
            (y - self.origin_y) / self.resolution,
        )
    }

    pub fn cell_of(&self, x: f32, y: f32) -> Option<(usize, usize)> {
        let (c, r) = self.to_cell(x, y);
        (c >= 0.0 && r >= 0.0 && c < self.size_x as f32 && r < self.size_y as f32)
            .then(|| (r as usize, c as usize))
    }

    pub fn encode(&self, timestamp_ns: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(COSTMAP_HEADER_LEN + self.data.len());
        out.extend_from_slice(&((timestamp_ns / 1_000_000_000) as i32).to_le_bytes());
        out.extend_from_slice(&((timestamp_ns % 1_000_000_000) as u32).to_le_bytes());
        out.extend_from_slice(&self.size_x.to_le_bytes());
        out.extend_from_slice(&self.size_y.to_le_bytes());
        out.extend_from_slice(&self.resolution.to_le_bytes());
        out.extend_from_slice(&self.origin_x.to_le_bytes());
        out.extend_from_slice(&self.origin_y.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&self.data);
        out
    }

    /// Returns `(timestamp_ns, costmap)`.
    pub fn decode(payload: &[u8]) -> Result<(u64, Self)> {
        ensure!(payload.len() >= COSTMAP_HEADER_LEN, "costmap too short");
        let u32_at = |o: usize| u32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
        let (size_x, size_y) = (u32_at(8), u32_at(12));
        ensure!(
            payload.len() == COSTMAP_HEADER_LEN + size_x as usize * size_y as usize,
            "costmap length does not match its size"
        );
        let stamp_ns = u32_at(0) as i32 as i64 as u64 * 1_000_000_000 + u32_at(4) as u64;
        Ok((
            stamp_ns,
            Self {
                size_x,
                size_y,
                resolution: f32_at(16),
                origin_x: f32_at(20),
                origin_y: f32_at(24),
                data: payload[COSTMAP_HEADER_LEN..].to_vec(),
            },
        ))
    }
}

pub struct CostmapBuilder {
    config: CostmapConfig,
    size_x: u32,
    size_y: u32,
}

impl CostmapBuilder {
    pub fn new(config: CostmapConfig) -> Self {
        let cells = |m: f32| ((m / config.resolution).round() as u32).max(1);
        Self {
            size_x: cells(config.size_x_m),
            size_y: cells(config.size_y_m),
            config,
        }
    }

    pub fn build(&self, lanes: &[GroundLane]) -> Costmap {
        let c = &self.config;
        let mut map = Costmap {
            size_x: self.size_x,
            size_y: self.size_y,
            resolution: c.resolution,
            origin_x: -(self.size_x as f32 * c.resolution) / 2.0,
            origin_y: -(self.size_y as f32 * c.resolution) / 2.0,
            data: vec![c.unknown_cost; self.size_x as usize * self.size_y as usize],
        };

        // Markings run under the car too; carry each one straight back to the
        // camera so the corridor reaches the planner's start cell.
        let lanes: Vec<GroundLane> = lanes
            .iter()
            .filter(|l| l.len() >= 2)
            .map(|l| {
                let mut extended = vec![(0.0, l[0].1)];
                extended.extend_from_slice(l);
                extended
            })
            .collect();

        if c.fill_ego_lane
            && let Some(corridor) = ego_corridor(&lanes, c.assumed_lane_width_m)
        {
            fill_polygon(&mut map, &corridor, c.free_cost);
        }
        for lane in &lanes {
            draw_polyline(&mut map, lane, c.line_width_m, c.lane_cost);
        }
        map
    }
}

/// Lateral offset of a lane where it is nearest the car.
fn near_offset(lane: &GroundLane) -> f32 {
    lane.iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(0.0, |p| p.1)
}

/// Polygon between the nearest line on each side of the car. With only one
/// side visible, the other is that line shifted by the assumed lane width.
fn ego_corridor(lanes: &[GroundLane], lane_width: f32) -> Option<Vec<(f32, f32)>> {
    let left = lanes
        .iter()
        .filter(|l| near_offset(l) >= 0.0)
        .min_by(|a, b| near_offset(a).total_cmp(&near_offset(b)));
    let right = lanes
        .iter()
        .filter(|l| near_offset(l) < 0.0)
        .max_by(|a, b| near_offset(a).total_cmp(&near_offset(b)));
    let shifted = |lane: &GroundLane, dy: f32| -> GroundLane {
        lane.iter().map(|&(x, y)| (x, y + dy)).collect()
    };
    let (left, right) = match (left, right) {
        (Some(l), Some(r)) => (l.clone(), r.clone()),
        (Some(l), None) => (l.clone(), shifted(l, -lane_width)),
        (None, Some(r)) => (shifted(r, lane_width), r.clone()),
        (None, None) => return None,
    };
    // Only the stretch both sides cover, so the polygon never self-crosses
    // when one line is seen much further out than the other.
    let far = |l: &GroundLane| l.iter().map(|p| p.0).fold(f32::MIN, f32::max);
    let limit = far(&left).min(far(&right));
    let mut polygon: Vec<_> = left.into_iter().filter(|p| p.0 <= limit).collect();
    polygon.extend(right.into_iter().filter(|p| p.0 <= limit).rev());
    (polygon.len() >= 3).then_some(polygon)
}

/// Even-odd scanline fill of a world-frame polygon, sampled at cell centres.
fn fill_polygon(map: &mut Costmap, polygon: &[(f32, f32)], cost: u8) {
    let cells: Vec<(f32, f32)> = polygon.iter().map(|&(x, y)| map.to_cell(x, y)).collect();
    let mut crossings = Vec::new();
    for row in 0..map.size_y as usize {
        let yc = row as f32 + 0.5;
        crossings.clear();
        for (i, &(c0, r0)) in cells.iter().enumerate() {
            let (c1, r1) = cells[(i + 1) % cells.len()];
            if (r0 <= yc) != (r1 <= yc) {
                crossings.push(c0 + (yc - r0) / (r1 - r0) * (c1 - c0));
            }
        }
        crossings.sort_by(f32::total_cmp);
        for span in crossings.chunks_exact(2) {
            let start = (span[0] - 0.5).ceil().max(0.0) as usize;
            let end = ((span[1] - 0.5).floor() as i64).min(map.size_x as i64 - 1);
            for col in start as i64..=end {
                map.data[row * map.size_x as usize + col as usize] = cost;
            }
        }
    }
}

/// Stamp a disk of `width` along every segment of a world-frame polyline.
fn draw_polyline(map: &mut Costmap, line: &[(f32, f32)], width: f32, cost: u8) {
    let radius = (width / 2.0 / map.resolution).max(0.5);
    let reach = radius.ceil() as i64;
    let (size_x, size_y) = (map.size_x as i64, map.size_y as i64);
    for pair in line.windows(2) {
        let (c0, r0) = map.to_cell(pair[0].0, pair[0].1);
        let (c1, r1) = map.to_cell(pair[1].0, pair[1].1);
        let steps = ((c1 - c0).hypot(r1 - r0) * 2.0).ceil().max(1.0) as usize;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let (c, r) = (c0 + (c1 - c0) * t, r0 + (r1 - r0) * t);
            for dr in -reach..=reach {
                for dc in -reach..=reach {
                    let (row, col) = (r.floor() as i64 + dr, c.floor() as i64 + dc);
                    if row < 0 || col < 0 || row >= size_y || col >= size_x {
                        continue;
                    }
                    let (dx, dy) = (col as f32 + 0.5 - c, row as f32 + 0.5 - r);
                    if dx * dx + dy * dy <= radius * radius {
                        map.data[(row * size_x + col) as usize] = cost;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight(left: f32) -> GroundLane {
        (2..=15).map(|x| (x as f32, left)).collect()
    }

    fn builder() -> CostmapBuilder {
        CostmapBuilder::new(CostmapConfig::default())
    }

    #[test]
    fn grid_is_centred_on_camera() {
        let map = builder().build(&[]);
        assert_eq!((map.size_x, map.size_y), (300, 300));
        assert_eq!(map.cell_of(0.0, 0.0), Some((150, 150)));
        assert!(map.data.iter().all(|&c| c == NO_INFORMATION));
    }

    #[test]
    fn ego_lane_is_free_and_lines_are_lethal() {
        let map = builder().build(&[straight(1.75), straight(-1.75), straight(5.25)]);
        let at = |x: f32, y: f32| {
            let (r, c) = map.cell_of(x, y).unwrap();
            map.at(r, c)
        };
        // Corridor, including the stretch back to the camera.
        assert_eq!(at(8.0, 0.0), FREE_SPACE);
        assert_eq!(at(0.5, 0.0), FREE_SPACE);
        assert_eq!(at(8.0, 1.75), LETHAL_OBSTACLE);
        assert_eq!(at(8.0, -1.75), LETHAL_OBSTACLE);
        assert_eq!(at(8.0, 5.25), LETHAL_OBSTACLE);
        // Next lane over is not the ego lane, and behind the camera is unseen.
        assert_eq!(at(8.0, 3.5), NO_INFORMATION);
        assert_eq!(at(-3.0, 0.0), NO_INFORMATION);
    }

    #[test]
    fn single_line_uses_assumed_width() {
        let map = builder().build(&[straight(-1.75)]);
        let (r, c) = map.cell_of(8.0, 1.0).unwrap();
        assert_eq!(map.at(r, c), FREE_SPACE);
        let (r, c) = map.cell_of(8.0, 2.5).unwrap();
        assert_eq!(map.at(r, c), NO_INFORMATION);
    }

    #[test]
    fn planner_can_walk_the_corridor_forward() {
        // Mirrors costmap-planner-rust: start at the centre, walk +col.
        let map = builder().build(&[straight(1.75), straight(-1.75)]);
        let (row, col) = (map.size_y as usize / 2, map.size_x as usize / 2);
        assert!((col..col + 140).all(|c| map.at(row, c) < 253));
    }

    #[test]
    fn wire_roundtrip() {
        let map = builder().build(&[straight(1.75)]);
        let bytes = map.encode(12_000_000_345);
        assert_eq!(bytes.len(), COSTMAP_HEADER_LEN + 300 * 300);
        assert_eq!(Costmap::decode(&bytes).unwrap(), (12_000_000_345, map));
    }

    #[test]
    fn header_matches_costmap_planner_rust() {
        // Offsets decode_costmap in costmap_planner_rs/src/extract.rs reads.
        let map = builder().build(&[]);
        let b = map.encode(0);
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        assert_eq!((u32_at(8), u32_at(12)), (map.size_x, map.size_y)); // cols, rows
        assert_eq!((f32_at(16), f32_at(20), f32_at(24)), (map.resolution, map.origin_x, map.origin_y));
        assert_eq!(&b[32..], &map.data[..]);
    }
}
