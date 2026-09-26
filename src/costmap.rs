//! Lane costmap, laid out like `nav2_msgs/Costmap` so costmap-planner-rust's
//! `Grid` reads it unchanged: `data[row * size_x + col]`, `col` along +x
//! (forward), `row` along +y (left), cell `(0, 0)` at `origin`.
//!
//! Costs follow nav2: 0 free, 254 lethal, 255 no information. The planner
//! treats anything >= 253 as impassable and starts its search at the grid
//! centre, so the grid is centred on the camera.
//!
//! Wire format on `perception/lane_costmap` (all little-endian), a 48-byte
//! header followed by `size_x * size_y` cost bytes:
//!
//! | Offset | Size | Field |
//! |---:|---:|---|
//! | 0 | 4 | ASCII magic `LCM1` |
//! | 4 | 1 | format version (`1`) |
//! | 5 | 3 | reserved, zero |
//! | 8 | 8 | ZED `frame_id` this map was built from, `u64` |
//! | 16 | 8 | ZED image timestamp in ns, `u64` |
//! | 24 | 4 | `size_x` (cells forward), `u32` |
//! | 28 | 4 | `size_y` (cells lateral), `u32` |
//! | 32 | 4 | resolution in m/cell, `f32` |
//! | 36 | 4 | `origin_x` in m, `f32` |
//! | 40 | 4 | `origin_y` in m, `f32` |
//! | 44 | 4 | reserved, zero |
use anyhow::{Result, ensure};

use crate::config::CostmapConfig;

pub const COSTMAP_KEY_EXPR: &str = "perception/lane_costmap";
pub const COSTMAP_ENCODING: &str = "application/vnd.cornellev.costmap";
pub const COSTMAP_HEADER_LEN: usize = 48;

pub const FREE_SPACE: u8 = 0;
pub const LETHAL_OBSTACLE: u8 = 254;
pub const NO_INFORMATION: u8 = 255;

const MAGIC: &[u8; 4] = b"LCM1";
const VERSION: u8 = 1;

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

    pub fn encode(&self, frame_id: u64, timestamp_ns: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(COSTMAP_HEADER_LEN + self.data.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&[VERSION, 0, 0, 0]);
        out.extend_from_slice(&frame_id.to_le_bytes());
        out.extend_from_slice(&timestamp_ns.to_le_bytes());
        out.extend_from_slice(&self.size_x.to_le_bytes());
        out.extend_from_slice(&self.size_y.to_le_bytes());
        out.extend_from_slice(&self.resolution.to_le_bytes());
        out.extend_from_slice(&self.origin_x.to_le_bytes());
        out.extend_from_slice(&self.origin_y.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&self.data);
        out
    }

    /// Returns `(frame_id, timestamp_ns, costmap)`.
    pub fn decode(payload: &[u8]) -> Result<(u64, u64, Self)> {
        ensure!(payload.len() >= COSTMAP_HEADER_LEN, "costmap too short");
        ensure!(&payload[..4] == MAGIC, "invalid costmap magic");
        ensure!(payload[4] == VERSION, "unsupported costmap version");
        let u32_at = |o: usize| u32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(payload[o..o + 8].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
        let (size_x, size_y) = (u32_at(24), u32_at(28));
        ensure!(
            payload.len() == COSTMAP_HEADER_LEN + size_x as usize * size_y as usize,
            "costmap length does not match its size"
        );
        Ok((
            u64_at(8),
            u64_at(16),
            Self {
                size_x,
                size_y,
                resolution: f32_at(32),
                origin_x: f32_at(36),
                origin_y: f32_at(40),
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
        let bytes = map.encode(7, 99);
        assert_eq!(bytes.len(), COSTMAP_HEADER_LEN + 300 * 300);
        assert_eq!(Costmap::decode(&bytes).unwrap(), (7, 99, map));
    }
}
