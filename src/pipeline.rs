//! One ZED frame in, lanes and a costmap out. Transport-free, so the node,
//! the offline tool and the tests all share it.
use anyhow::Result;

use crate::config::Config;
use crate::costmap::{Costmap, CostmapBuilder, GroundLane};
use crate::geometry::{DepthImage, GroundProjector};
use crate::model::LaneDetector;
use crate::preprocess::{Crop, bgra_to_input};
use crate::zed_frame::{HEIGHT, WIDTH};

pub struct FrameResult {
    /// Lanes in source-image pixels, bottom of the image upwards.
    pub image_lanes: Vec<Vec<(f32, f32)>>,
    pub ground_lanes: Vec<GroundLane>,
    pub costmap: Costmap,
}

pub struct LanePipeline<D> {
    detector: D,
    crop: Crop,
    projector: GroundProjector,
    costmap: CostmapBuilder,
    input: Vec<f32>,
}

impl<D: LaneDetector> LanePipeline<D> {
    pub fn new(detector: D, config: &Config) -> Self {
        Self {
            detector,
            crop: Crop::new(WIDTH, HEIGHT, config.model.cut_top_fraction),
            projector: GroundProjector::new(config.camera.clone(), config.costmap.max_range_m),
            costmap: CostmapBuilder::new(config.costmap.clone()),
            input: Vec::new(),
        }
    }

    /// `bgra` is a full `WIDTH x HEIGHT` ZED left image; `depth` the matching
    /// depth frame in metres, if one arrived.
    pub fn process(&mut self, bgra: &[u8], depth: Option<&[f32]>) -> Result<FrameResult> {
        let (w, h) = self.detector.input_size();
        bgra_to_input(bgra, &self.crop, w, h, &mut self.input);
        let lanes = self.detector.detect(&self.input)?;

        let depth = depth.map(|values| DepthImage {
            width: WIDTH,
            height: HEIGHT,
            values,
        });
        let image_lanes: Vec<Vec<(f32, f32)>> = lanes
            .iter()
            .map(|lane| {
                lane.points
                    .iter()
                    .map(|&(x, y)| self.crop.to_source(x, y))
                    .collect()
            })
            .collect();
        let ground_lanes: Vec<GroundLane> = image_lanes
            .iter()
            .map(|lane| {
                let mut ground: GroundLane = lane
                    .iter()
                    .filter_map(|&(u, v)| self.projector.project(u, v, depth.as_ref()))
                    .collect();
                // Depth noise can make neighbouring rows step backwards;
                // the corridor fill assumes near-to-far order.
                ground.sort_by(|a, b| a.0.total_cmp(&b.0));
                ground
            })
            .filter(|lane| lane.len() >= 2)
            .collect();

        Ok(FrameResult {
            costmap: self.costmap.build(&ground_lanes),
            image_lanes,
            ground_lanes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clrnet::Lane;
    use crate::config::CameraConfig;
    use crate::costmap::{FREE_SPACE, LETHAL_OBSTACLE};
    use crate::zed_frame::BGRA_LEN;

    /// Two straight lanes either side of the image centre.
    struct FakeDetector;

    impl LaneDetector for FakeDetector {
        fn input_size(&self) -> (usize, usize) {
            (800, 320)
        }
        fn detect(&mut self, input: &[f32]) -> Result<Vec<Lane>> {
            assert_eq!(input.len(), 3 * 800 * 320);
            let lane = |x_bottom: f32| Lane {
                score: 0.9,
                points: (0..40)
                    .map(|i| {
                        let y = 1.0 - i as f32 / 71.0;
                        // Converge towards the vanishing point at the centre.
                        (0.5 + (x_bottom - 0.5) * y, y)
                    })
                    .collect(),
            };
            Ok(vec![lane(0.15), lane(0.85)])
        }
    }

    #[test]
    fn frame_to_costmap_without_depth() {
        let config = Config {
            camera: CameraConfig {
                cy: 150.0,
                height_m: 1.2,
                ..CameraConfig::default()
            },
            ..Config::default()
        };
        let mut pipeline = LanePipeline::new(FakeDetector, &config);
        let result = pipeline.process(&vec![0; BGRA_LEN], None).unwrap();
        assert_eq!(result.ground_lanes.len(), 2);

        // One lane left of the car, one right, and free space between them.
        let near_y = |l: &GroundLane| l[0].1;
        assert!(near_y(&result.ground_lanes[0]) > 0.0);
        assert!(near_y(&result.ground_lanes[1]) < 0.0);
        // With CLRNet's 270/590 crop these lanes reach about 2.8 m ahead.
        let map = &result.costmap;
        let (row, col) = map.cell_of(2.0, 0.0).unwrap();
        assert_eq!(map.at(row, col), FREE_SPACE);
        assert!(map.data.contains(&LETHAL_OBSTACLE));
    }
}
