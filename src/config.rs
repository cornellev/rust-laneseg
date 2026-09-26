//! Node configuration, loaded from TOML. Every field has a default so a
//! partial file (or none at all) works; see `config/lane_seg.toml`.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    pub camera: CameraConfig,
    pub costmap: CostmapConfig,
    pub zenoh: ZenohConfig,
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read config {}", path.display()))?;
        let mut config: Self =
            toml::from_str(&text).with_context(|| format!("parse config {}", path.display()))?;
        // Relative model paths are relative to the config file, not the cwd.
        let base = path.parent().unwrap_or(Path::new("."));
        for model_path in [&mut config.model.backbone, &mut config.model.roi_head] {
            if model_path.is_relative() {
                *model_path = base.join(&*model_path);
            }
        }
        Ok(config)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
    Tensorrt,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Backbone + neck + SPGU + ALAU. Outputs `hm`, `shape_hm`, `batch_features`.
    pub backbone: PathBuf,
    /// Prior pooling + cls/reg layers. Inputs `batch_features`, `prior_xs`.
    pub roi_head: PathBuf,
    pub execution_provider: ExecutionProvider,
    pub trt_fp16: bool,
    /// Where TensorRT caches built engines, so only the first start is slow.
    pub trt_cache_dir: PathBuf,
    /// Fraction of image rows dropped from the top (sky) before resizing.
    /// ADNet's CULane config crops 270/590 = 0.46; tune for the ZED mount.
    pub cut_top_fraction: f32,
    pub conf_threshold: f32,
    /// Lane NMS threshold: mean horizontal distance in model-input pixels.
    pub nms_threshold: f32,
    pub max_lanes: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            backbone: "models/adnet_backbone.onnx".into(),
            roi_head: "models/adnet_roi_head.onnx".into(),
            execution_provider: ExecutionProvider::Cpu,
            trt_fp16: true,
            trt_cache_dir: "trt_cache".into(),
            cut_top_fraction: 0.35,
            conf_threshold: 0.3,
            nms_threshold: 45.0,
            max_lanes: 5,
        }
    }
}

/// Left-camera pinhole intrinsics at VGA plus the mount, used to turn lane
/// pixels into ground-plane metres. Take the real values from
/// `/usr/local/zed/settings/SN<serial>.conf`, section `[LEFT_CAM_VGA]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// Lens height above the road; only used where depth is missing.
    pub height_m: f32,
    /// Downward tilt of the optical axis; only used where depth is missing.
    pub pitch_deg: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        // Typical ZED 2 VGA calibration — replace with the unit's own values.
        Self {
            fx: 264.0,
            fy: 264.0,
            cx: 336.0,
            cy: 188.0,
            height_m: 1.0,
            pitch_deg: 0.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CostmapConfig {
    pub resolution: f32,
    /// Grid extent; the camera's ground point sits at the grid centre, which
    /// is where costmap-planner-rust starts its search.
    pub size_x_m: f32,
    pub size_y_m: f32,
    /// Lane points further than this (or with no usable depth) are dropped.
    pub max_range_m: f32,
    pub line_width_m: f32,
    /// Mark the corridor between the ego lane's two lines as free space.
    pub fill_ego_lane: bool,
    /// When only one ego line is seen, mirror it this far to close the corridor.
    pub assumed_lane_width_m: f32,
    pub free_cost: u8,
    pub lane_cost: u8,
    pub unknown_cost: u8,
}

impl Default for CostmapConfig {
    fn default() -> Self {
        Self {
            resolution: 0.1,
            size_x_m: 30.0,
            size_y_m: 30.0,
            max_range_m: 20.0,
            line_width_m: 0.3,
            fill_ego_lane: true,
            assumed_lane_width_m: 3.5,
            free_cost: crate::costmap::FREE_SPACE,
            lane_cost: crate::costmap::LETHAL_OBSTACLE,
            unknown_cost: crate::costmap::NO_INFORMATION,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ZenohConfig {
    /// Optional Zenoh json5 config (router endpoints, scouting, ...).
    pub config: Option<PathBuf>,
    pub costmap_key: String,
    /// How long to wait for the depth frame matching an image.
    pub depth_wait_ms: u64,
}

impl Default for ZenohConfig {
    fn default() -> Self {
        Self {
            config: None,
            costmap_key: crate::costmap::COSTMAP_KEY_EXPR.into(),
            depth_wait_ms: 20,
        }
    }
}
