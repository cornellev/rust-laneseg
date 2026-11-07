use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// File the stack puts beside each car's `car.toml` (`$CEV_CAR_DIR/lane_seg.toml`).
pub const CONFIG_FILE: &str = "lane_seg.toml";

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    pub camera: CameraConfig,
    pub costmap: CostmapConfig,
    pub zenoh: ZenohConfig,
}

impl Config {
    /// No file means all defaults. Relative paths are resolved against the file's directory.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let text = std::fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
        let mut config: Self = toml::from_str(&text).with_context(|| format!("parse config {}", path.display()))?;
        let base = path.parent().unwrap_or(Path::new("."));
        for p in [&mut config.model.path, &mut config.model.trt_cache_dir] {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        }
        if let Some(p) = &mut config.zenoh.config
            && p.is_relative()
        {
            *p = base.join(&*p);
        }
        Ok(config)
    }

    /// First argument, else `$LANE_SEG_CONFIG`, else `$CEV_CAR_DIR/lane_seg.toml`.
    pub fn find_path() -> Option<PathBuf> {
        std::env::args_os()
            .nth(1)
            .or_else(|| std::env::var_os("LANE_SEG_CONFIG"))
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("CEV_CAR_DIR").map(|dir| Path::new(&dir).join(CONFIG_FILE)))
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
    pub path: PathBuf,
    pub execution_provider: ExecutionProvider,
    pub trt_fp16: bool,            // TensorRT builds an FP16 engine from the FP32 ONNX
    pub trt_cache_dir: PathBuf,    // engine cache, so only the first start is slow
    pub cut_top_fraction: f32,     // CULane crops 270 of 590 rows = 0.4576
    pub conf_threshold: f32,
    pub nms_threshold: f32,
    pub max_lanes: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: "models/clrnet_r18.onnx".into(),
            execution_provider: ExecutionProvider::Cpu,
            trt_fp16: true,
            trt_cache_dir: "trt_cache".into(),
            cut_top_fraction: 270.0 / 590.0,
            conf_threshold: 0.4,
            nms_threshold: 50.0,
            max_lanes: 4,
        }
    }
}

/// Left-camera pinhole intrinsics at VGA plus the mount, used to turn lane pixels into
/// ground-plane metres. Real values: `/usr/local/zed/settings/SN<serial>.conf`, `[LEFT_CAM_VGA]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub height_m: f32,  // lens above the road; only used where depth is missing
    pub pitch_deg: f32, // downward tilt; only used where depth is missing
}

impl Default for CameraConfig {
    fn default() -> Self {
        // Typical ZED 2 VGA calibration -- replace with the unit's own values.
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
    pub size_x_m: f32, // grid is centred on the camera
    pub size_y_m: f32,
    pub max_range_m: f32, // lane points further than this are dropped
    pub line_width_m: f32,
    pub fill_ego_lane: bool, // mark the corridor between the ego lane's lines as free
    pub assumed_lane_width_m: f32, // mirror a lone ego line this far to close the corridor
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
    pub config: Option<PathBuf>, // optional Zenoh json5 config (endpoints, scouting, ...)
    pub costmap_key: String,
    pub depth_wait_ms: u64, // how long to wait for the depth frame matching an image
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
