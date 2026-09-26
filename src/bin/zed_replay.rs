//! Stand-in for `zed-reader` when there is no ZED: publishes image files on
//! the same Zenoh keys and wire format, looping at 30 FPS. Depth is synthesised
//! as a flat road from the config's camera height/pitch (or omitted with
//! `--no-depth`, like `ZED_READER_DEPTH_MODE=none`).
//!
//!   zed_replay [--config lane_seg.toml] [--no-depth] <image or dir>...
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use image::imageops::FilterType;
use lane_seg::{
    config::{CameraConfig, Config},
    zed_frame::{
        DEPTH_KEY_EXPR, DEPTH_KIND, FRAME_ENCODING, HEIGHT, IMAGE_KEY_EXPR, IMAGE_KIND, WIDTH,
        encode_frame,
    },
};
use zenoh::{Wait, qos::CongestionControl};

const FPS: u64 = 30;

fn load_bgra(path: &PathBuf) -> Result<Vec<u8>> {
    let rgb = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("open {}", path.display()))?
        .resize_exact(WIDTH as u32, HEIGHT as u32, FilterType::Triangle)
        .to_rgb8();
    Ok(rgb.pixels().flat_map(|p| [p[2], p[1], p[0], 255]).collect())
}

/// Z-depth of a flat road seen by the configured camera; NaN above the horizon.
fn flat_depth(camera: &CameraConfig) -> Vec<u8> {
    let (sin, cos) = camera.pitch_deg.to_radians().sin_cos();
    (0..HEIGHT)
        .flat_map(|v| {
            let ry = (v as f32 - camera.cy) / camera.fy;
            let down = ry * cos + sin;
            let z = if down > 1e-3 { camera.height_m / down } else { f32::NAN };
            std::iter::repeat_n(z, WIDTH)
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let (mut config_path, mut with_depth, mut inputs) = (None, true, Vec::new());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => config_path = Some(PathBuf::from(args.next().context("--config PATH")?)),
            "--no-depth" => with_depth = false,
            _ => inputs.push(PathBuf::from(arg)),
        }
    }
    let config = Config::load(config_path.as_deref())?;

    let mut paths = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(&input)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| ["png", "jpg", "jpeg"].contains(&e.to_ascii_lowercase().as_str()))
                })
                .collect();
            entries.sort();
            paths.extend(entries);
        } else {
            paths.push(input);
        }
    }
    ensure!(!paths.is_empty(), "usage: zed_replay [--config FILE] [--no-depth] <image|dir>...");
    let frames = paths.iter().map(load_bgra).collect::<Result<Vec<_>>>()?;
    let depth = flat_depth(&config.camera);

    let session = zenoh::open(zenoh::Config::default())
        .wait()
        .map_err(|e| anyhow::anyhow!("open Zenoh session: {e}"))?;
    let publisher = |key: &'static str| {
        session
            .declare_publisher(key)
            .encoding(FRAME_ENCODING)
            .congestion_control(CongestionControl::Drop)
            .wait()
            .map_err(|e| anyhow::anyhow!("declare {key}: {e}"))
    };
    let image_pub = publisher(IMAGE_KEY_EXPR)?;
    let depth_pub = publisher(DEPTH_KEY_EXPR)?;
    tracing::info!(frames = frames.len(), with_depth, "replaying");

    let start = Instant::now();
    for (frame_id, bgra) in (0u64..).zip(frames.iter().cycle()) {
        let timestamp_ns = start.elapsed().as_nanos() as u64;
        image_pub
            .put(encode_frame(IMAGE_KIND, frame_id, timestamp_ns, bgra))
            .wait()
            .map_err(|e| anyhow::anyhow!("publish image: {e}"))?;
        if with_depth {
            depth_pub
                .put(encode_frame(DEPTH_KIND, frame_id, timestamp_ns, &depth))
                .wait()
                .map_err(|e| anyhow::anyhow!("publish depth: {e}"))?;
        }
        let next = start + Duration::from_nanos((frame_id + 1) * 1_000_000_000 / FPS);
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    Ok(())
}
