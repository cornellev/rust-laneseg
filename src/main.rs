//! clrnet-lane: ZED image (+ depth) over Zenoh -> CLRNet -> lane costmap over Zenoh.
//!
//! Config: first argument, else `$LANE_SEG_CONFIG`, else `$CEV_CAR_DIR/lane_seg.toml`
//! (rust-autonomy-stack sets `CEV_CAR_DIR` to the car's directory), else defaults.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use clrnet_lane::{
    config::Config,
    costmap::COSTMAP_ENCODING,
    model::ClrnetOnnx,
    pipeline::LanePipeline,
    zed_frame::{DEPTH_KEY_EXPR, IMAGE_KEY_EXPR, decode_depth_frame, decode_image_frame},
};
use zenoh::{Wait, handlers::RingChannel, qos::CongestionControl};

const LOG_EVERY_N_FRAMES: u64 = 30;
/// Depth frames kept around while waiting for their matching image.
const DEPTH_BACKLOG: usize = 4;
static RUNNING: AtomicBool = AtomicBool::new(true);

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    ctrlc::set_handler(|| RUNNING.store(false, Ordering::SeqCst))?;

    let config_path = Config::find_path();
    let config = Config::load(config_path.as_deref())?;
    match &config_path {
        Some(path) => tracing::info!(config = %path.display(), "loaded config"),
        None => tracing::warn!("no config given and CEV_CAR_DIR unset, using defaults"),
    }

    // The first TensorRT start builds the engines (minutes); later starts load them from the cache.
    tracing::info!(model = %config.model.path.display(), provider = ?config.model.execution_provider, "loading model");
    let mut pipeline = LanePipeline::new(ClrnetOnnx::load(&config.model)?, &config);

    let zenoh_config = match &config.zenoh.config {
        Some(path) => zenoh::Config::from_file(path)
            .map_err(|e| anyhow::anyhow!("load Zenoh config {}: {e}", path.display()))?,
        None => zenoh::Config::default(),
    };
    let session = zenoh::open(zenoh_config)
        .wait()
        .map_err(|e| anyhow::anyhow!("open Zenoh session: {e}"))?;
    // Latest image only: if inference falls behind, skip frames rather than lag.
    let images = session
        .declare_subscriber(IMAGE_KEY_EXPR)
        .with(RingChannel::new(1))
        .wait()
        .map_err(|e| anyhow::anyhow!("subscribe {IMAGE_KEY_EXPR}: {e}"))?;
    let depths = session
        .declare_subscriber(DEPTH_KEY_EXPR)
        .with(RingChannel::new(DEPTH_BACKLOG))
        .wait()
        .map_err(|e| anyhow::anyhow!("subscribe {DEPTH_KEY_EXPR}: {e}"))?;
    let publisher = session
        .declare_publisher(config.zenoh.costmap_key.clone())
        .encoding(COSTMAP_ENCODING)
        .congestion_control(CongestionControl::Drop)
        .wait()
        .map_err(|e| anyhow::anyhow!("declare costmap publisher: {e}"))?;
    tracing::info!(image = IMAGE_KEY_EXPR, depth = DEPTH_KEY_EXPR, costmap = config.zenoh.costmap_key, "running");

    let depth_wait = Duration::from_millis(config.zenoh.depth_wait_ms);
    let mut depth_backlog: VecDeque<(u64, Vec<f32>)> = VecDeque::new();
    let mut bgra = Vec::new();
    let (mut processed, mut without_depth) = (0u64, 0u64);
    let mut last_log = Instant::now();

    while RUNNING.load(Ordering::SeqCst) {
        let Some(sample) = images
            .recv_timeout(Duration::from_millis(100))
            .map_err(|e| anyhow::anyhow!("receive image: {e}"))?
        else {
            continue;
        };
        // Copy out and drop the sample so its SHM slot goes back to zed-reader.
        let (frame_id, timestamp_ns) = {
            let payload = sample.payload().to_bytes();
            let frame = match decode_image_frame(&payload) {
                Ok(frame) => frame,
                Err(error) => {
                    tracing::warn!(%error, "bad image frame");
                    continue;
                }
            };
            bgra.clear();
            bgra.extend_from_slice(frame.data);
            (frame.frame_id, frame.timestamp_ns)
        };
        drop(sample);

        // Image and depth are published separately; join them on frame_id.
        let deadline = Instant::now() + depth_wait;
        let depth = loop {
            if let Some(i) = depth_backlog.iter().position(|(id, _)| *id == frame_id) {
                break depth_backlog.remove(i).map(|(_, d)| d);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            match depths.recv_timeout(remaining) {
                Ok(Some(sample)) => match decode_depth_frame(&sample.payload().to_bytes()) {
                    Ok(d) if d.frame_id >= frame_id => {
                        if depth_backlog.len() == DEPTH_BACKLOG {
                            depth_backlog.pop_front();
                        }
                        depth_backlog.push_back((d.frame_id, d.values().collect()));
                    }
                    Ok(_) => {} // older than the image we are on
                    Err(error) => tracing::warn!(%error, "bad depth frame"),
                },
                // Timed out, or no depth publisher (ZED_READER_DEPTH_MODE=none).
                _ => break None,
            }
        };
        if depth.is_none() {
            without_depth += 1;
        }

        let result = match pipeline.process(&bgra, depth.as_deref()) {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(%error, frame_id, "inference failed");
                continue;
            }
        };
        if let Err(error) = publisher.put(result.costmap.encode(timestamp_ns)).wait() {
            tracing::warn!(%error, "publish costmap");
        }

        processed += 1;
        if processed.is_multiple_of(LOG_EVERY_N_FRAMES) {
            let fps = LOG_EVERY_N_FRAMES as f64 / last_log.elapsed().as_secs_f64();
            tracing::info!(
                frame_id,
                lanes = result.image_lanes.len(),
                ground_lanes = result.ground_lanes.len(),
                without_depth,
                fps = format!("{fps:.1}"),
                "processing"
            );
            last_log = Instant::now();
            without_depth = 0;
        }
    }

    tracing::info!("shutting down");
    Ok(())
}
