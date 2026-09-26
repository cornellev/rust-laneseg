//! Verification subscriber for the lane costmap, like zed-reader's debug_sub:
//! decodes each message and prints its size and cell counts.
//!
//!   COSTMAP_SUB_MAX=60 costmap_sub [key]
use anyhow::Result;
use lane_seg::costmap::{COSTMAP_KEY_EXPR, Costmap, FREE_SPACE, NO_INFORMATION};
use zenoh::{Wait, handlers::RingChannel};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();
    let key = std::env::args().nth(1).unwrap_or_else(|| COSTMAP_KEY_EXPR.into());
    let max: u64 = std::env::var("COSTMAP_SUB_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let session = zenoh::open(zenoh::Config::default())
        .wait()
        .map_err(|e| anyhow::anyhow!("open Zenoh session: {e}"))?;
    let subscriber = session
        .declare_subscriber(&key)
        .with(RingChannel::new(4))
        .wait()
        .map_err(|e| anyhow::anyhow!("subscribe {key}: {e}"))?;
    tracing::info!(key, "subscribed");

    for _ in 0..max {
        let sample = subscriber
            .recv()
            .map_err(|e| anyhow::anyhow!("receive costmap: {e}"))?;
        let (frame_id, timestamp_ns, map) = Costmap::decode(&sample.payload().to_bytes())?;
        let count = |pred: fn(u8) -> bool| map.data.iter().filter(|&&c| pred(c)).count();
        tracing::info!(
            frame_id,
            timestamp_ns,
            size = format!("{}x{}", map.size_x, map.size_y),
            resolution = map.resolution,
            free = count(|c| c == FREE_SPACE),
            lethal = count(|c| c >= 253 && c != NO_INFORMATION),
            unknown = count(|c| c == NO_INFORMATION),
            "received"
        );
    }
    Ok(())
}
