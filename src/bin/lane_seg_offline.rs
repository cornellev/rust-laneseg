//! Run the full pipeline on one image file, no Zenoh or camera needed. The
//! quickest check that dropped-in ONNX files load and find lanes.
//!
//!   lane_seg_offline <config.toml> <image> [out_dir]
//!
//! Writes `<out_dir>/overlay.png` (lanes on the image) and
//! `<out_dir>/costmap.png` (forward is up, white free, black lethal, grey unknown).
use std::path::PathBuf;

use anyhow::{Context, Result};
use image::{Rgb, RgbImage, imageops::FilterType};
use lane_seg::{
    config::Config,
    costmap::{FREE_SPACE, NO_INFORMATION},
    model::AdnetOnnx,
    pipeline::LanePipeline,
    zed_frame::{HEIGHT, WIDTH},
};

const COLOURS: [[u8; 3]; 6] = [
    [230, 60, 60],
    [60, 200, 60],
    [60, 120, 240],
    [240, 200, 40],
    [200, 60, 220],
    [40, 210, 210],
];

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let usage = "usage: lane_seg_offline <config.toml> <image> [out_dir]";
    let config = Config::load(Some(&args.next().context(usage)?))?;
    let image_path = args.next().context(usage)?;
    let out_dir = args.next().unwrap_or_else(|| PathBuf::from("."));

    let mut rgb = image::ImageReader::open(&image_path)?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("open {}", image_path.display()))?
        .resize_exact(WIDTH as u32, HEIGHT as u32, FilterType::Triangle)
        .to_rgb8();
    let bgra: Vec<u8> = rgb.pixels().flat_map(|p| [p[2], p[1], p[0], 255]).collect();

    let mut pipeline = LanePipeline::new(AdnetOnnx::load(&config.model)?, &config);
    let started = std::time::Instant::now();
    let result = pipeline.process(&bgra, None)?;
    println!(
        "{} lanes ({} projected to ground) in {:.1} ms",
        result.image_lanes.len(),
        result.ground_lanes.len(),
        started.elapsed().as_secs_f64() * 1e3
    );

    for (lane, colour) in result.image_lanes.iter().zip(COLOURS.iter().cycle()) {
        for &(u, v) in lane {
            for (du, dv) in (-2i32..=2).flat_map(|a| (-2i32..=2).map(move |b| (a, b))) {
                let (x, y) = (u as i32 + du, v as i32 + dv);
                if x >= 0 && y >= 0 && (x as u32) < rgb.width() && (y as u32) < rgb.height() {
                    rgb.put_pixel(x as u32, y as u32, Rgb(*colour));
                }
            }
        }
    }
    std::fs::create_dir_all(&out_dir)?;
    rgb.save(out_dir.join("overlay.png"))?;

    // Rotate so forward (+col) is up and left (+row) is left.
    let map = &result.costmap;
    let costmap = RgbImage::from_fn(map.size_y, map.size_x, |px, py| {
        let (row, col) = ((map.size_y - 1 - px) as usize, (map.size_x - 1 - py) as usize);
        let v = match map.at(row, col) {
            FREE_SPACE => 255,
            NO_INFORMATION => 128,
            _ => 0,
        };
        Rgb([v, v, v])
    });
    costmap.save(out_dir.join("costmap.png"))?;
    println!("wrote {}/overlay.png and costmap.png", out_dir.display());
    Ok(())
}
