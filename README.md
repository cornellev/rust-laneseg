# rust-laneseg

Lane detection node for the Jetson. Reads the ZED left image and depth that
[rust-autonomy-sensors](https://github.com/cornellev/rust-autonomy-sensors)'
`zed-reader` publishes over Zenoh, runs [CLRNet](https://github.com/Turoad/CLRNet)
(ResNet-18, CULane weights), and publishes a lane costmap in the same format as
the lidar occupancy grid that
[costmap-planner-rust](https://github.com/cornellev/costmap-planner-rust) reads.

```
zed/zed2/image_left_vga_bgra ─┐
zed/zed2/depth_vga_f32 ───────┴─> clrnet-lane ──> cev/perception/lane_costmap
```

It runs as the `lane-seg` node in
[rust-autonomy-stack](https://github.com/cornellev/rust-autonomy-stack), linked at
`nodes/lane-seg`. The same binary runs on the Jetson (ONNX Runtime with TensorRT)
and on a laptop (ONNX Runtime on CPU).

## Layout

```
src/
  main.rs        the node: Zenoh in, costmap out
  pipeline.rs    one frame -> lanes -> costmap (no transport)
  model.rs       ONNX Runtime session (CPU / CUDA / TensorRT)
  clrnet.rs      CLRNet decode: lane score, NMS (port of nms_kernel.cu), lane points
  preprocess.rs  BGRA -> CLRNet input (crop sky, bicubic resize, BGR/255)
  geometry.rs    lane pixels -> ground metres (ZED depth, flat-road fallback)
  costmap.rs     rasterisation + wire format
  zed_frame.rs   ZED wire format, vendored from zed-reader
  config.rs      lane_seg.toml
config/lane_seg.toml     every setting, with defaults
models/clrnet_r18.onnx   the model (committed, so a gitman clone is runnable)
tools/setup_jetson_runtime.sh   GPU runtime for the Jetson, no sudo
```

## Model

`models/clrnet_r18.onnx` is CLRNet's released CULane ResNet-18 checkpoint
(`culane_r18.pth` from the CLRNet releases page), exported with
`configs/clrnet/clr_resnet18_culane.py`.

| | |
|---|---|
| Input `input` | `[1, 3, 320, 800]`, BGR / 255, no mean subtraction |
| Output `predictions` | `[1, 192, 78]`: per prior, 2 class logits, start y, start x, theta, length, 72 x offsets |
| Crop | top 270 of 590 rows (`cut_top_fraction = 0.4576`), then bicubic resize to 800×320 |
| Decode | softmax score ≥ 0.4, lane NMS at 50 px, at most 4 lanes (CULane test settings) |

The input size is read from the ONNX file. NMS is not in the graph; `clrnet.rs`
does it. To swap in a fine-tuned model, export it the same way and replace the file.

The weights were trained on CULane's camera, so `cut_top_fraction` (the horizon)
needs tuning for the ZED mount.

## Configuration

The node reads, in order: its first argument, `$LANE_SEG_CONFIG`, or
`$CEV_CAR_DIR/lane_seg.toml` (the stack sets `CEV_CAR_DIR` to the car's
directory). Every key is optional; `config/lane_seg.toml` lists them with their
defaults. Relative paths are relative to the config file.

Copy the camera's real intrinsics from `/usr/local/zed/settings/SN<serial>.conf`
(`[LEFT_CAM_VGA]`) into `[camera]`. The defaults are generic ZED 2 values, and
ground distances scale directly with `fx`.

## Building

The default features are the Jetson build, because the stack builds every node
with `cargo build --release --package <pkg>` and no extra features:
`libonnxruntime.so` (with TensorRT and CUDA) is loaded at runtime from
`ORT_DYLIB_PATH`.

```sh
cargo build --release                       # Jetson / stack build
cargo build --release --no-default-features --features ort-download   # laptop, bundled CPU ONNX Runtime
cargo test --release
```

## Running on the Jetson

1. Install the GPU runtime (ONNX Runtime 1.24 with the TensorRT and CUDA
   execution providers, TensorRT 10.13, CUDA 13 libraries). On a Thor with
   JetPack 7 this needs no sudo:
   ```sh
   tools/setup_jetson_runtime.sh      # once, ~4.5 GB into ort-runtime/
   source ort-runtime/env.sh          # sets ORT_DYLIB_PATH and LD_LIBRARY_PATH
   ```
2. Set `execution_provider = "tensorrt"`, then run
   `./target/release/clrnet-lane config/lane_seg.toml`.
   The first start builds the FP16 TensorRT engine (several minutes) and caches
   it in `trt_cache_dir`; later starts take about a second.

On a Thor, TensorRT FP16 runs the model in about 2.7 ms per frame (CUDA alone:
about 5.6 ms).

## In rust-autonomy-stack

The stack's `gitman.yml` clones this repo and links its root at `nodes/lane-seg`.
`cars/<car>/car.toml` has a `lane-seg` node (`package = "clrnet-lane"`), and
`cars/<car>/lane_seg.toml` holds that car's settings.

`cargo deploy` ships the binary and the car directory only. On the car, once:

```sh
# from a checkout of this repo on the car
tools/setup_jetson_runtime.sh /opt/cev/ort-runtime
mkdir -p /opt/cev/models && cp models/clrnet_r18.onnx /opt/cev/models/
```

`cars/merceves/car.toml` points `ORT_DYLIB_PATH` and `LD_LIBRARY_PATH` at
`/opt/cev/ort-runtime`. The node ships with `enabled = false` on every car: the
stack stops all nodes when one exits, and this one exits if the runtime or model
is missing. Enable it once those are in place and `zed-reader` is running.

## Costmap output

Published on `cev/perception/lane_costmap`. Each message is a 32-byte header
followed by `size_x * size_y` bytes of cost, laid out like `nav2_msgs/Costmap`
(`data[row * size_x + col]`, `col` = +x forward, `row` = +y left). The header is
the one virtual-rgbd-sensor's occupancy grid uses, so costmap-planner-rust's
`decode_costmap` reads it unchanged.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | ZED image timestamp, seconds, `i32` |
| 4 | 4 | ZED image timestamp, nanoseconds, `u32` |
| 8 | 4 | `size_x`, `u32` |
| 12 | 4 | `size_y`, `u32` |
| 16 | 4 | resolution (m/cell), `f32` |
| 20 | 4 | `origin_x` (m), `f32` |
| 24 | 4 | `origin_y` (m), `f32` |
| 28 | 4 | reserved, zero |

The frame is REP-103 (x forward, y left), with its origin on the road below the
left lens, and the grid is centred there.

| Cost | Meaning |
|---:|---|
| 0 | free: the ego lane corridor between the nearest line on each side |
| 254 | lethal: lane markings, `line_width_m` wide |
| 255 | unknown: everything else |

The planner currently subscribes only to the lidar grid
(`cev/sensors/lidar/occupancy_grid/global`, in the odom frame). Using the lane
costmap there still needs a fusion step that moves it into the odom frame and
merges it with the lidar grid.
