# lane-seg

Lane detection node for the Jetson. Reads the ZED left image and depth that
[rust-autonomy-sensors](https://github.com/cornellev/rust-autonomy-sensors)'
`zed-reader` publishes over Zenoh, runs ADNet, and publishes a lane costmap
laid out the way
[costmap-planner-rust](https://github.com/cornellev/costmap-planner-rust)
reads one.

```
zed/zed2/image_left_vga_bgra ─┐
zed/zed2/depth_vga_f32 ───────┴─> lane-seg ──> perception/lane_costmap
```

The same binary runs on a laptop (ONNX Runtime on CPU) and on the Jetson
(ONNX Runtime with TensorRT). You can develop and test without a Jetson or a ZED.

## Layout

```
src/
  main.rs          the node: Zenoh in, costmap out
  pipeline.rs      one frame -> lanes -> costmap (no transport; shared by everything)
  model.rs         ONNX Runtime sessions (CPU / CUDA / TensorRT)
  adnet.rs         ADNet's weight-free head logic, ported from SPGHead + the CUDA NMS
  preprocess.rs    BGRA -> ADNet input (crop sky, resize, BGR/255)
  geometry.rs      lane pixels -> ground metres (ZED depth, flat-road fallback)
  costmap.rs       rasterisation + wire format
  zed_frame.rs     ZED wire format, vendored from zed-reader (see file header)
  bin/
    zed_replay.rs        fake zed-reader: publishes image files on the ZED keys
    lane_seg_offline.rs  run one image through the pipeline, write PNGs
    costmap_sub.rs       print received costmaps (like zed-reader's debug_sub)
tools/
  export_adnet_onnx.py   ADNet checkpoint -> the two ONNX files
config/lane_seg.toml     every setting, with defaults
models/                  put the two .onnx files here
```

## Model files

The node loads **two** ONNX files:

| File | Inputs | Outputs |
|---|---|---|
| `adnet_backbone.onnx` | `img` `[1,3,H,W]`, BGR / 255 | `hm` `[1,1,h,w]`, `shape_hm` `[1,1,h,w]`, `batch_features` `[1,C,h',w']` |
| `adnet_roi_head.onnx` | `batch_features`, `prior_xs` `[1,K,72]` | `cls_logits` `[1,K,2]`, `reg` `[1,K,73]` |

The model is split in two because ADNet's start-point decode (a
data-dependent top-k) can't be exported. The Rust node does that step between
the two graphs.

**The Jetson export only produced the first file.** `SPGHead` still has
trained layers after the decode: the prior pooling conv, a Linear layer,
LayerNorm and the cls/reg layers. The second file holds those. Re-export both
with:

```sh
cd tools
pixi run python export_adnet_onnx.py \
  --adnet-repo /path/to/ADNet \
  --config /path/to/ADNet/configs/adnet/culane/resnet18_culane.py \
  --weights /path/to/your_checkpoint.pth \
  --out-dir ../models
```

The script runs on a laptop without CUDA, compiled mmcv ops or ADNet's NMS
extension. It rewrites the deformable conv with `grid_sample`, which ONNX
Runtime and TensorRT both support without a plugin, and checks the ONNX
output against PyTorch before it exits. The input size, anchor count (K) and
rows (72) are read from the ONNX files, so you don't need to change any Rust
code when you swap in a different model.

`models/` currently holds an export of ADNet's released **CULane ResNet-18**
weights (320×800), so everything works out of the box. Swap in your own
checkpoint when it's trained.

## Running on a laptop

```sh
cargo build --release

# one image -> overlay.png + costmap.png
./target/release/lane_seg_offline config/lane_seg.toml road.png out/

# full Zenoh loop, three terminals
./target/release/zed_replay --config config/lane_seg.toml frames/   # fake ZED, 30 FPS
./target/release/lane-seg config/lane_seg.toml
./target/release/costmap_sub
```

`zed_replay` publishes a flat-road depth image computed from `[camera]`. Pass
`--no-depth` to test the fallback used when `ZED_READER_DEPTH_MODE=none`.

## Running on the Jetson

1. Install an ONNX Runtime build that includes the TensorRT and CUDA execution
   providers and matches your JetPack version (for example, the
   `onnxruntime-gpu` wheel from the Jetson AI Lab index ships
   `libonnxruntime.so`).
2. Build against it at runtime instead of downloading a CPU build:
   ```sh
   cargo build --release --no-default-features --features ort-dynamic,tensorrt,cuda
   export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
   ```
3. Set `execution_provider = "tensorrt"` in the config. The first start builds
   the FP16 TensorRT engine and caches it in `trt_cache_dir`; later starts load
   the cached engine. There's no separate `trtexec` step.
4. Copy the camera's real intrinsics from
   `/usr/local/zed/settings/SN<serial>.conf` (`[LEFT_CAM_VGA]`) into `[camera]`.
   The defaults are generic ZED 2 values, and ground distances scale directly
   with `fx`.

## Costmap output

The node publishes on `perception/lane_costmap`, encoding
`application/vnd.cornellev.costmap`. Each message is a 48-byte header followed
by `size_x * size_y` bytes of cost, laid out like `nav2_msgs/Costmap`
(`data[row * size_x + col]`, `col` = +x forward, `row` = +y left).

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | ASCII magic `LCM1` |
| 4 | 1 | format version (`1`) |
| 5 | 3 | reserved, zero |
| 8 | 8 | ZED `frame_id`, `u64` |
| 16 | 8 | ZED image timestamp in ns, `u64` |
| 24 | 4 | `size_x`, `u32` |
| 28 | 4 | `size_y`, `u32` |
| 32 | 4 | resolution (m/cell), `f32` |
| 36 | 4 | `origin_x` (m), `f32` |
| 40 | 4 | `origin_y` (m), `f32` |
| 44 | 4 | reserved, zero |

The frame is REP-103 (x forward, y left), with its origin on the road below the
left lens. The grid is centred there, because the planner starts its search at
the grid centre and plans towards +col.

Costs use nav2's values:

| Cost | Meaning |
|---:|---|
| 0 | free: the ego lane corridor between the nearest line on each side |
| 254 | lethal: lane markings, `line_width_m` wide |
| 255 | unknown: everything else |

The planner treats any cost of 253 or higher as blocked, so it stays inside the
ego lane. If only one line is visible, the corridor uses the assumed lane
width. `Costmap::decode` in `costmap.rs` reads the format back.

The planner currently subscribes to ROS 2 `/costmap_raw`. To use this output it
needs a Zenoh subscriber that decodes this header into its `Grid`. The data
layout already matches, so nothing else has to change.

## Limitations

- ADNet finds lane lines only. It doesn't tell dashed lines from solid ones
  and doesn't detect drivable road, so all markings are treated as lethal.
- `cut_top_fraction` should roughly match the ZED mount's horizon. ADNet
  expects the sky cropped, the way CULane images were during training.
- `zed_frame.rs` is a copy of `zed-reader`'s codec, because that crate's
  `build.rs` requires the ZED SDK. If the ZED wire format changes, update the
  copy too, or move both onto a shared codec crate that doesn't need the SDK.

## Tests

```sh
cargo test
```

The tests cover the ADNet decode, anchors, NMS and lane trimming, the crop and
resize, ground projection with and without depth, the costmap corridor and
wire format, the ZED codec round trip, and a full frame-to-costmap run with a
fake detector. None of them need model files.
