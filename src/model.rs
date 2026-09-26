//! ADNet inference through ONNX Runtime.
//!
//! The same code runs on a laptop CPU and on the Jetson; only the execution
//! provider changes (`model.execution_provider` in the config). With
//! `tensorrt`, ONNX Runtime builds the TensorRT engine itself on first start
//! and caches it, so there is no separate `trtexec` / `.engine` step.
//!
//! Model contract (see `tools/export_adnet_onnx.py`):
//!
//! - backbone: `img` `[1, 3, H, W]` BGR/255 -> `hm` `[1, 1, h, w]` (logits),
//!   `shape_hm` `[1, 1, h, w]`, `batch_features` `[1, C, h', w']`
//! - roi head: `batch_features`, `prior_xs` `[1, K, S]` (pixels) ->
//!   `cls_logits` `[1, K, 2]`, `reg` `[1, K, S + 1]`
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use ort::{
    ep::ExecutionProviderDispatch,
    session::{Session, SessionOutputs},
    value::TensorRef,
};

use crate::adnet::{self, HeadGeometry, Lane};
use crate::config::{ExecutionProvider, ModelConfig};

/// Anything that turns a preprocessed image into lanes. The ONNX model is the
/// real one; tests substitute a fake.
pub trait LaneDetector {
    /// Model input `(width, height)`.
    fn input_size(&self) -> (usize, usize);
    /// `input` is `[3, height, width]` BGR / 255.
    fn detect(&mut self, input: &[f32]) -> Result<Vec<Lane>>;
}

pub struct AdnetOnnx {
    backbone: Session,
    roi_head: Session,
    geom: HeadGeometry,
    num_anchors: usize,
    img_input: String,
    conf_threshold: f32,
    nms_threshold: f32,
    max_lanes: usize,
    prior_xs: Vec<f32>,
}

fn open_session(path: &Path, config: &ModelConfig) -> Result<Session> {
    ensure!(
        path.is_file(),
        "model file {} not found (see README: drop the exported ONNX files there)",
        path.display()
    );
    let providers: Vec<ExecutionProviderDispatch> = match config.execution_provider {
        ExecutionProvider::Cpu => vec![],
        #[cfg(feature = "cuda")]
        ExecutionProvider::Cuda => vec![ort::ep::CUDA::default().build().error_on_failure()],
        #[cfg(feature = "tensorrt")]
        ExecutionProvider::Tensorrt => {
            std::fs::create_dir_all(&config.trt_cache_dir).ok();
            let cache = config.trt_cache_dir.display().to_string();
            vec![
                ort::ep::TensorRT::default()
                    .with_fp16(config.trt_fp16)
                    .with_engine_cache(true)
                    .with_engine_cache_path(&cache)
                    .with_timing_cache(true)
                    .with_timing_cache_path(&cache)
                    .build()
                    .error_on_failure(),
                // Anything TensorRT rejects falls back to CUDA, not the CPU.
                #[cfg(feature = "cuda")]
                ort::ep::CUDA::default().build(),
            ]
        }
        #[allow(unreachable_patterns)]
        other => bail!(
            "execution provider {other:?} needs lane-seg built with `--features {}`",
            format!("{other:?}").to_lowercase()
        ),
    };
    let session = Session::builder()?
        .with_execution_providers(providers)
        .map_err(|e| anyhow::anyhow!("register execution providers: {e}"))?
        .commit_from_file(path)
        .with_context(|| format!("load {}", path.display()))?;
    Ok(session)
}

fn shape_of(session: &Session, input: &str) -> Result<Vec<i64>> {
    let outlet = session
        .inputs()
        .iter()
        .find(|o| o.name() == input)
        .with_context(|| format!("model has no input `{input}`"))?;
    Ok(outlet
        .dtype()
        .tensor_shape()
        .with_context(|| format!("input `{input}` is not a tensor"))?
        .to_vec())
}

fn require_names(session: &Session, what: &str, inputs: &[&str], outputs: &[&str]) -> Result<()> {
    let have_in: Vec<&str> = session.inputs().iter().map(|o| o.name()).collect();
    let have_out: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    for name in inputs {
        ensure!(have_in.contains(name), "{what}: missing input `{name}` (has {have_in:?})");
    }
    for name in outputs {
        ensure!(have_out.contains(name), "{what}: missing output `{name}` (has {have_out:?})");
    }
    Ok(())
}

fn tensor<'a>(outputs: &'a SessionOutputs, name: &str) -> Result<(Vec<usize>, &'a [f32])> {
    let (shape, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .with_context(|| format!("output `{name}` is not an f32 tensor"))?;
    Ok((shape.iter().map(|&d| d as usize).collect(), data))
}

impl AdnetOnnx {
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let backbone = open_session(&config.backbone, config)?;
        let roi_head = open_session(&config.roi_head, config)?;

        let img_input = match backbone.inputs() {
            [only] => only.name().to_string(),
            other => bail!("backbone should have exactly one input, has {}", other.len()),
        };
        require_names(&backbone, "backbone", &[], &["hm", "shape_hm", "batch_features"])?;
        require_names(&roi_head, "roi head", &["batch_features", "prior_xs"], &["cls_logits", "reg"])?;

        let img_shape = shape_of(&backbone, &img_input)?;
        let prior_shape = shape_of(&roi_head, "prior_xs")?;
        let fixed = |s: &[i64]| s.iter().all(|&d| d > 0);
        ensure!(
            img_shape.len() == 4 && img_shape[1] == 3 && fixed(&img_shape),
            "backbone input must be a static [1, 3, H, W], got {img_shape:?}"
        );
        ensure!(
            prior_shape.len() == 3 && fixed(&prior_shape),
            "prior_xs must be a static [1, K, S], got {prior_shape:?}"
        );

        let geom = HeadGeometry {
            img_w: img_shape[3] as f32,
            img_h: img_shape[2] as f32,
            n_offsets: prior_shape[2] as usize,
        };
        let num_anchors = prior_shape[1] as usize;
        tracing::info!(
            input = format!("{}x{}", geom.img_w, geom.img_h),
            num_anchors,
            n_offsets = geom.n_offsets,
            provider = ?config.execution_provider,
            "loaded ADNet"
        );
        Ok(Self {
            backbone,
            roi_head,
            geom,
            num_anchors,
            img_input,
            conf_threshold: config.conf_threshold,
            nms_threshold: config.nms_threshold,
            max_lanes: config.max_lanes,
            prior_xs: Vec::with_capacity(num_anchors * geom.n_offsets),
        })
    }
}

impl LaneDetector for AdnetOnnx {
    fn input_size(&self) -> (usize, usize) {
        (self.geom.img_w as usize, self.geom.img_h as usize)
    }

    fn detect(&mut self, input: &[f32]) -> Result<Vec<Lane>> {
        let (w, h) = self.input_size();
        let image = TensorRef::from_array_view(([1usize, 3, h, w], input))?;
        let outputs = self
            .backbone
            .run(ort::inputs![self.img_input.as_str() => image])?;

        let (hm_shape, hm) = tensor(&outputs, "hm")?;
        let (shape_shape, shape_hm) = tensor(&outputs, "shape_hm")?;
        let (feat_shape, features) = tensor(&outputs, "batch_features")?;
        ensure!(hm_shape.len() == 4, "hm must be [1, 1, h, w], got {hm_shape:?}");
        let (hm_h, hm_w) = (hm_shape[2], hm_shape[3]);
        let plane = hm_h * hm_w;
        ensure!(
            shape_shape.len() == 4 && shape_shape[2..] == hm_shape[2..],
            "shape_hm {shape_shape:?} does not match hm {hm_shape:?}"
        );

        // decode_theta_xy -> generate_anchors_in_one
        let starts = adnet::decode_starts(
            &hm[..plane],
            &shape_hm[..plane],
            hm_h,
            hm_w,
            self.num_anchors,
            &self.geom,
        );
        ensure!(
            starts.len() == self.num_anchors,
            "heatmap has {plane} cells, fewer than the {} anchors the RoI head expects",
            self.num_anchors
        );
        let anchors: Vec<Vec<f32>> = starts
            .iter()
            .map(|s| adnet::anchor_xs(s, &self.geom))
            .collect();
        self.prior_xs.clear();
        self.prior_xs.extend(anchors.iter().flatten());

        let features = TensorRef::from_array_view((feat_shape, features))?;
        let prior_xs = TensorRef::from_array_view((
            [1usize, self.num_anchors, self.geom.n_offsets],
            self.prior_xs.as_slice(),
        ))?;
        let roi = self
            .roi_head
            .run(ort::inputs!["batch_features" => features, "prior_xs" => prior_xs])?;
        let (_, cls_logits) = tensor(&roi, "cls_logits")?;
        let (_, reg) = tensor(&roi, "reg")?;

        let proposals = adnet::build_proposals(&starts, &anchors, cls_logits, reg, &self.geom);
        Ok(adnet::postprocess(
            proposals,
            self.conf_threshold,
            self.nms_threshold,
            self.max_lanes,
            &self.geom,
        ))
    }
}
