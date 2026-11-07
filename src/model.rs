use anyhow::{Context, Result, bail, ensure};
use ort::{ep::ExecutionProviderDispatch, session::Session, value::TensorRef};

use crate::clrnet::{self, DecodeParams, Lane, N_PRIORS, ROW_LEN};
use crate::config::{ExecutionProvider, ModelConfig};

const INPUT_NAME: &str = "input";
const OUTPUT_NAME: &str = "predictions";

/// Anything that turns a preprocessed image into lanes (the real model, or a fake in tests).
pub trait LaneDetector {
    fn input_size(&self) -> (usize, usize); // (width, height)
    fn detect(&mut self, input: &[f32]) -> Result<Vec<Lane>>; // input: [3, H, W] BGR/255
}

pub struct ClrnetOnnx {
    session: Session,
    width: usize,
    height: usize,
    params: DecodeParams,
}

fn open_session(config: &ModelConfig) -> Result<Session> {
    ensure!(config.path.is_file(), "model file {} not found", config.path.display());
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
                #[cfg(feature = "cuda")]
                ort::ep::CUDA::default().build(), // anything TensorRT rejects falls back to CUDA
            ]
        }
        #[allow(unreachable_patterns)]
        other => bail!("{other:?} needs clrnet-lane built with that cargo feature"),
    };
    Session::builder()?
        .with_execution_providers(providers)
        .map_err(|e| anyhow::anyhow!("register execution providers: {e}"))?
        .commit_from_file(&config.path)
        .with_context(|| format!("load {}", config.path.display()))
}

impl ClrnetOnnx {
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let session = open_session(config)?;
        let shape = session
            .inputs()
            .iter()
            .find(|o| o.name() == INPUT_NAME)
            .and_then(|o| o.dtype().tensor_shape().map(|s| s.to_vec()))
            .context("model has no tensor input `input`")?;
        ensure!(
            shape.len() == 4 && shape[1] == 3 && shape[2] > 0 && shape[3] > 0,
            "input must be a static [1, 3, H, W], got {shape:?}"
        );
        Ok(Self {
            session,
            width: shape[3] as usize,
            height: shape[2] as usize,
            params: DecodeParams {
                conf_threshold: config.conf_threshold,
                nms_threshold: config.nms_threshold,
                max_lanes: config.max_lanes,
            },
        })
    }

    /// Raw [192 * 78] model output for one preprocessed [3, H, W] input.
    pub fn run_raw(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        ensure!(input.len() == 3 * self.width * self.height, "wrong input length");
        let image = TensorRef::from_array_view(([1usize, 3, self.height, self.width], input))?;
        let outputs = self.session.run(ort::inputs![INPUT_NAME => image])?;
        let (shape, data) = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;
        ensure!(data.len() == N_PRIORS * ROW_LEN, "predictions has shape {shape:?}");
        Ok(data.to_vec())
    }
}

impl LaneDetector for ClrnetOnnx {
    fn input_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
    fn detect(&mut self, input: &[f32]) -> Result<Vec<Lane>> {
        let raw = self.run_raw(input)?;
        clrnet::decode(&raw, &self.params)
    }
}