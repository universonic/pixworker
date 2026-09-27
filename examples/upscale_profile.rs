use anyhow::{Result, bail};
use ndarray::Array;
#[cfg(target_os = "linux")]
use ort::ep::CUDA;
use ort::ep::ExecutionProvider;
#[cfg(target_os = "macos")]
use ort::ep::{
    CoreML,
    coreml::{ComputeUnits, ModelFormat},
};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Instant;

fn load_input(frame: &Path) -> Result<Tensor<f32>> {
    let img = image::open(frame)?.to_rgb8();
    let (w, h) = img.dimensions();
    if (w, h) != (1280, 720) {
        bail!("expected 1280x720 PNG, got {w}x{h}");
    }
    let data = img.into_raw();
    Ok(Tensor::from_array(Array::from_shape_fn(
        (1, 3, h as usize, w as usize),
        |(_, c, y, x)| data[(y * w as usize + x) * 3 + c] as f32 / 255.0,
    ))?)
}

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let frame = PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow::anyhow!("expected PNG path or frame directory"))?,
    );
    let profile = PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow::anyhow!("expected profile prefix"))?,
    );
    let mode = args
        .next()
        .map(|arg| arg.to_string_lossy().into_owned())
        .unwrap_or_else(|| "default".into());
    if !["cpu", "default", "gpu", "ane", "mlprogram", "cuda"].contains(&mode.as_str()) {
        bail!("mode must be cpu, default, gpu, ane, mlprogram, or cuda");
    }
    let raw_path = args.next();
    let model = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("home directory not found"))?
        .join(".cache/pixworker/models/upscale/realesr-animevideov3_fp32.onnx");
    if !model.is_file() {
        bail!("cached model missing: {}", model.display());
    }

    let threads = thread::available_parallelism()?.get();
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_threads(if threads > 8 { 4 } else { 2 })
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if !frame.is_dir() {
        builder = builder
            .with_profiling(&profile)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    #[cfg(target_os = "macos")]
    if mode != "cpu" {
        let mut coreml = CoreML::default().with_subgraphs(true);
        coreml = match mode.as_str() {
            "gpu" => coreml.with_compute_units(ComputeUnits::CPUAndGPU),
            "ane" => coreml.with_compute_units(ComputeUnits::CPUAndNeuralEngine),
            "mlprogram" => coreml.with_model_format(ModelFormat::MLProgram),
            _ => coreml,
        };
        if coreml.is_available()? {
            coreml.register(&mut builder)?;
        } else {
            bail!("CoreML unavailable");
        }
    }
    #[cfg(target_os = "linux")]
    if mode != "cpu" {
        if mode != "cuda" && mode != "default" {
            bail!("on Linux use cpu, cuda, or default");
        }
        let cuda = CUDA::default();
        if !cuda.is_available()? {
            bail!("CUDA execution provider unavailable");
        }
        cuda.register(&mut builder)?;
    }
    let started = Instant::now();
    let mut session = builder.commit_from_file(model)?;
    println!("session: {:?}", started.elapsed());
    if frame.is_dir() {
        if raw_path.is_some() {
            bail!("directory mode does not accept raw output path");
        }
        let mut frames = std::fs::read_dir(&frame)?
            .map(|entry| {
                let path = entry?.path();
                let index = path
                    .file_stem()
                    .ok_or_else(|| anyhow::anyhow!("missing frame index"))?
                    .to_string_lossy()
                    .parse::<usize>()?;
                Ok((index, path))
            })
            .collect::<Result<Vec<_>>>()?;
        frames.sort_unstable_by_key(|(index, _)| *index);
        if frames.len() < 8
            || frames.iter().enumerate().any(|(n, (index, path))| {
                *index != n || path.file_name().unwrap() != format!("{n}.png").as_str()
            })
        {
            bail!("expected at least 8 consecutive frames named 0.png..N.png");
        }
        println!("frame,preprocess_ms,inference_ms,total_ms");
        for (index, path) in frames {
            let frame_start = Instant::now();
            let input = load_input(&path)?;
            let inference_start = Instant::now();
            let outputs = session.run(ort::inputs![input])?;
            if outputs[0].try_extract_array::<f32>()?.shape() != [1, 3, 2880, 5120] {
                bail!("frame {index}: unexpected output shape");
            }
            let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;
            if index > 0 {
                println!(
                    "{index},{:.3},{inference_ms:.3},{:.3}",
                    (inference_start - frame_start).as_secs_f64() * 1000.0,
                    frame_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        return Ok(());
    }
    let input = load_input(&frame)?;
    let started = Instant::now();
    let outputs = session.run(ort::inputs![input])?;
    println!("run: {:?}", started.elapsed());
    let array = outputs[0].try_extract_array::<f32>()?;
    println!("output shape: {:?}", array.shape());
    if let Some(path) = raw_path {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(bytemuck::cast_slice(
                array
                    .as_slice()
                    .ok_or_else(|| anyhow::anyhow!("output not contiguous"))?,
            ))?;
    }
    drop(outputs);
    println!("profile: {}", session.end_profiling()?);
    Ok(())
}
