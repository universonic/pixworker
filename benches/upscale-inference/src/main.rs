use anyhow::{Result, bail};
use std::env;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use tract::prelude::*;

fn load_input(path: &Path) -> Result<Tensor> {
    let img = image::open(path)?.to_rgb8();
    let (w, h) = img.dimensions();
    if (w, h) != (1280, 720) {
        bail!("expected 1280x720 PNG, got {w}x{h}");
    }
    let rgb = img.into_raw();
    let chw: Vec<f32> = (0..3)
        .flat_map(|c| {
            let rgb = &rgb;
            (0..h as usize * w as usize).map(move |pixel| rgb[pixel * 3 + c] as f32 / 255.0)
        })
        .collect();
    Ok(Tensor::from_slice(&[1, 3, h as usize, w as usize], &chw)?)
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("expected model path"))?;
    let frame_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("expected PNG path"))?;
    let device = args.next().unwrap_or_else(|| "default".to_owned());
    let profile_path = args.next();
    let reference_path = args.next();
    if device != "default" && device != "metal" {
        bail!("device must be default or metal");
    }
    let started = Instant::now();
    let model = tract::onnx()?.load(model_path)?.into_model()?;
    println!("import: {:?}", started.elapsed());
    let started = Instant::now();
    let runnable = tract::runtime_for_name(&device)?.prepare(model)?;
    println!("prepare ({device}): {:?}", started.elapsed());
    if Path::new(&frame_path).is_dir() {
        if profile_path.is_some() || reference_path.is_some() {
            bail!("directory mode does not accept a profile or reference path");
        }
        let mut frames = std::fs::read_dir(&frame_path)?
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
        let mut state = runnable.spawn_state()?;
        println!("frame,preprocess_ms,inference_ms,total_ms");
        for (index, path) in frames {
            let frame_start = Instant::now();
            let input = load_input(&path)?;
            let inference_start = Instant::now();
            let outputs = state.run([input])?;
            if outputs[0].shape()? != [1, 3, 2880, 5120] {
                bail!("frame {index}: unexpected output shape");
            }
            let _ = outputs[0].as_slice::<f32>()?;
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
    let input = load_input(Path::new(&frame_path))?;
    let started = Instant::now();
    let outputs = runnable.run([input.clone()])?;
    println!("first run: {:?}", started.elapsed());
    println!("output shape: {:?}", outputs[0].shape()?);
    if outputs[0].shape()? != [1, 3, 2880, 5120] {
        bail!("unexpected 4x output shape");
    }
    if let Some(path) = reference_path {
        let reference = std::fs::read(path)?;
        let output = outputs[0].as_slice::<f32>()?;
        if reference.len() != output.len() * 4 {
            bail!("reference tensor has wrong size");
        }
        let (mut square_error, mut peak_error) = (0.0_f64, 0.0_f64);
        for (bytes, &value) in reference.chunks_exact(4).zip(output) {
            let reference = f32::from_ne_bytes(bytes.try_into()?);
            let error = (reference - value).abs() as f64;
            if !error.is_finite() {
                bail!("non-finite tensor difference");
            }
            square_error += error * error;
            peak_error = peak_error.max(error);
        }
        let mse = square_error / output.len() as f64;
        println!(
            "float mse: {mse}, peak error: {peak_error}, psnr (0..1): {} dB",
            -10.0 * mse.log10()
        );
    }
    drop(outputs);
    if let Some(path) = profile_path {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(runnable.profile_json(Some([input]))?.as_bytes())?;
    }
    Ok(())
}
