use burn::{
    backend::{Metal, wgpu::WgpuDevice},
    tensor::{Tensor, TensorData},
};
use std::{error::Error, path::Path, time::Instant};

mod imported {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/realesr-animevideov3_fp32.rs"
    ));
}

fn main() -> Result<(), Box<dyn Error>> {
    let frame = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .ok_or("expected PNG path")?;
    let reference_path = std::env::args_os().nth(2);
    if frame.is_dir() {
        return run_frames(&frame);
    }
    let image = image::open(frame)?.to_rgb8();
    let (width, height) = image.dimensions();
    assert_eq!((width, height), (1280, 720), "expected 720p RGB frame");
    let rgb = image.into_raw();
    let chw: Vec<f32> = (0..3)
        .flat_map(|channel| {
            rgb.chunks_exact(3)
                .map(move |pixel| pixel[channel] as f32 / 255.0)
        })
        .collect();

    let device = WgpuDevice::IntegratedGpu(0);
    let setup = burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &device,
        Default::default(),
    );
    eprintln!(
        "backend={:?} adapter={:?}",
        setup.backend,
        setup.adapter.get_info()
    );
    let started = Instant::now();
    let model = imported::Model::<Metal>::from_file(
        concat!(env!("OUT_DIR"), "/model/realesr-animevideov3_fp32.bpk"),
        &device,
    );
    eprintln!("model load: {:?}", started.elapsed());
    let input = Tensor::<Metal, 4>::from_data(
        TensorData::new(chw, [1, 3, height as usize, width as usize]),
        &device,
    );
    let started = Instant::now();
    let output = model.forward(input).into_data();
    assert_eq!(output.shape.dims(), [1, 3, 2880, 5120]);
    assert!(
        output
            .as_slice::<f32>()?
            .iter()
            .all(|value| value.is_finite())
    );
    eprintln!("one 720p frame, 4x output: {:?}", started.elapsed());
    if let Some(path) = reference_path {
        let reference = std::fs::read(path)?;
        let values = output.as_slice::<f32>()?;
        if reference.len() != values.len() * 4 {
            return Err("reference tensor has wrong size".into());
        }
        let (mut square_error, mut peak_error) = (0.0_f64, 0.0_f64);
        for (bytes, &value) in reference.chunks_exact(4).zip(values) {
            let reference = f32::from_ne_bytes(bytes.try_into()?);
            let error = (reference - value).abs() as f64;
            if !error.is_finite() {
                return Err("non-finite tensor difference".into());
            }
            square_error += error * error;
            peak_error = peak_error.max(error);
        }
        let mse = square_error / values.len() as f64;
        eprintln!(
            "ORT CPU reference: float mse={mse}, peak error={peak_error}, psnr (0..1)={} dB",
            -10.0 * mse.log10()
        );
    }
    Ok(())
}

fn run_frames(dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut frames = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if let Some(index) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix(".png"))
            .and_then(|stem| {
                stem.parse::<usize>()
                    .ok()
                    .filter(|index| index.to_string() == stem)
            })
        {
            frames.push((index, path));
        }
    }
    frames.sort_by_key(|(index, _)| *index);
    if frames.is_empty() {
        return Err("no numbered PNG frames (starting at 0.png)".into());
    }
    for (expected, (index, _)) in frames.iter().enumerate() {
        if *index != expected {
            return Err(format!("missing {expected}.png").into());
        }
    }

    let device = WgpuDevice::IntegratedGpu(0);
    let setup = burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &device,
        Default::default(),
    );
    eprintln!(
        "backend={:?} adapter={:?}",
        setup.backend,
        setup.adapter.get_info()
    );
    let started = Instant::now();
    let model = imported::Model::<Metal>::from_file(
        concat!(env!("OUT_DIR"), "/model/realesr-animevideov3_fp32.bpk"),
        &device,
    );
    eprintln!("model load: {:?}", started.elapsed());

    infer_frame(&model, &device, &frames[0].1)?;
    println!("frame,preprocess_ms,inference_ms,total_ms");
    for (index, path) in frames.into_iter().skip(1) {
        let (preprocess, inference, total) = infer_frame(&model, &device, &path)?;
        println!("{index},{preprocess:.3},{inference:.3},{total:.3}");
    }
    Ok(())
}

fn infer_frame(
    model: &imported::Model<Metal>,
    device: &WgpuDevice,
    path: &Path,
) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let frame_start = Instant::now();
    let image = image::open(path)?.to_rgb8();
    let (width, height) = image.dimensions();
    if (width, height) != (1280, 720) {
        return Err(format!("{}: expected 1280x720 RGB frame", path.display()).into());
    }
    let rgb = image.into_raw();
    let chw: Vec<f32> = (0..3)
        .flat_map(|channel| {
            rgb.chunks_exact(3)
                .map(move |pixel| pixel[channel] as f32 / 255.0)
        })
        .collect();
    let input = Tensor::<Metal, 4>::from_data(
        TensorData::new(chw, [1, 3, height as usize, width as usize]),
        device,
    );
    let inference_start = Instant::now();
    let output = model.forward(input).into_data();
    let inference = inference_start.elapsed().as_secs_f64() * 1000.0;
    if output.shape.dims() != [1, 3, 2880, 5120]
        || !output
            .as_slice::<f32>()?
            .iter()
            .all(|value| value.is_finite())
    {
        return Err(format!("{}: invalid 4x output", path.display()).into());
    }
    Ok((
        (inference_start - frame_start).as_secs_f64() * 1000.0,
        inference,
        frame_start.elapsed().as_secs_f64() * 1000.0,
    ))
}
