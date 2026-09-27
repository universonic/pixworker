use anyhow::{Context, Result, bail, ensure};
use candle_core::{Device, Module, Tensor, safetensors::Load};
use candle_nn::{Conv2d, activation::PReLU, conv::Conv2dConfig, ops::pixel_shuffle};
use safetensors::{Dtype, SafeTensors};
use std::{env, fs, path::Path, time::Instant};

struct Model {
    layers: Vec<(Conv2d, Option<PReLU>)>,
}

impl Model {
    fn load(path: &Path, device: &Device) -> Result<Self> {
        let bytes = fs::read(path)?;
        let weights = SafeTensors::deserialize(&bytes)?;
        ensure!(
            weights.len() == 53,
            "expected 53 weight tensors, got {}",
            weights.len()
        );
        let mut layers = Vec::with_capacity(18);
        for i in 0..18 {
            let index = i * 2;
            let (input, output) = (if i == 0 { 3 } else { 64 }, if i == 17 { 48 } else { 64 });
            let weight = load_weight(
                &weights,
                &format!("body.{index}.weight"),
                &[output, input, 3, 3],
                device,
            )?;
            let bias = load_weight(&weights, &format!("body.{index}.bias"), &[output], device)?;
            let conv = Conv2d::new(
                weight,
                Some(bias),
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
            );
            let activation = if i == 17 {
                None
            } else {
                let weight = load_weight(
                    &weights,
                    &format!("body.{}.weight", index + 1),
                    &[64],
                    device,
                )?;
                Some(PReLU::new(weight, false))
            };
            layers.push((conv, activation));
        }
        Ok(Self { layers })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for (conv, activation) in &self.layers {
            x = conv.forward(&x)?;
            if let Some(activation) = activation {
                x = activation.forward(&x)?;
            }
        }
        let (_, _, h, w) = input.dims4()?;
        Ok(pixel_shuffle(&x, 4)?.add(&input.upsample_nearest2d(h * 4, w * 4)?)?)
    }
}

fn load_weight(
    weights: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let view = weights
        .tensor(name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(
        view.dtype() == Dtype::F32 && view.shape() == shape,
        "{name}: expected F32 {shape:?}, got {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    Ok(view.load(device)?)
}

fn input(path: &Path, device: &Device) -> Result<Tensor> {
    let image = image::open(path)?.to_rgb8();
    let (width, height) = image.dimensions();
    ensure!(
        (width, height) == (1280, 720),
        "{}: expected 1280x720 PNG, got {width}x{height}",
        path.display()
    );
    let rgb = image.into_raw();
    let chw: Vec<f32> = (0..3)
        .flat_map(|channel| {
            rgb.chunks_exact(3)
                .map(move |pixel| pixel[channel] as f32 / 255.0)
        })
        .collect();
    Ok(Tensor::from_vec(chw, (1, 3, 720, 1280), device)?)
}

fn infer(model: &Model, device: &Device, frame: &Path) -> Result<(Vec<f32>, f64, f64, f64)> {
    let start = Instant::now();
    let x = input(frame, device)?;
    let preprocess = start.elapsed().as_secs_f64() * 1000.0;
    let run = Instant::now();
    let output = model.forward(&x)?;
    ensure!(
        output.device().same_device(device),
        "output left requested device"
    );
    ensure!(
        output.dims() == [1, 3, 2880, 5120],
        "{}: wrong output shape: {:?}",
        frame.display(),
        output.dims()
    );
    let values = output
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let inference = run.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "{}: non-finite output",
        frame.display()
    );
    Ok((
        values,
        preprocess,
        inference,
        start.elapsed().as_secs_f64() * 1000.0,
    ))
}

fn compare(values: &[f32], reference_path: &Path) -> Result<()> {
    let reference = fs::read(reference_path)?;
    ensure!(
        reference.len() == values.len() * 4,
        "reference size {} != output size {}",
        reference.len(),
        values.len() * 4
    );
    let (mut sum, mut max) = (0f64, 0f64);
    for (bytes, &value) in reference.chunks_exact(4).zip(values) {
        let expected = f32::from_le_bytes(bytes.try_into()?);
        let error = (expected as f64 - value as f64).abs();
        ensure!(error.is_finite(), "non-finite reference/difference");
        sum += error * error;
        max = max.max(error);
    }
    let mse = sum / values.len() as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    };
    eprintln!("ORT CPU reference: float mse={mse:.12e} max={max:.12e} psnr_0..1={psnr:.4} dB");
    ensure!(
        psnr >= 100.0 && max < 1e-3,
        "FP32 output fails equivalence gate (PSNR >= 100 dB and max < 1e-3); no benchmark ranking"
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let mode = args
        .next()
        .context("usage: native-candle keys|cpu|metal WEIGHTS [FRAME_OR_DIR REFERENCE]")?;
    let weights_path = args.next().context("expected FP32 safetensors path")?;
    if mode == "keys" {
        ensure!(args.next().is_none(), "keys only accepts WEIGHTS");
        let bytes = fs::read(weights_path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let mut names = tensors.names();
        names.sort_unstable();
        for name in names {
            let tensor = tensors.tensor(name)?;
            println!("{name} {:?} {:?}", tensor.dtype(), tensor.shape());
        }
        return Ok(());
    }
    let frame_path = args
        .next()
        .context("expected 1280x720 PNG or numbered PNG directory")?;
    let reference_path = args
        .next()
        .context("expected ORT CPU frame0.f32 reference")?;
    ensure!(args.next().is_none(), "too many arguments");
    let device = match mode.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        _ => bail!("backend must be cpu or metal"),
    };
    let start = Instant::now();
    let model = Model::load(Path::new(&weights_path), &device)?;
    eprintln!(
        "backend={mode} device={device:?} model_load_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    let path = Path::new(&frame_path);
    if path.is_dir() {
        let entries = fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
        let count = entries
            .iter()
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "png"))
            .count();
        ensure!(count >= 8, "expected at least 8 frames, got {count}");
        for i in 0..count {
            ensure!(path.join(format!("{i}.png")).is_file(), "missing {i}.png");
        }
        let (values, preprocess, inference, total) = infer(&model, &device, &path.join("0.png"))?;
        compare(&values, Path::new(&reference_path))?;
        eprintln!(
            "warmup frame=0 preprocess_ms={preprocess:.3} inference_sync_readback_ms={inference:.3} total_ms={total:.3}"
        );
        println!("frame,preprocess_ms,inference_sync_readback_ms,total_ms");
        for i in 1..count {
            let (_, preprocess, inference, total) =
                infer(&model, &device, &path.join(format!("{i}.png")))?;
            println!("{i},{preprocess:.3},{inference:.3},{total:.3}");
        }
    } else {
        let (values, preprocess, inference, total) = infer(&model, &device, path)?;
        compare(&values, Path::new(&reference_path))?;
        eprintln!(
            "frame={} shape=[1,3,2880,5120] preprocess_ms={preprocess:.3} inference_sync_readback_ms={inference:.3} total_ms={total:.3}",
            path.display()
        );
    }
    Ok(())
}
