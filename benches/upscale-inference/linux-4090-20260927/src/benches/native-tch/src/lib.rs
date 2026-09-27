use anyhow::{Context, Result, ensure};
use safetensors::{SafeTensors, tensor::Dtype};
use std::path::Path;
use tch::{Device, Kind, Tensor};

pub const OUTPUT_SHAPE: [i64; 4] = [1, 3, 2880, 5120];

pub struct Conv {
    weight: Tensor,
    bias: Tensor,
    prelu: Option<Tensor>,
}

fn weight(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
    device: Device,
) -> Result<Tensor> {
    let view = tensors.tensor(name)?;
    ensure!(
        view.dtype() == Dtype::F32 && view.shape() == shape,
        "{name}: expected FP32 {shape:?}, got {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    Ok(Tensor::f_from_data_size(
        view.data(),
        &shape.iter().map(|&x| x as i64).collect::<Vec<_>>(),
        Kind::Float,
    )?
    .f_to_device(device)?)
}

pub fn load_model(path: &Path, device: Device) -> Result<Vec<Conv>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    ensure!(
        tensors.names().len() == 53,
        "expected 53 tensors, got {}",
        tensors.names().len()
    );
    let mut layers = Vec::with_capacity(18);
    for i in 0..18 {
        let n = 2 * i;
        let in_channels = if i == 0 { 3 } else { 64 };
        let out_channels = if i == 17 { 48 } else { 64 };
        layers.push(Conv {
            weight: weight(
                &tensors,
                &format!("body.{n}.weight"),
                &[out_channels, in_channels, 3, 3],
                device,
            )?,
            bias: weight(&tensors, &format!("body.{n}.bias"), &[out_channels], device)?,
            prelu: (i < 17)
                .then(|| weight(&tensors, &format!("body.{}.weight", n + 1), &[64], device))
                .transpose()?,
        });
    }
    println!("loaded 53 FP32 tensors, 18 conv + 17 PReLU, device={device:?}");
    Ok(layers)
}

pub fn load_frame(path: &Path, device: Device) -> Result<Tensor> {
    let image = image::open(path)?.to_rgb8();
    let (width, height) = image.dimensions();
    ensure!(
        (width, height) == (1280, 720),
        "{}: expected 1280x720, got {width}x{height}",
        path.display()
    );
    input_from_rgb(&image, device)
}

pub fn input_from_rgb(image: &image::RgbImage, device: Device) -> Result<Tensor> {
    let (width, height) = image.dimensions();
    ensure!(
        (width, height) == (1280, 720),
        "expected 1280x720 RGB frame"
    );
    let rgb = image.as_raw();
    let chw: Vec<f32> = (0..3)
        .flat_map(|c| {
            rgb.chunks_exact(3)
                .map(move |pixel| pixel[c] as f32 / 255.0)
        })
        .collect();
    Ok(Tensor::f_from_slice(&chw)?
        .f_view([1, 3, 720, 1280])?
        .f_to_device(device)?)
}

pub fn infer(layers: &[Conv], input: Tensor, readback: &mut [f32]) -> Result<()> {
    let residual = input.f_upsample_nearest2d([2880, 5120], None, None)?;
    let mut x = input;
    for layer in layers {
        x = if matches!(x.device(), Device::Cuda(_)) {
            x.f_internal_convolution(
                &layer.weight,
                Some(&layer.bias),
                [1, 1],
                [1, 1],
                [1, 1],
                false,
                [0, 0],
                1,
                false,
                false,
                true,
                false,
            )?
        } else {
            x.f_conv2d(&layer.weight, Some(&layer.bias), [1, 1], [1, 1], [1, 1], 1)?
        };
        if let Some(slope) = &layer.prelu {
            x = x.f_prelu(slope)?;
        }
    }
    // Match the ONNX reference's DepthToSpace CRD blocksize=4 explicitly.
    x = x
        .f_view([1, 3, 4, 4, 720, 1280])?
        .f_permute([0, 1, 4, 2, 5, 3])?
        .f_contiguous()?
        .f_view(OUTPUT_SHAPE)?
        .f_add(&residual)?;
    ensure!(
        x.size() == OUTPUT_SHAPE,
        "unexpected output shape: {:?}",
        x.size()
    );
    let cpu = x.f_to_device(Device::Cpu)?.f_contiguous()?;
    cpu.f_copy_data(readback, readback.len())?; // MPS -> CPU copy synchronizes the device.
    Ok(())
}

pub fn check_reference(output: &[f32], path: &Path) -> Result<()> {
    let reference = std::fs::read(path)?;
    ensure!(
        reference.len() == output.len() * 4,
        "reference length {} != {}",
        reference.len(),
        output.len() * 4
    );
    let (mut sum, mut max) = (0.0_f64, 0.0_f64);
    for (bytes, &value) in reference.chunks_exact(4).zip(output) {
        let error = (f32::from_le_bytes(bytes.try_into()?) - value).abs() as f64;
        ensure!(error.is_finite(), "non-finite output difference");
        sum += error * error;
        max = max.max(error);
    }
    let mse = sum / output.len() as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    };
    println!(
        "frame=0 shape={OUTPUT_SHAPE:?} float_mse={mse:.12e} max_abs={max:.12e} psnr_0_1_db={psnr:.3}"
    );
    ensure!(
        psnr >= 100.0 && max <= 0.001,
        "first frame is not equivalent to ORT CPU; timing disqualified"
    );
    Ok(())
}
