use anyhow::{Context, Result, ensure};
use std::{collections::HashMap, path::Path};
use tch::{Device, Kind, Tensor};

type Weights = HashMap<String, Tensor>;
type Shapes = HashMap<String, Vec<i64>>;

struct Conv {
    weight: Tensor,
    bias: Tensor,
}

impl Conv {
    fn load(weights: &mut Weights, name: &str) -> Self {
        Self {
            weight: weights.remove(&format!("{name}.weight")).unwrap(),
            bias: weights.remove(&format!("{name}.bias")).unwrap(),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if matches!(x.device(), Device::Cuda(_)) {
            // The last argument disables TF32 for strict FP32 convolution on CUDA.
            Ok(x.f_internal_convolution(
                &self.weight,
                Some(&self.bias),
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
            )?)
        } else {
            Ok(x.f_conv2d(&self.weight, Some(&self.bias), [1, 1], [1, 1], [1, 1], 1)?)
        }
    }
}

struct Srvgg {
    layers: Vec<(Conv, Option<Tensor>)>,
}

struct Rrdb {
    first: Conv,
    body: Vec<Vec<Vec<Conv>>>,
    trunk: Conv,
    up1: Conv,
    up2: Conv,
    hr: Conv,
    last: Conv,
}

enum Network {
    Srvgg(Srvgg),
    Rrdb(Rrdb),
}

pub struct Upscaler {
    network: Network,
    passes: usize,
    device: Device,
    kind: Kind,
}

// Keys match --upscale-model; the second path is needed only for general-x4v3 DNI.
fn spec(key: &str) -> Result<(&'static str, Option<&'static str>, usize, usize, Kind)> {
    let spec = match key {
        "realesr-animevideov3" => (
            "realesr-animevideov3_fp32.safetensors",
            None,
            18,
            0,
            Kind::Float,
        ),
        "realesr-animevideov3-hf" => (
            "realesr-animevideov3_fp16.safetensors",
            None,
            18,
            0,
            Kind::Half,
        ),
        "realesr-generalx4v3" => (
            "realesr-general-x4v3_fp32.safetensors",
            Some("realesr-general-wdn-x4v3_fp32.safetensors"),
            34,
            0,
            Kind::Float,
        ),
        "realesr-generalx4v3-hf" => (
            "realesr-general-x4v3_fp16.safetensors",
            Some("realesr-general-wdn-x4v3_fp16.safetensors"),
            34,
            0,
            Kind::Half,
        ),
        "realesrgan-x4plus" | "realesrganx4plus" => (
            "RealESRGAN_x4plus_fp32.safetensors",
            None,
            0,
            23,
            Kind::Float,
        ),
        "realesrgan-x4plus-hf" | "realesrganx4plus-hf" => (
            "RealESRGAN_x4plus_fp16.safetensors",
            None,
            0,
            23,
            Kind::Half,
        ),
        "realesrgan-x4plus-anime" | "realesrganx4plus-anime" => (
            "RealESRGAN_x4plus_anime_6B_fp32.safetensors",
            None,
            0,
            6,
            Kind::Float,
        ),
        "realesrgan-x4plus-anime-hf" | "realesrganx4plus-anime-hf" => (
            "RealESRGAN_x4plus_anime_6B_fp16.safetensors",
            None,
            0,
            6,
            Kind::Half,
        ),
        _ => anyhow::bail!("unsupported upscale model: {key}"),
    };
    Ok(spec)
}

fn conv_shape(shapes: &mut Shapes, name: String, input: i64, output: i64) {
    shapes.insert(format!("{name}.weight"), vec![output, input, 3, 3]);
    shapes.insert(format!("{name}.bias"), vec![output]);
}

fn expected_shapes(layers: usize, blocks: usize) -> Shapes {
    let mut shapes = Shapes::new();
    if layers != 0 {
        for i in 0..layers {
            conv_shape(
                &mut shapes,
                format!("body.{}", 2 * i),
                if i == 0 { 3 } else { 64 },
                if i + 1 == layers { 48 } else { 64 },
            );
            if i + 1 < layers {
                shapes.insert(format!("body.{}.weight", 2 * i + 1), vec![64]);
            }
        }
    } else {
        conv_shape(&mut shapes, "conv_first".into(), 3, 64);
        for block in 0..blocks {
            for rdb in 1..=3 {
                for i in 1..=5 {
                    conv_shape(
                        &mut shapes,
                        format!("body.{block}.rdb{rdb}.conv{i}"),
                        64 + 32 * (i - 1),
                        if i == 5 { 64 } else { 32 },
                    );
                }
            }
        }
        for name in ["conv_body", "conv_up1", "conv_up2", "conv_hr"] {
            conv_shape(&mut shapes, name.into(), 64, 64);
        }
        conv_shape(&mut shapes, "conv_last".into(), 64, 3);
    }
    shapes
}

fn read_weights(path: &Path, shapes: &Shapes, kind: Kind) -> Result<Weights> {
    let mut weights = Weights::new();
    for (name, tensor) in
        Tensor::read_safetensors(path).with_context(|| format!("read {}", path.display()))?
    {
        ensure!(
            weights.insert(name.clone(), tensor).is_none(),
            "{}: duplicate tensor {name}",
            path.display()
        );
    }
    ensure!(
        weights.len() == shapes.len(),
        "{}: expected {} tensors, got {}",
        path.display(),
        shapes.len(),
        weights.len()
    );
    for (name, expected) in shapes {
        let tensor = weights
            .get(name)
            .with_context(|| format!("{}: missing tensor {name}", path.display()))?;
        ensure!(
            tensor.kind() == kind && tensor.size() == *expected,
            "{}: {name}: expected {kind:?} {expected:?}, got {:?} {:?}",
            path.display(),
            tensor.kind(),
            tensor.size()
        );
    }
    Ok(weights)
}

impl Upscaler {
    pub fn files(key: &str) -> Result<(&'static str, Option<&'static str>)> {
        let (file, wdn, _, _, _) = spec(key)?;
        Ok((file, wdn))
    }

    pub fn default_device() -> Device {
        if std::env::var("PIXWORKER_CPU_ONLY").as_deref() == Ok("1") {
            Device::Cpu
        } else if tch::utils::has_mps() {
            Device::Mps
        } else if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }

    pub fn new(
        key: &str,
        path: &Path,
        wdn_path: Option<&Path>,
        passes: usize,
        device: Device,
    ) -> Result<Self> {
        let (_, wdn, layers, blocks, file_kind) = spec(key)?;
        ensure!(
            wdn.is_some() == wdn_path.is_some(),
            "{key}: incorrect number of model paths"
        );
        let shapes = expected_shapes(layers, blocks);
        let mut weights = read_weights(path, &shapes, file_kind)?;
        if let Some(wdn_path) = wdn_path {
            let other = read_weights(wdn_path, &shapes, file_kind)?;
            // Interpolate before moving to the device, retaining FP32 precision for DNI.
            for (name, tensor) in &mut weights {
                let general = tensor.f_to_kind(Kind::Float)?;
                let denoise = other[name].f_to_kind(Kind::Float)?;
                *tensor = general
                    .f_mul_scalar(0.1)?
                    .f_add(&denoise.f_mul_scalar(0.9)?)?;
            }
        }
        let kind = if device == Device::Cpu {
            Kind::Float
        } else {
            file_kind
        };
        for tensor in weights.values_mut() {
            *tensor = tensor.f_to_kind(kind)?.f_to_device(device)?;
        }
        let network = if layers != 0 {
            Network::Srvgg(Srvgg {
                layers: (0..layers)
                    .map(|i| {
                        let n = 2 * i;
                        let conv = Conv::load(&mut weights, &format!("body.{n}"));
                        let prelu = (i + 1 < layers)
                            .then(|| weights.remove(&format!("body.{}.weight", n + 1)).unwrap());
                        (conv, prelu)
                    })
                    .collect(),
            })
        } else {
            let first = Conv::load(&mut weights, "conv_first");
            let body = (0..blocks)
                .map(|block| {
                    (1..=3)
                        .map(|rdb| {
                            (1..=5)
                                .map(|i| {
                                    Conv::load(
                                        &mut weights,
                                        &format!("body.{block}.rdb{rdb}.conv{i}"),
                                    )
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();
            Network::Rrdb(Rrdb {
                first,
                body,
                trunk: Conv::load(&mut weights, "conv_body"),
                up1: Conv::load(&mut weights, "conv_up1"),
                up2: Conv::load(&mut weights, "conv_up2"),
                hr: Conv::load(&mut weights, "conv_hr"),
                last: Conv::load(&mut weights, "conv_last"),
            })
        };
        debug_assert!(weights.is_empty());
        Ok(Self {
            network,
            passes,
            device,
            kind,
        })
    }

    pub fn run(&self, x: Tensor) -> Result<Tensor> {
        ensure!(
            x.device() == self.device
                && x.kind() == Kind::Float
                && x.size().len() == 4
                && x.size()[0] == 1
                && x.size()[1] == 3
                && x.size()[2] > 0
                && x.size()[3] > 0,
            "upscale input must be [1,3,H,W] Float on {:?}",
            self.device
        );
        let _guard = tch::no_grad_guard();
        let mut x = x.f_to_kind(self.kind)?;
        for _ in 0..self.passes {
            x = match &self.network {
                Network::Srvgg(model) => model.forward(&x)?,
                Network::Rrdb(model) => model.forward(&x)?,
            };
        }
        Ok(x.f_to_kind(Kind::Float)?
            .f_mul_scalar(255.0)?
            .f_clamp(0.0, 255.0)?
            .f_floor()?
            .f_to_kind(Kind::Uint8)?
            .f_squeeze_dim(0)?
            .f_permute([1, 2, 0])?
            .f_contiguous()?)
    }
}

impl Srvgg {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let size = input.size();
        let (h, w) = (size[2], size[3]);
        let mut x = input.shallow_clone();
        for (conv, prelu) in &self.layers {
            x = conv.forward(&x)?;
            if let Some(slope) = prelu {
                x = x.f_prelu(slope)?;
            }
        }
        // PixelShuffle(4), DepthToSpace CRD ordering.
        Ok(x.f_view([1, 3, 4, 4, h, w])?
            .f_permute([0, 1, 4, 2, 5, 3])?
            .f_contiguous()?
            .f_view([1, 3, h * 4, w * 4])?
            .f_add(&input.f_upsample_nearest2d([h * 4, w * 4], None, None)?)?)
    }
}

fn lrelu(x: Tensor) -> Result<Tensor> {
    Ok(x.f_maximum(&x.f_mul_scalar(0.2)?)?)
}

fn dense(input: &Tensor, convs: &[Conv]) -> Result<Tensor> {
    let mut features = vec![input.shallow_clone()];
    for conv in &convs[..4] {
        let x = lrelu(conv.forward(&Tensor::f_cat(&features, 1)?)?)?;
        features.push(x);
    }
    Ok(convs[4]
        .forward(&Tensor::f_cat(&features, 1)?)?
        .f_mul_scalar(0.2)?
        .f_add(input)?)
}

impl Rrdb {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let feat = self.first.forward(input)?;
        let mut body = feat.shallow_clone();
        for block in &self.body {
            let residual = body.shallow_clone();
            for rdb in block {
                body = dense(&body, rdb)?;
            }
            body = body.f_mul_scalar(0.2)?.f_add(&residual)?;
        }
        let mut x = feat.f_add(&self.trunk.forward(&body)?)?;
        for conv in [&self.up1, &self.up2] {
            let size = x.size();
            x = lrelu(conv.forward(&x.f_upsample_nearest2d(
                [size[2] * 2, size[3] * 2],
                None,
                None,
            )?)?)?;
        }
        self.last.forward(&lrelu(self.hr.forward(&x)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_schema() {
        for (key, count, secondary) in [
            ("realesr-animevideov3", 53, false),
            ("realesr-generalx4v3", 101, true),
            ("realesrgan-x4plus", 702, false),
            ("realesrgan-x4plus-anime", 192, false),
        ] {
            for key in [key.to_owned(), format!("{key}-hf")] {
                let (_, wdn, layers, blocks, _) = spec(&key).unwrap();
                assert_eq!(wdn.is_some(), secondary);
                assert_eq!(expected_shapes(layers, blocks).len(), count);
            }
        }
        assert!(spec("invalid").is_err());
    }

    #[test]
    fn cached_models() -> Result<()> {
        let Ok(dir) = std::env::var("PIXWORKER_UPSCALE_MODELS_DIR") else {
            return Ok(());
        };
        let dir = Path::new(&dir);
        let keys = [
            "realesr-animevideov3",
            "realesr-generalx4v3",
            "realesrgan-x4plus",
            "realesrgan-x4plus-anime",
        ];
        for key in keys {
            for key in [key.to_owned(), format!("{key}-hf")] {
                let (file, wdn) = Upscaler::files(&key)?;
                let model = Upscaler::new(
                    &key,
                    &dir.join(file),
                    wdn.map(|name| dir.join(name)).as_deref(),
                    1,
                    Device::Cpu,
                )?;
                let x = Tensor::arange(18, (Kind::Float, Device::Cpu)).view([1, 3, 2, 3]) / 17.0;
                let result = model.run(x)?;
                assert_eq!(result.size(), [8, 12, 3]);
                assert_eq!(result.kind(), Kind::Uint8);
                let mut bytes = vec![0u8; result.numel()];
                result.f_copy_data(&mut bytes, result.numel())?;
                let checksum: u64 = bytes
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u64 + 1) * v as u64)
                    .sum();
                println!("{key}: {checksum}");
            }
        }
        let path = dir.join("realesr-animevideov3_fp32.safetensors");
        assert!(Upscaler::new("realesr-animevideov3-hf", &path, None, 1, Device::Cpu).is_err());
        assert!(Upscaler::new("realesr-generalx4v3", &path, None, 1, Device::Cpu).is_err());
        let zero = Upscaler::new("realesr-animevideov3", &path, None, 0, Device::Cpu)?;
        let input =
            Tensor::f_from_slice(&[-0.1f32, 0.5, 1.1, 0.0, 1.0, 0.1])?.f_view([1, 3, 1, 2])?;
        let output = zero.run(input)?;
        let mut bytes = [0u8; 6];
        output.f_copy_data(&mut bytes, 6)?;
        assert_eq!(bytes, [0, 255, 255, 127, 0, 25]);
        let twice = Upscaler::new("realesr-animevideov3", &path, None, 2, Device::Cpu)?;
        assert_eq!(
            twice
                .run(Tensor::zeros([1, 3, 2, 3], (Kind::Float, Device::Cpu)))?
                .size(),
            [32, 48, 3]
        );
        if tch::utils::has_mps() {
            for key in ["realesr-animevideov3", "realesr-animevideov3-hf"] {
                let (file, _) = Upscaler::files(key)?;
                let model = Upscaler::new(key, &dir.join(file), None, 1, Device::Mps)?;
                let x = Tensor::zeros([1, 3, 2, 3], (Kind::Float, Device::Mps));
                let output = model.run(x)?;
                assert_eq!(output.size(), [8, 12, 3]);
                assert_eq!(output.kind(), Kind::Uint8);
                assert_eq!(output.device(), Device::Mps);
            }
        }
        Ok(())
    }
}
