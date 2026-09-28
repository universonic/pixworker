use anyhow::{Context, Result, ensure};
use std::{cell::RefCell, collections::HashMap, path::Path};
use tch::{Device, Kind, Tensor};

// Practical-RIFE v4.25 standard IFNet_HDv3.py / RIFE_HDv3.py: encode each RGB
// frame with 3->16 stride-2 conv, two 16->16 convs and 16->4 transpose conv.
// Five IFBlocks use (conv input, width) = (15,192), (28,128), (28,96),
// (28,64), (28,32), with eight residual convolutions each. At scale=1 the wrapper
// supplies [16,8,4,2,1], not IFNet.forward's default four scales. The
// timestep is a full-resolution constant channel; each block refines flow,
// replaces the mask/feature and warps both images/features. The final mask
// is sigmoid-blended; inference_video.py zero-pads right/bottom to multiples
// of 128, crops back, and the public result is clamped to [0,1]. The unused
// teacher/caltime checkpoint tensors are excluded from the inference weights.

struct Conv {
    weight: Tensor,
    bias: Tensor,
    stride: i64,
    transpose: bool,
}

impl Conv {
    fn load(
        weights: &mut HashMap<String, Tensor>,
        name: &str,
        shape: [i64; 4],
        stride: i64,
        transpose: bool,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: take(weights, &format!("{name}.weight"), &shape, device)?,
            bias: take(
                weights,
                &format!("{name}.bias"),
                &[if transpose { shape[1] } else { shape[0] }],
                device,
            )?,
            stride,
            transpose,
        })
    }

    fn run(&self, input: &Tensor) -> Result<Tensor> {
        if matches!(input.device(), Device::Cuda(_)) {
            Ok(input.f_internal_convolution(
                &self.weight,
                Some(&self.bias),
                [self.stride; 2],
                [1, 1],
                [1, 1],
                self.transpose,
                [0, 0],
                1,
                false,
                false,
                true,
                false,
            )?)
        } else if self.transpose {
            Ok(input.f_conv_transpose2d(
                &self.weight,
                Some(&self.bias),
                [self.stride; 2],
                [1, 1],
                [0, 0],
                1,
                [1, 1],
            )?)
        } else {
            Ok(input.f_conv2d(
                &self.weight,
                Some(&self.bias),
                [self.stride; 2],
                [1, 1],
                [1, 1],
                1,
            )?)
        }
    }
}

fn leaky(x: Tensor) -> Tensor {
    x.maximum(&(&x * 0.2))
}

fn take(
    weights: &mut HashMap<String, Tensor>,
    name: &str,
    shape: &[i64],
    device: Device,
) -> Result<Tensor> {
    let tensor = weights
        .remove(name)
        .with_context(|| format!("missing RIFE weight {name}"))?;
    ensure!(
        tensor.kind() == Kind::Float && tensor.size() == shape,
        "invalid RIFE weight {name}: {:?} {:?}, expected F32 {shape:?}",
        tensor.kind(),
        tensor.size()
    );
    Ok(tensor.to_device(device))
}

struct Block {
    down0: Conv,
    down1: Conv,
    residual: Vec<(Conv, Tensor)>,
    up: Conv,
}

impl Block {
    fn load(
        weights: &mut HashMap<String, Tensor>,
        index: usize,
        channels: i64,
        device: Device,
    ) -> Result<Self> {
        let name = format!("block{index}");
        let input = if index == 0 { 15 } else { 28 };
        let down0 = Conv::load(
            weights,
            &format!("{name}.conv0.0.0"),
            [channels / 2, input, 3, 3],
            2,
            false,
            device,
        )?;
        let down1 = Conv::load(
            weights,
            &format!("{name}.conv0.1.0"),
            [channels, channels / 2, 3, 3],
            2,
            false,
            device,
        )?;
        let mut residual = Vec::with_capacity(8);
        for i in 0..8 {
            let prefix = format!("{name}.convblock.{i}");
            residual.push((
                Conv::load(
                    weights,
                    &format!("{prefix}.conv"),
                    [channels, channels, 3, 3],
                    1,
                    false,
                    device,
                )?,
                take(
                    weights,
                    &format!("{prefix}.beta"),
                    &[1, channels, 1, 1],
                    device,
                )?,
            ));
        }
        let up = Conv::load(
            weights,
            &format!("{name}.lastconv.0"),
            [channels, 52, 4, 4],
            2,
            true,
            device,
        )?;
        Ok(Self {
            down0,
            down1,
            residual,
            up,
        })
    }

    fn run(
        &self,
        input: &Tensor,
        flow: Option<&Tensor>,
        scale: i64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let h = input.size()[2] / scale;
        let w = input.size()[3] / scale;
        let mut x = input.f_upsample_bilinear2d([h, w], false, None, None)?;
        if let Some(flow) = flow {
            let scaled = flow.f_upsample_bilinear2d([h, w], false, None, None)? / scale;
            x = Tensor::cat(&[x, scaled], 1);
        }
        x = leaky(self.down0.run(&x)?);
        x = leaky(self.down1.run(&x)?);
        for (conv, beta) in &self.residual {
            x = leaky(conv.run(&x)? * beta + x);
        }
        x = self.up.run(&x)?.pixel_shuffle(2);
        x = x.f_upsample_bilinear2d([input.size()[2], input.size()[3]], false, None, None)?;
        Ok((
            x.narrow(1, 0, 4) * scale,
            x.narrow(1, 4, 1),
            x.narrow(1, 5, 8),
        ))
    }
}

pub struct Rife {
    head: [Conv; 4],
    blocks: Vec<Block>,
    device: Device,
    grids: RefCell<HashMap<(i64, i64), Tensor>>,
}

impl Rife {
    pub fn new(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let mut weights: HashMap<_, _> = Tensor::read_safetensors(path)?.into_iter().collect();
        ensure!(
            weights.len() == 158,
            "expected 158 RIFE inference tensors, got {}",
            weights.len()
        );
        let head = [
            Conv::load(&mut weights, "encode.cnn0", [16, 3, 3, 3], 2, false, device)?,
            Conv::load(
                &mut weights,
                "encode.cnn1",
                [16, 16, 3, 3],
                1,
                false,
                device,
            )?,
            Conv::load(
                &mut weights,
                "encode.cnn2",
                [16, 16, 3, 3],
                1,
                false,
                device,
            )?,
            Conv::load(&mut weights, "encode.cnn3", [16, 4, 4, 4], 2, true, device)?,
        ];
        let mut blocks = Vec::with_capacity(5);
        for (i, channels) in [192, 128, 96, 64, 32].into_iter().enumerate() {
            blocks.push(Block::load(&mut weights, i, channels, device)?);
        }
        ensure!(
            weights.is_empty(),
            "unexpected RIFE weight names: {:?}",
            weights.keys().collect::<Vec<_>>()
        );
        Ok(Self {
            head,
            blocks,
            device,
            grids: RefCell::new(HashMap::new()),
        })
    }

    fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let mut x = image.shallow_clone();
        for conv in &self.head[..3] {
            x = leaky(conv.run(&x)?);
        }
        self.head[3].run(&x)
    }

    // warplayer.py: normalized linspace grid + pixel flow normalized by
    // (width-1)/2 and (height-1)/2; bilinear, border, align_corners=True.
    fn warp(&self, image: &Tensor, flow: &Tensor) -> Result<Tensor> {
        let (h, w) = (flow.size()[2], flow.size()[3]);
        if !self.grids.borrow().contains_key(&(h, w)) {
            let horizontal = Tensor::linspace(-1., 1., w, (Kind::Float, self.device))
                .view([1, 1, 1, w])
                .expand([1, 1, h, w], true);
            let vertical = Tensor::linspace(-1., 1., h, (Kind::Float, self.device))
                .view([1, 1, h, 1])
                .expand([1, 1, h, w], true);
            self.grids
                .borrow_mut()
                .insert((h, w), Tensor::cat(&[horizontal, vertical], 1));
        }
        let delta = Tensor::cat(
            &[
                flow.narrow(1, 0, 1) / ((w - 1) as f64 / 2.),
                flow.narrow(1, 1, 1) / ((h - 1) as f64 / 2.),
            ],
            1,
        );
        let grid = (self.grids.borrow()[&(h, w)].shallow_clone() + delta).permute([0, 2, 3, 1]);
        Ok(image.f_grid_sampler(&grid, 0, 1, true)?)
    }

    /// Interpolate device-resident [1,3,H,W] FP32 frames in [0,1] at arbitrary times.
    pub fn interpolate(&self, f0: &Tensor, f1: &Tensor, ts: &[f32]) -> Result<Vec<Tensor>> {
        let _guard = tch::no_grad_guard();
        let shape = f0.size();
        ensure!(
            shape.len() == 4 && shape[0] == 1 && shape[1] == 3 && shape[2] > 0 && shape[3] > 0,
            "expected [1,3,H,W] input, got {shape:?}"
        );
        ensure!(
            f1.size() == shape
                && f0.kind() == Kind::Float
                && f1.kind() == Kind::Float
                && f0.device() == self.device
                && f1.device() == self.device,
            "RIFE inputs must match shape/device and be FP32"
        );
        ensure!(
            ts.iter().all(|t| t.is_finite() && (0. ..=1.).contains(t)),
            "RIFE times must be finite and within [0,1]"
        );
        if ts.is_empty() {
            return Ok(Vec::new());
        }
        let (h, w) = (shape[2], shape[3]);
        let ph = ((h - 1) / 128 + 1) * 128;
        let pw = ((w - 1) / 128 + 1) * 128;
        let a = f0.f_constant_pad_nd([0, pw - w, 0, ph - h])?;
        let b = f1.f_constant_pad_nd([0, pw - w, 0, ph - h])?;
        let fa = self.encode(&a)?;
        let fb = self.encode(&b)?;
        let mut outputs = Vec::with_capacity(ts.len());
        for &t in ts {
            let time = Tensor::full([1, 1, ph, pw], t as f64, (Kind::Float, self.device));
            let mut flow: Option<Tensor> = None;
            let mut mask: Option<Tensor> = None;
            let mut feature: Option<Tensor> = None;
            let (mut wa, mut wb) = (a.shallow_clone(), b.shallow_clone());
            for (i, block) in self.blocks.iter().enumerate() {
                let input = if let Some(ref current) = flow {
                    let wf0 = self.warp(&fa, &current.narrow(1, 0, 2))?;
                    let wf1 = self.warp(&fb, &current.narrow(1, 2, 2))?;
                    Tensor::cat(
                        &[
                            &wa,
                            &wb,
                            &wf0,
                            &wf1,
                            &time,
                            mask.as_ref().unwrap(),
                            feature.as_ref().unwrap(),
                        ],
                        1,
                    )
                } else {
                    Tensor::cat(&[&a, &b, &fa, &fb, &time], 1)
                };
                let (delta, next_mask, next_feature) =
                    block.run(&input, flow.as_ref(), [16, 8, 4, 2, 1][i])?;
                flow = Some(match flow {
                    Some(old) => old + delta,
                    None => delta,
                });
                mask = Some(next_mask);
                feature = Some(next_feature);
                let current = flow.as_ref().unwrap();
                wa = self.warp(&a, &current.narrow(1, 0, 2))?;
                wb = self.warp(&b, &current.narrow(1, 2, 2))?;
            }
            let m = mask.unwrap().sigmoid();
            outputs.push(
                (wa * &m + wb * (1.0f64 - &m))
                    .narrow(2, 0, h)
                    .narrow(3, 0, w)
                    .clamp(0., 1.),
            );
        }
        Ok(outputs)
    }
}
