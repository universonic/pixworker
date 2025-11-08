use anyhow::{Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;

/// Real-ESRGAN model wrapper supporting Safetensors weights and inference
pub struct RealESRGAN {
    model: Box<dyn Module>,
    dtype: DType,
}

impl RealESRGAN {
    /// Create RealESRGAN wrapper
    pub fn new(model: Box<dyn Module>, dtype: DType) -> Self {
        RealESRGAN { model, dtype }
    }

    /// Run inference on a frame tensor [B, C, H, W].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.model.forward(x).map_err(Error::from)
    }

    /// Inference on Tensor input with scaling
    ///
    /// # Arguments
    /// * `input` - Input tensor [C, H, W] with f16/f32 values [0.0, 1.0] (normalized)
    /// * `scale_factor` - Target upscaling factor (4x model can apply multiple times)
    ///
    /// # Returns
    /// Output tensor [C, 4H, 4W] (for one pass) with same dtype, values [0.0, 1.0]
    pub fn inference(&self, input: &Tensor, scale_factor: &f64) -> Result<Tensor> {
        // All Real-ESRGAN models are 4x upscale models
        const MODEL_SCALE: f64 = 4.0;

        // Determine how many times we need to apply 4x upscaling
        let num_upscale_passes = if *scale_factor <= 1.0 {
            0
        } else if *scale_factor <= MODEL_SCALE {
            1
        } else {
            (scale_factor.log(MODEL_SCALE).ceil() as usize).max(1)
        };

        // Convert input to model's dtype if needed
        let mut result = if input.dtype() != self.dtype {
            input.to_dtype(self.dtype)?
        } else {
            input.to_owned()
        };

        // Apply upscaling multiple times if needed
        for _ in 0..num_upscale_passes {
            // Add batch dimension: [C, H, W] -> [1, C, H, W]
            let batched = result.unsqueeze(0).map_err(|e| anyhow::anyhow!("Failed to unsqueeze: {}", e))?;

            // Run through model
            let output_batched = self.model.forward(&batched)?;

            // Remove batch: [1, C, 4H, 4W] -> [C, 4H, 4W]
            result = output_batched.squeeze(0).map_err(|e| anyhow::anyhow!("Failed to squeeze: {}", e))?;
        }
        Ok(result)
    }
}

/// Blend two frames with a balanced factor
///
/// # Arguments
/// * `frame1` - First frame to blend as Tensor [C, H, W] with f16/f32 values [0.0, 1.0]
/// * `frame2` - Second frame to blend as Tensor [C, H, W] with f16/f32 values [0.0, 1.0]
/// * `factor` - Blend factor in range [0.0, 1.0]
///   - 0.0: returns frame1 unchanged
///   - 1.0: returns frame2 unchanged
///   - 0.5: equal blend of both frames
///
/// # Returns
/// Blended frame: frame1 * (1 - factor) + frame2 * factor (same dtype as inputs)
pub fn blend_balanced_frame(frame1: &Tensor, frame2: &Tensor, factor: f32) -> Result<Tensor> {
    // Clamp factor to valid range [0.0, 1.0]
    let blend_factor = factor.clamp(0.0, 1.0);
    let inv_factor = 1.0 - blend_factor;

    // Use broadcasting_mul for proper element-wise scalar multiplication
    // Create scalar tensors with proper shape for broadcasting
    let inv_factor_t = Tensor::full(inv_factor as f32, frame1.shape(), &frame1.device())
        .map_err(|e| anyhow::anyhow!("Failed to create inv_factor tensor: {}", e))?
        .to_dtype(frame1.dtype())?;
    let blend_factor_t = Tensor::full(blend_factor as f32, frame2.shape(), &frame2.device())
        .map_err(|e| anyhow::anyhow!("Failed to create blend_factor tensor: {}", e))?
        .to_dtype(frame2.dtype())?;

    // Element-wise blend: frame1 * (1 - factor) + frame2 * factor
    let weighted1 = frame1
        .mul(&inv_factor_t)
        .map_err(|e| anyhow::anyhow!("Failed to scale frame1: {}", e))?;
    let weighted2 = frame2
        .mul(&blend_factor_t)
        .map_err(|e| anyhow::anyhow!("Failed to scale frame2: {}", e))?;
    let blended =
        (&weighted1 + &weighted2).map_err(|e| anyhow::anyhow!("Failed to blend frames: {}", e))?;

    Ok(blended)
}

/// Helper to load a safetensors file into the weight HashMap.
fn load_weights<P: AsRef<Path>>(
    path: P,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut file = fs::File::open(path.as_ref())?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let safetensors = SafeTensors::deserialize(buffer.as_slice())?;

    let mut weights_map = HashMap::new();
    for key in safetensors.names() {
        let tensor_data = safetensors.tensor(key)?;
        let shape: Vec<usize> = tensor_data.shape().iter().map(|&s| s as usize).collect();
        let data_bytes = tensor_data.data();

        let tensor = match (tensor_data.dtype(), dtype) {
            (safetensors::Dtype::F32, DType::F32) => {
                let data: Vec<f32> = bytemuck::cast_slice(data_bytes).to_vec();
                Tensor::from_vec(data, shape, device)?
            }
            (safetensors::Dtype::F16, DType::F16) => {
                let data: Vec<half::f16> = bytemuck::cast_slice(data_bytes).to_vec();
                Tensor::from_vec(data, shape, device)?
            }
            (safetensors::Dtype::F16, DType::F32) => {
                let data_f16: Vec<half::f16> = bytemuck::cast_slice(data_bytes).to_vec();
                let data_f32: Vec<f32> = data_f16.iter().map(|x| x.to_f32()).collect();
                Tensor::from_vec(data_f32, shape, device)?
            }
            (safetensors::Dtype::F32, DType::F16) => {
                let data_f32: Vec<f32> = bytemuck::cast_slice(data_bytes).to_vec();
                let tensor_f32 = Tensor::from_vec(data_f32, shape, device)?;
                tensor_f32.to_dtype(DType::F16)?
            }
            _ => return Err(anyhow::anyhow!("Unsupported dtype combination")),
        };

        weights_map.insert(key.to_string(), tensor);
    }

    Ok(weights_map)
}

// ============================================================================
// LAYER IMPLEMENTATIONS - Templates for Candle-based Real-ESRGAN
// ============================================================================

/// Configuration constants for Real-ESRGAN models
pub mod config {
    /// RRDBNet (23-block) configuration
    pub const RRDBNET_23: RRDBNetConfig = RRDBNetConfig {
        num_in_ch: 3,
        num_out_ch: 3,
        num_feat: 64,
        num_block: 23,
        num_grow_ch: 32,
        scale: 4,
    };

    /// RRDBNet (6-block anime) configuration
    pub const RRDBNET_6: RRDBNetConfig = RRDBNetConfig {
        num_in_ch: 3,
        num_out_ch: 3,
        num_feat: 64,
        num_block: 6,
        num_grow_ch: 32,
        scale: 4,
    };

    /// SRVGGNetCompact (16-conv) configuration
    pub const SRVGG_16: SRVGGConfig = SRVGGConfig {
        num_in_ch: 3,
        num_out_ch: 3,
        num_feat: 64,
        num_conv: 16,
        upscale: 4,
        act_type: "prelu",
    };

    /// SRVGGNetCompact (32-conv) configuration
    pub const SRVGG_32: SRVGGConfig = SRVGGConfig {
        num_in_ch: 3,
        num_out_ch: 3,
        num_feat: 64,
        num_conv: 32,
        upscale: 4,
        act_type: "prelu",
    };

    #[derive(Clone, Copy, Debug)]
    pub struct RRDBNetConfig {
        pub num_in_ch: usize,
        pub num_out_ch: usize,
        pub num_feat: usize,
        pub num_block: usize,
        pub num_grow_ch: usize,
        pub scale: usize,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct SRVGGConfig {
        pub num_in_ch: usize,
        pub num_out_ch: usize,
        pub num_feat: usize,
        pub num_conv: usize,
        pub upscale: usize,
        pub act_type: &'static str,
    }
}

/// Trait for all layer types
pub trait Layer: Send + Sync {
    /// Forward pass
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

/// Conv2d layer - wrapper around Candle's Conv2d
pub struct Conv2d {
    inner: candle_nn::Conv2d,
}

impl Conv2d {
    /// Create Conv2d from weight and bias tensors
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: candle_nn::Conv2dConfig) -> Self {
        Conv2d {
            inner: candle_nn::Conv2d::new(weight, bias, config),
        }
    }
}

/// Helper to load Conv2d from weights HashMap
pub fn conv2d_from_weights(
    weights: &HashMap<String, Tensor>,
    weight_key: &str,
    bias_key: Option<&str>,
    padding: usize,
    stride: usize,
) -> Result<Conv2d> {
    let weight = weights
        .get(weight_key)
        .ok_or_else(|| anyhow::anyhow!("Weight key not found: {}", weight_key))?;

    let bias = bias_key.and_then(|k| weights.get(k).cloned());

    let config = candle_nn::Conv2dConfig {
        padding,
        stride,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };

    Ok(Conv2d::new(weight.to_owned(), bias, config))
}

/// Implement Layer trait for Conv2d
impl Layer for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        <candle_nn::Conv2d as candle_core::Module>::forward(&self.inner, x)
            .map_err(|e| anyhow::anyhow!("Conv2d forward failed: {}", e))
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

/// LeakyReLU activation - wrapper around Candle's leaky_relu function
pub struct LeakyReLU {
    negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        LeakyReLU { negative_slope }
    }
}

impl Layer for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use Candle's official leaky_relu function (GPU-accelerated)
        candle_nn::ops::leaky_relu(x, self.negative_slope as f64)
            .map_err(|e| anyhow::anyhow!("LeakyReLU failed: {}", e))
    }
}

/// Parametric ReLU activation - wrapper around Candle's official PReLU
pub struct PReLU {
    inner: candle_nn::activation::PReLU,
}

impl PReLU {
    /// Create PReLU with learnable slope parameter
    pub fn new(weight: Tensor) -> Self {
        PReLU {
            inner: candle_nn::activation::PReLU::new(weight, false),
        }
    }

    /// Load PReLU from weights HashMap
    pub fn from_weights(weights: &HashMap<String, Tensor>, weight_key: &str) -> Result<Self> {
        let weight = weights
            .get(weight_key)
            .ok_or_else(|| anyhow::anyhow!("PReLU weight not found: {}", weight_key))?
            .to_owned();
        Ok(PReLU::new(weight))
    }
}

impl Layer for PReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Delegate to Candle's PReLU implementation
        <candle_nn::activation::PReLU as candle_core::Module>::forward(&self.inner, x)
            .map_err(|e| anyhow::anyhow!("PReLU forward failed: {}", e))
    }
}

// ============================================================================
// Residual Dense Block (RDB)
// ============================================================================

/// ResidualDenseBlock as used in RRDB
/// Structure:
/// - conv1: Conv2d(num_feat, num_grow_ch, 3, 1, 1)
/// - conv2: Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
/// - conv3: Conv2d(num_feat + 2*num_grow_ch, num_grow_ch, 3, 1, 1)
/// - conv4: Conv2d(num_feat + 3*num_grow_ch, num_grow_ch, 3, 1, 1)
/// - conv5: Conv2d(num_feat + 4*num_grow_ch, num_feat, 3, 1, 1)
/// - activation: LeakyReLU(0.2)
pub struct ResidualDenseBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
    activation: LeakyReLU,
}

impl ResidualDenseBlock {
    /// Create RDB from separate conv layers
    pub fn new(conv1: Conv2d, conv2: Conv2d, conv3: Conv2d, conv4: Conv2d, conv5: Conv2d) -> Self {
        ResidualDenseBlock {
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            activation: LeakyReLU::new(0.2),
        }
    }

    /// Load RDB from weights HashMap
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
        rdb_idx: usize,
    ) -> Result<Self> {
        let conv1 = conv2d_from_weights(
            weights,
            &format!("body.{}.rdb{}.conv1.weight", block_idx, rdb_idx),
            Some(&format!("body.{}.rdb{}.conv1.bias", block_idx, rdb_idx)),
            1,
            1,
        )?;
        let conv2 = conv2d_from_weights(
            weights,
            &format!("body.{}.rdb{}.conv2.weight", block_idx, rdb_idx),
            Some(&format!("body.{}.rdb{}.conv2.bias", block_idx, rdb_idx)),
            1,
            1,
        )?;
        let conv3 = conv2d_from_weights(
            weights,
            &format!("body.{}.rdb{}.conv3.weight", block_idx, rdb_idx),
            Some(&format!("body.{}.rdb{}.conv3.bias", block_idx, rdb_idx)),
            1,
            1,
        )?;
        let conv4 = conv2d_from_weights(
            weights,
            &format!("body.{}.rdb{}.conv4.weight", block_idx, rdb_idx),
            Some(&format!("body.{}.rdb{}.conv4.bias", block_idx, rdb_idx)),
            1,
            1,
        )?;
        let conv5 = conv2d_from_weights(
            weights,
            &format!("body.{}.rdb{}.conv5.weight", block_idx, rdb_idx),
            Some(&format!("body.{}.rdb{}.conv5.bias", block_idx, rdb_idx)),
            1,
            1,
        )?;

        Ok(Self::new(conv1, conv2, conv3, conv4, conv5))
    }
}

impl Layer for ResidualDenseBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Forward logic:
        // x1 = activation(conv1(x))
        // x2 = activation(conv2(cat(x, x1, dim=1)))
        // x3 = activation(conv3(cat(x, x1, x2, dim=1)))
        // x4 = activation(conv4(cat(x, x1, x2, x3, dim=1)))
        // x5 = conv5(cat(x, x1, x2, x3, x4, dim=1))
        // return x + x5 * 0.2

        // x1 = activation(conv1(x))
        let x1 = self.conv1.forward(x)?;
        let x1 = self.activation.forward(&x1)?;

        // x2 = activation(conv2(cat(x, x1)))
        let x_x1 = Tensor::cat(&[x, &x1], 1)
            .map_err(|e| anyhow::anyhow!("Failed to concatenate x and x1: {}", e))?;
        let x2 = self.conv2.forward(&x_x1)?;
        let x2 = self.activation.forward(&x2)?;

        // x3 = activation(conv3(cat(x, x1, x2)))
        let x_x1_x2 = Tensor::cat(&[x, &x1, &x2], 1)
            .map_err(|e| anyhow::anyhow!("Failed to concatenate for x3: {}", e))?;
        let x3 = self.conv3.forward(&x_x1_x2)?;
        let x3 = self.activation.forward(&x3)?;

        // x4 = activation(conv4(cat(x, x1, x2, x3)))
        let x_x1_x2_x3 = Tensor::cat(&[x, &x1, &x2, &x3], 1)
            .map_err(|e| anyhow::anyhow!("Failed to concatenate for x4: {}", e))?;
        let x4 = self.conv4.forward(&x_x1_x2_x3)?;
        let x4 = self.activation.forward(&x4)?;

        // x5 = conv5(cat(x, x1, x2, x3, x4))
        let x_x1_x2_x3_x4 = Tensor::cat(&[x, &x1, &x2, &x3, &x4], 1)
            .map_err(|e| anyhow::anyhow!("Failed to concatenate for x5: {}", e))?;
        let x5 = self.conv5.forward(&x_x1_x2_x3_x4)?;

        // return x + x5 * 0.2
        let scaled_x5 = (x5 * 0.2).map_err(|e| anyhow::anyhow!("Failed to scale x5: {}", e))?;
        let output =
            (x + &scaled_x5).map_err(|e| anyhow::anyhow!("Failed to add residual: {}", e))?;

        Ok(output)
    }
}

// ============================================================================
// RRDB (Residual in Residual Dense Block)
// ============================================================================

/// RRDB: Residual in Residual Dense Block
/// Consists of 3 sequential ResidualDenseBlocks
pub struct RRDB {
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
}

impl RRDB {
    /// Create RRDB from three RDB blocks
    pub fn new(
        rdb1: ResidualDenseBlock,
        rdb2: ResidualDenseBlock,
        rdb3: ResidualDenseBlock,
    ) -> Self {
        RRDB { rdb1, rdb2, rdb3 }
    }

    /// Load RRDB from weights HashMap
    pub fn from_weights(weights: &HashMap<String, Tensor>, block_idx: usize) -> Result<Self> {
        let rdb1 = ResidualDenseBlock::from_weights(weights, block_idx, 1)?;
        let rdb2 = ResidualDenseBlock::from_weights(weights, block_idx, 2)?;
        let rdb3 = ResidualDenseBlock::from_weights(weights, block_idx, 3)?;
        Ok(Self::new(rdb1, rdb2, rdb3))
    }
}

impl Layer for RRDB {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Forward logic:
        // out = rdb1(x)
        // out = rdb2(out)
        // out = rdb3(out)
        // return x + out * 0.2

        let out = self.rdb1.forward(x)?;
        let out = self.rdb2.forward(&out)?;
        let out = self.rdb3.forward(&out)?;

        // Apply residual connection with 0.2 scaling
        let scaled_out =
            (out * 0.2).map_err(|e| anyhow::anyhow!("Failed to scale RRDB output: {}", e))?;
        let result =
            (x + &scaled_out).map_err(|e| anyhow::anyhow!("Failed to add RRDB residual: {}", e))?;

        Ok(result)
    }
}

// ============================================================================
// Pixel Shuffle (Sub-pixel Convolution)
// ============================================================================

/// PixelShuffle layer for upsampling
/// Rearranges tensor from (B, C*r*r, H, W) to (B, C, H*r, W*r)
/// Uses Candle's built-in pixel_shuffle for GPU-accelerated sub-pixel convolution
pub struct PixelShuffle {
    upscale_factor: usize,
}

impl PixelShuffle {
    pub fn new(upscale_factor: usize) -> Self {
        PixelShuffle { upscale_factor }
    }
}

impl Layer for PixelShuffle {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use Candle's built-in pixel_shuffle function
        // Rearranges (B, C*r*r, H, W) -> (B, C, H*r, W*r)
        candle_nn::ops::pixel_shuffle(x, self.upscale_factor)
            .map_err(|e| anyhow::anyhow!("PixelShuffle failed: {}", e))
    }
}

// ============================================================================
// RRDBNet Architecture
// ============================================================================

/// RRDBNet: Main super-resolution network
/// - Input: (B, 3, H, W)
/// - Output: (B, 3, scale*H, scale*W)
pub struct RRDBNet {
    conv_first: Conv2d,
    body: Vec<RRDB>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
    conv_hr: Conv2d,
    conv_last: Conv2d,
    lrelu: LeakyReLU,
}

impl RRDBNet {
    /// Load RRDBNet from weights HashMap
    pub fn from_weights(weights: &HashMap<String, Tensor>, num_block: usize) -> Result<Self> {
        // Load front-end conv
        let conv_first =
            conv2d_from_weights(weights, "conv_first.weight", Some("conv_first.bias"), 1, 1)?;

        // Load body RRDB blocks
        let mut body = Vec::new();
        for i in 0..num_block {
            let rrdb = RRDB::from_weights(weights, i)?;
            body.push(rrdb);
        }

        // Load fusion layer
        let conv_body =
            conv2d_from_weights(weights, "conv_body.weight", Some("conv_body.bias"), 1, 1)?;

        // Load upsampling layers
        let conv_up1 =
            conv2d_from_weights(weights, "conv_up1.weight", Some("conv_up1.bias"), 1, 1)?;
        let conv_up2 =
            conv2d_from_weights(weights, "conv_up2.weight", Some("conv_up2.bias"), 1, 1)?;

        // Load HR processing layers
        let conv_hr = conv2d_from_weights(weights, "conv_hr.weight", Some("conv_hr.bias"), 1, 1)?;
        let conv_last =
            conv2d_from_weights(weights, "conv_last.weight", Some("conv_last.bias"), 1, 1)?;

        Ok(RRDBNet {
            conv_first,
            body,
            conv_body,
            conv_up1,
            conv_up2,
            conv_hr,
            conv_last,
            lrelu: LeakyReLU::new(0.2),
        })
    }
}

impl Layer for RRDBNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Forward logic:
        // 1. feat = conv_first(x)
        // 2. body_feat = conv_body(body(feat))
        // 3. feat = feat + body_feat
        // 4. feat = lrelu(conv_up1(interpolate(feat, 2)))
        // 5. feat = lrelu(conv_up2(interpolate(feat, 2)))
        // 6. out = conv_last(lrelu(conv_hr(feat)))

        // Step 1: First convolution
        let mut feat = self.conv_first.forward(x)?;

        // Step 2: Apply all RRDB blocks
        let mut body_out = feat.to_owned();
        for rrdb in &self.body {
            body_out = rrdb.forward(&body_out)?;
        }

        // Step 3: Fusion layer
        let body_feat = self.conv_body.forward(&body_out)?;
        feat = (&feat + &body_feat)
            .map_err(|e| anyhow::anyhow!("Failed to add body features: {}", e))?;

        // Step 4: First upsampling (2x) - using nearest interpolation + conv
        let feat_up1 = Self::interpolate_nearest(&feat, 2)?;
        let feat_up1 = self.conv_up1.forward(&feat_up1)?;
        let feat = self.lrelu.forward(&feat_up1)?;

        // Step 5: Second upsampling (2x) - total 4x
        let feat_up2 = Self::interpolate_nearest(&feat, 2)?;
        let feat_up2 = self.conv_up2.forward(&feat_up2)?;
        let feat = self.lrelu.forward(&feat_up2)?;

        // Step 6: HR processing and final convolution
        let feat_hr = self.conv_hr.forward(&feat)?;
        let feat_hr = self.lrelu.forward(&feat_hr)?;
        let out = self.conv_last.forward(&feat_hr)?;

        Ok(out)
    }
}

// Implement Module trait for RRDBNet to allow it to be stored in Box<dyn Module>
impl Module for RRDBNet {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        Layer::forward(self, x).map_err(candle_core::Error::msg)
    }
}

impl RRDBNet {
    /// Helper function for nearest neighbor interpolation (upsampling)
    /// Uses Candle's built-in upsample_nearest2d for GPU-accelerated upsampling
    fn interpolate_nearest(x: &Tensor, scale: usize) -> Result<Tensor> {
        let (_, _, h, w) = x
            .dims4()
            .map_err(|e| anyhow::anyhow!("Input must be 4D: {}", e))?;

        let out_h = h * scale;
        let out_w = w * scale;

        x.upsample_nearest2d(out_h, out_w)
            .map_err(|e| anyhow::anyhow!("Nearest neighbor upsample failed: {}", e))
    }
}

impl RRDBNet {
    /// Convenience loader: load a safetensors file and return a RealESRGAN wrapper
    pub fn from_model<P: AsRef<Path>>(path: P, device: Device, dtype: DType) -> Result<RealESRGAN> {
        // Load weights
        let weights_map = load_weights(path, &device, dtype)?;

        // Determine num_block as in detection logic
        let num_block = weights_map
            .keys()
            .filter(|k| k.starts_with("body.") && k.contains(".rdb1.conv1.weight"))
            .filter_map(|k| {
                k.split("body.")
                    .nth(1)?
                    .split(".")
                    .next()?
                    .parse::<usize>()
                    .ok()
            })
            .max()
            .ok_or_else(|| anyhow::anyhow!("Could not determine num_block for RRDBNet"))?
            + 1;

        let rrdb = RRDBNet::from_weights(&weights_map, num_block)?;
        let model: Box<dyn Module> = Box::new(rrdb);
        Ok(RealESRGAN { dtype, model })
    }
}

// ============================================================================
// SRVGGNetCompact Architecture
// ============================================================================

/// SRVGGNetCompact: Lightweight VGG-style super-resolution network
pub struct SRVGGNetCompact {
    body: Vec<Box<dyn Layer>>, // Alternating Conv2d and activation layers
    pixel_shuffle: PixelShuffle,
}

impl SRVGGNetCompact {
    /// Load SRVGGNetCompact from weights HashMap
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        num_conv: usize,
        upscale: usize,
        act_type: &str,
    ) -> Result<Self> {
        let mut body: Vec<Box<dyn Layer>> = Vec::new();

        // First conv layer: 3 -> num_feat
        let first_conv = conv2d_from_weights(weights, "body.0.weight", Some("body.0.bias"), 1, 1)?;
        body.push(Box::new(first_conv));

        // First activation
        let first_activation = Self::create_activation(weights, 1, act_type)?;
        body.push(first_activation);

        // Main body: num_conv times (Conv2d + activation)
        for i in 0..num_conv {
            let conv_idx = 2 + i * 2;
            let act_idx = conv_idx + 1;

            let conv = conv2d_from_weights(
                weights,
                &format!("body.{}.weight", conv_idx),
                Some(&format!("body.{}.bias", conv_idx)),
                1,
                1,
            )?;
            body.push(Box::new(conv));

            let activation = Self::create_activation(weights, act_idx, act_type)?;
            body.push(activation);
        }

        // Last conv: num_feat -> num_out_ch * upscale * upscale
        let last_conv_idx = 2 + num_conv * 2;
        let last_conv = conv2d_from_weights(
            weights,
            &format!("body.{}.weight", last_conv_idx),
            Some(&format!("body.{}.bias", last_conv_idx)),
            1,
            1,
        )?;
        body.push(Box::new(last_conv));

        let pixel_shuffle = PixelShuffle::new(upscale);

        Ok(SRVGGNetCompact {
            body,
            pixel_shuffle,
        })
    }

    /// Helper to create activation layer
    fn create_activation(
        weights: &HashMap<String, Tensor>,
        layer_idx: usize,
        act_type: &str,
    ) -> Result<Box<dyn Layer>> {
        match act_type {
            "relu" => {
                // ReLU has no parameters
                Ok(Box::new(ReLU))
            }
            "prelu" => {
                // PReLU has per-channel weight parameter
                let prelu = PReLU::from_weights(weights, &format!("body.{}.weight", layer_idx))?;
                Ok(Box::new(prelu))
            }
            "leakyrelu" => Ok(Box::new(LeakyReLU::new(0.1))),
            _ => Err(anyhow::anyhow!("Unknown activation type: {}", act_type)),
        }
    }
}

impl Layer for SRVGGNetCompact {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Forward logic:
        // 1. Apply all body layers sequentially
        // 2. out = pixel_shuffle(out)
        // 3. base = interpolate(x, scale_factor=upscale, mode='nearest')
        // 4. out = out + base  [residual]

        // Step 1: Apply all layers in body
        let mut out = x.to_owned();
        for layer in &self.body {
            out = layer.forward(&out)?;
        }

        // Step 2: Pixel shuffle upsampling
        out = self.pixel_shuffle.forward(&out)?;

        // Step 3: Prepare base image (nearest neighbor interpolation)
        let base = RRDBNet::interpolate_nearest(x, self.pixel_shuffle.upscale_factor)?;

        // Step 4: Add residual
        let output = (&out + &base)
            .map_err(|e| anyhow::anyhow!("Failed to add residual in SRVGG: {}", e))?;

        Ok(output)
    }
}

// Implement Module trait for SRVGGNetCompact to allow it to be stored in Box<dyn Module>
impl Module for SRVGGNetCompact {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        Layer::forward(self, x).map_err(candle_core::Error::msg)
    }
}

impl SRVGGNetCompact {
    /// Convenience loader: load a safetensors file and return a RealESRGAN wrapper
    pub fn from_model<P: AsRef<Path>>(path: P, device: Device, dtype: DType) -> Result<RealESRGAN> {
        // Load weights
        let weights_map = load_weights(path, &device, dtype)?;

        // Determine body range and compute num_conv from last conv index
        let max_body_idx = weights_map
            .keys()
            .filter(|k| k.starts_with("body.") && k.ends_with(".weight"))
            .filter_map(|k| {
                k.split("body.")
                    .nth(1)?
                    .split(".")
                    .next()?
                    .parse::<usize>()
                    .ok()
            })
            .max()
            .ok_or_else(|| anyhow::anyhow!("Could not determine body range for SRVGG"))?;

        if max_body_idx < 2 {
            return Err(anyhow::anyhow!("Invalid SRVGG body layout: too few layers"));
        }

        let num_conv = (max_body_idx - 2) / 2;

        // Detect activation type: check if body.1.weight exists and has shape (64,)
        let act_type = if let Some(activation_weight) = weights_map.get("body.1.weight") {
            let dims = activation_weight.shape().dims();
            if dims.len() == 1 && dims[0] == 64 {
                "prelu"
            } else {
                "relu"
            }
        } else {
            "relu"
        };

        // Detect output channels from last conv: body.{max_body_idx}.weight
        let last_conv_key = format!("body.{}.weight", max_body_idx);
        let out_channels = if let Some(last_weight) = weights_map.get(&last_conv_key) {
            last_weight.shape().dims()[0]
        } else {
            return Err(anyhow::anyhow!("Could not find last conv layer"));
        };

        // Determine upscale factor from output channels
        let upscale = if out_channels == 12 {
            2
        } else if out_channels == 48 {
            4
        } else {
            return Err(anyhow::anyhow!(
                "Unknown output channels {} for SRVGG",
                out_channels
            ));
        };

        let srvgg = SRVGGNetCompact::from_weights(&weights_map, num_conv, upscale, act_type)?;
        let model: Box<dyn Module> = Box::new(srvgg);
        Ok(RealESRGAN { dtype, model })
    }
}

/// Simple ReLU activation (no parameters)
pub struct ReLU;

impl Layer for ReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // relu(x) = max(0, x)
        // Using Candle's tensor operations (GPU-compatible)
        let zeros = Tensor::zeros_like(x)?;
        Tensor::where_cond(&x.ge(&zeros)?, x, &zeros)
            .map_err(|e| anyhow::anyhow!("ReLU failed: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::fs;
    use std::path::PathBuf;

    /// Download a model from HuggingFace if it doesn't exist locally
    fn download_model(url: &str, cache_path: &PathBuf) -> Result<PathBuf> {
        // Create cache directory
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // If already cached, return cached path
        if cache_path.exists() {
            eprintln!("✓ Using cached model: {}", cache_path.display());
            return Ok(cache_path.clone());
        }

        eprintln!("⏬ Downloading model from: {}", url);
        let response = reqwest::blocking::get(url)
            .map_err(|e| anyhow::anyhow!("Failed to download model: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Download failed with status {}: {}",
                response.status(),
                url
            ));
        }

        let bytes = response
            .bytes()
            .map_err(|e| anyhow::anyhow!("Failed to read response body: {}", e))?;

        eprintln!("💾 Saving model to: {}", cache_path.display());
        fs::write(&cache_path, &bytes)
            .map_err(|e| anyhow::anyhow!("Failed to write model file: {}", e))?;

        eprintln!(
            "✓ Model downloaded successfully ({} MB)",
            bytes.len() / 1024 / 1024
        );
        Ok(cache_path.clone())
    }

    /// Get the test cache directory for models
    fn get_test_cache_dir() -> Result<PathBuf> {
        if let Some(home_dir) = dirs::home_dir() {
            let cache_dir = home_dir
                .join(".cache")
                .join("pixworker")
                .join("models")
                .join("upscale");
            fs::create_dir_all(&cache_dir)?;
            Ok(cache_dir)
        } else {
            Err(anyhow::anyhow!("Could not determine home directory"))
        }
    }

    /// Download model and return path, using cache if available
    fn get_model_path(model_name: &str, url: &str) -> Result<PathBuf> {
        let cache_dir = get_test_cache_dir()?;
        let cache_path = cache_dir.join(model_name);
        download_model(url, &cache_path)
    }

    /// Helper to test model loading and small-scale inference
    fn test_model_loading_and_inference(
        url: &str,
        model_name: &str,
        model_filename: &str,
        _expected_blocks: Option<usize>,
        dtype: DType,
    ) -> Result<()> {
        eprintln!("\n============================================================");
        eprintln!("Testing: {} ({:?})", model_name, dtype);
        eprintln!("Download URL: {}", url);
        eprintln!("============================================================");

        let mut device = Device::Cpu;
        #[cfg(target_os = "macos")]
        {
            if candle_core::utils::metal_is_available() {
                println!("Using Metal device for acceleration");
                device = Device::new_metal(0)?;
            }
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            if candle_core::utils::cuda_is_available() {
                println!("Using CUDA device for acceleration");
                device = Device::new_cuda(0)?;
            }
        }

        // Step 1: Download model
        eprintln!("\nStep 1: Downloading/caching model...");
        let model_path = get_model_path(model_filename, url)?;

        // Step 2: Load model
        eprintln!("Step 2: Loading model with dtype {:?}...", dtype);
        let real_esrgan = if model_name.contains("RRDB") {
            RRDBNet::from_model(&model_path, device.to_owned(), dtype)?
        } else if model_name.contains("SRVGG") {
            SRVGGNetCompact::from_model(&model_path, device.to_owned(), dtype)?
        } else {
            return Err(anyhow::anyhow!("Unknown model architecture for test"));
        };

        // Step 3: Create small test input (1, 3, 64, 64) with matching dtype
        eprintln!(
            "\nStep 3: Creating test input tensor (1, 3, 64, 64) with dtype {:?}...",
            dtype
        );
        let input_f32_data: Vec<f32> = vec![0.5; 1 * 3 * 64 * 64];
        let mut input = match Tensor::from_vec(input_f32_data, (1, 3, 64, 64), &device) {
            Ok(t) => {
                eprintln!("✓ Input tensor created as F32");
                t
            }
            Err(e) => {
                eprintln!("✗ Failed to create input tensor: {}", e);
                return Err(anyhow::anyhow!("Tensor creation failed: {}", e));
            }
        };

        // Convert input to the same dtype as the model if needed
        if dtype != DType::F32 {
            input = input
                .to_dtype(dtype)
                .map_err(|e| anyhow::anyhow!("Failed to convert input to {:?}: {}", dtype, e))?;
            eprintln!("✓ Input tensor converted to {:?}", dtype);
        }

        // Step 4: Run inference
        eprintln!("\nStep 4: Running inference...");
        let output = match real_esrgan.forward(&input) {
            Ok(o) => {
                eprintln!("✓ Inference completed");
                o
            }
            Err(e) => {
                eprintln!("✗ Inference failed: {}", e);
                return Err(e);
            }
        };

        // Step 5: Validate output shape
        eprintln!("\nStep 5: Validating output shape...");
        let output_shape = output.shape();
        let expected_height = 64 * 4; // 4x upscaling
        let expected_width = 64 * 4;
        eprintln!("Output shape: {:?}", output_shape);
        eprintln!(
            "Expected shape: (1, 3, {}, {})",
            expected_height, expected_width
        );

        if let Ok((batch, channels, height, width)) = output.dims4() {
            if batch == 1 && channels == 3 && height == expected_height && width == expected_width {
                eprintln!("✓ Output shape is correct!");
            } else {
                eprintln!("✗ Output shape mismatch!");
                return Err(anyhow::anyhow!(
                    "Shape mismatch: got ({}, {}, {}, {}), expected (1, 3, {}, {})",
                    batch,
                    channels,
                    height,
                    width,
                    expected_height,
                    expected_width
                ));
            }
        } else {
            eprintln!("✗ Could not parse output dimensions");
            return Err(anyhow::anyhow!("Could not parse output dimensions"));
        }

        eprintln!("\n✓✓✓ {} test PASSED ✓✓✓\n", model_name);
        Ok(())
    }

    #[test]
    fn test_rrdbnet_23_full_fp32() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_fp32.safetensors",
            "RRDBNet (23-block, RealESRGAN_x4plus)",
            "RealESRGAN_x4plus_fp32.safetensors",
            Some(23),
            DType::F32,
        );

        if let Err(e) = result {
            panic!("RRDBNet 23-block test failed: {}", e);
        }
    }

    #[test]
    fn test_rrdbnet_6_full_fp32() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp32.safetensors",
            "RRDBNet (6-block anime, RealESRGAN_x4plus_anime_6B)",
            "RealESRGAN_x4plus_anime_6B_fp32.safetensors",
            Some(6),
            DType::F32,
        );

        if let Err(e) = result {
            panic!("RRDBNet 6-block test failed: {}", e);
        }
    }

    #[test]
    fn test_srvgg_16_full_fp32() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-animevideov3_fp32.safetensors",
            "SRVGGNetCompact (16-conv, realesr-animevideov3)",
            "realesr-animevideov3_fp32.safetensors",
            None,
            DType::F32,
        );

        if let Err(e) = result {
            panic!("SRVGG 16-conv test failed: {}", e);
        }
    }

    #[test]
    fn test_srvgg_32_full_fp32() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-x4v3_fp32.safetensors",
            "SRVGGNetCompact (32-conv, realesr-general-x4v3)",
            "realesr-general-x4v3_fp32.safetensors",
            None,
            DType::F32,
        );

        if let Err(e) = result {
            panic!("SRVGG 32-conv test failed: {}", e);
        }
    }

    #[test]
    fn test_srvgg_32_wdn_full_fp32() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-wdn-x4v3_fp32.safetensors",
            "SRVGGNetCompact (32-conv with denoise, realesr-general-wdn-x4v3)",
            "realesr-general-wdn-x4v3_fp32.safetensors",
            None,
            DType::F32,
        );

        if let Err(e) = result {
            panic!("SRVGG 32-conv WDN test failed: {}", e);
        }
    }

    // ===== F16 Tests =====

    #[test]
    fn test_rrdbnet_23_full_fp16() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_fp32.safetensors",
            "RRDBNet (23-block, RealESRGAN_x4plus)",
            "RealESRGAN_x4plus_fp32.safetensors",
            Some(23),
            DType::F16,
        );

        if let Err(e) = result {
            panic!("RRDBNet 23-block F16 test failed: {}", e);
        }
    }

    #[test]
    fn test_rrdbnet_6_full_fp16() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp32.safetensors",
            "RRDBNet (6-block anime, RealESRGAN_x4plus_anime_6B)",
            "RealESRGAN_x4plus_anime_6B_fp32.safetensors",
            Some(6),
            DType::F16,
        );

        if let Err(e) = result {
            panic!("RRDBNet 6-block F16 test failed: {}", e);
        }
    }

    #[test]
    fn test_srvgg_16_full_fp16() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-animevideov3_fp32.safetensors",
            "SRVGGNetCompact (16-conv, realesr-animevideov3)",
            "realesr-animevideov3_fp32.safetensors",
            None,
            DType::F16,
        );

        if let Err(e) = result {
            panic!("SRVGG 16-conv F16 test failed: {}", e);
        }
    }

    #[test]
    fn test_srvgg_32_full_fp16() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-x4v3_fp32.safetensors",
            "SRVGGNetCompact (32-conv, realesr-general-x4v3)",
            "realesr-general-x4v3_fp32.safetensors",
            None,
            DType::F16,
        );

        if let Err(e) = result {
            panic!("SRVGG 32-conv F16 test failed: {}", e);
        }
    }

    #[test]
    fn test_srvgg_32_wdn_full_fp16() {
        let result = test_model_loading_and_inference(
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-wdn-x4v3_fp32.safetensors",
            "SRVGGNetCompact (32-conv with denoise, realesr-general-wdn-x4v3)",
            "realesr-general-wdn-x4v3_fp32.safetensors",
            None,
            DType::F16,
        );

        if let Err(e) = result {
            panic!("SRVGG 32-conv WDN F16 test failed: {}", e);
        }
    }

    /// Test inference method with Tensor input (F32)
    #[test]
    fn test_inference_tensor_fp32() {
        eprintln!("\n============================================================");
        eprintln!("Testing inference method with Tensor input (F32)");
        eprintln!("============================================================");

        let mut device = Device::Cpu;
        #[cfg(target_os = "macos")]
        {
            if candle_core::utils::metal_is_available() {
                eprintln!("Using Metal device for acceleration");
                device = Device::new_metal(0).expect("Failed to create Metal device");
            }
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            if candle_core::utils::cuda_is_available() {
                eprintln!("Using CUDA device for acceleration");
                device = Device::new_cuda(0).expect("Failed to create CUDA device");
            }
        }

        // Download a small model for testing
        eprintln!("\nDownloading test model...");
        let model_path = get_model_path(
            "RealESRGAN_x4plus_anime_6B_fp32.safetensors",
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp32.safetensors",
        ).expect("Failed to download model");

        eprintln!("Loading model with DType::F32...");
        let real_esrgan = RRDBNet::from_model(&model_path, device.to_owned(), DType::F32)
            .expect("Failed to load RRDBNet for inference test");

        // Create a small test input Tensor [C, H, W] in F32 with values [0.0, 1.0]
        eprintln!("Creating test input Tensor [3, 32, 32] in F32...");
        let input_data: Vec<f32> = vec![0.5; 3 * 32 * 32];
        let input = Tensor::from_vec(input_data, (3, 32, 32), &device)
            .expect("Failed to create input tensor");

        eprintln!(
            "Input tensor shape: {:?}, dtype: {:?}",
            input.shape(),
            input.dtype()
        );

        // Call inference with scale_factor = 4.0
        eprintln!("Running inference with scale_factor = 4.0...");
        let result = real_esrgan.inference(&input, &4.0);

        match result {
            Ok(output) => {
                eprintln!("✓ inference succeeded");
                eprintln!("  Input shape: {:?}", input.shape());
                eprintln!("  Output shape: {:?}", output.shape());
                eprintln!("  Output dtype: {:?}", output.dtype());

                // Validate output shape: should be [3, 128, 128] for 4x upscaling
                let dims = output.dims();
                assert_eq!(dims.len(), 3, "Output should be 3D tensor");
                assert_eq!(dims[0], 3, "Channel dimension should be 3");
                assert_eq!(dims[1], 128, "Height should be 4x input (32*4=128)");
                assert_eq!(dims[2], 128, "Width should be 4x input (32*4=128)");

                eprintln!("✓ Output shape validation passed!");
                eprintln!("\n✓✓✓ inference (F32) test PASSED ✓✓✓\n");
            }
            Err(e) => {
                panic!("✗ inference failed: {}", e);
            }
        }
    }

    /// Test inference method with Tensor input (F16)
    #[test]
    fn test_inference_tensor_fp16() {
        eprintln!("\n============================================================");
        eprintln!("Testing inference method with Tensor input (F16)");
        eprintln!("============================================================");

        let mut device = Device::Cpu;
        #[cfg(target_os = "macos")]
        {
            if candle_core::utils::metal_is_available() {
                eprintln!("Using Metal device for acceleration");
                device = Device::new_metal(0).expect("Failed to create Metal device");
            }
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            if candle_core::utils::cuda_is_available() {
                eprintln!("Using CUDA device for acceleration");
                device = Device::new_cuda(0).expect("Failed to create CUDA device");
            }
        }

        // Download a small model for testing
        eprintln!("\nDownloading test model...");
        let model_path = get_model_path(
            "RealESRGAN_x4plus_anime_6B_fp16.safetensors",
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp16.safetensors",
        ).expect("Failed to download model");

        eprintln!("Loading model with DType::F16...");
        let real_esrgan = RRDBNet::from_model(&model_path, device.to_owned(), DType::F16)
            .expect("Failed to load RRDBNet for inference test");

        // Create a small test input Tensor [C, H, W] in F16 with values [0.0, 1.0]
        eprintln!("Creating test input Tensor [3, 32, 32] in F16...");
        let input_data_f32: Vec<f32> = vec![0.5; 3 * 32 * 32];
        let input = Tensor::from_vec(input_data_f32, (3, 32, 32), &device)
            .expect("Failed to create input tensor F32")
            .to_dtype(DType::F16)
            .expect("Failed to convert to F16");

        eprintln!(
            "Input tensor shape: {:?}, dtype: {:?}",
            input.shape(),
            input.dtype()
        );

        // Call inference with scale_factor = 4.0
        eprintln!("Running inference with scale_factor = 4.0...");
        let result = real_esrgan.inference(&input, &4.0);

        match result {
            Ok(output) => {
                eprintln!("✓ inference succeeded");
                eprintln!("  Input shape: {:?}", input.shape());
                eprintln!("  Output shape: {:?}", output.shape());
                eprintln!("  Output dtype: {:?}", output.dtype());

                // Validate output shape: should be [3, 128, 128] for 4x upscaling
                let dims = output.dims();
                assert_eq!(dims.len(), 3, "Output should be 3D tensor");
                assert_eq!(dims[0], 3, "Channel dimension should be 3");
                assert_eq!(dims[1], 128, "Height should be 4x input (32*4=128)");
                assert_eq!(dims[2], 128, "Width should be 4x input (32*4=128)");

                eprintln!("✓ Output shape validation passed!");
                eprintln!("\n✓✓✓ inference (F16) test PASSED ✓✓✓\n");
            }
            Err(e) => {
                panic!("✗ inference failed: {}", e);
            }
        }
    }

    /// Test inference method with dtype mismatch (F16 input, F32 model)
    #[test]
    fn test_inference_dtype_conversion() {
        eprintln!("\n============================================================");
        eprintln!("Testing inference with dtype conversion (F16 input, F32 model)");
        eprintln!("============================================================");

        let mut device = Device::Cpu;
        #[cfg(target_os = "macos")]
        {
            if candle_core::utils::metal_is_available() {
                eprintln!("Using Metal device for acceleration");
                device = Device::new_metal(0).expect("Failed to create Metal device");
            }
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            if candle_core::utils::cuda_is_available() {
                eprintln!("Using CUDA device for acceleration");
                device = Device::new_cuda(0).expect("Failed to create CUDA device");
            }
        }

        // Download a small model for testing
        eprintln!("\nDownloading test model...");
        let model_path = get_model_path(
            "RealESRGAN_x4plus_anime_6B_fp32.safetensors",
            "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp32.safetensors",
        ).expect("Failed to download model");

        eprintln!("Loading model with DType::F32...");
        let real_esrgan = RRDBNet::from_model(&model_path, device.to_owned(), DType::F32)
            .expect("Failed to load RRDBNet");

        // Create F16 input tensor
        eprintln!("Creating test input Tensor [3, 32, 32] in F16...");
        let input_data_f32: Vec<f32> = vec![0.5; 3 * 32 * 32];
        let input = Tensor::from_vec(input_data_f32, (3, 32, 32), &device)
            .expect("Failed to create input tensor")
            .to_dtype(DType::F16)
            .expect("Failed to convert to F16");

        eprintln!(
            "Input dtype: {:?}, Model dtype: {:?}",
            input.dtype(),
            real_esrgan.dtype
        );

        // Call inference - should auto-convert F16 to F32
        eprintln!("Running inference (should auto-convert F16 → F32)...");
        let result = real_esrgan.inference(&input, &4.0);

        match result {
            Ok(output) => {
                eprintln!("✓ inference succeeded with auto dtype conversion");
                eprintln!("  Output dtype: {:?}", output.dtype());

                // Validate output shape
                let dims = output.dims();
                assert_eq!(dims.len(), 3, "Output should be 3D tensor");
                assert_eq!(dims[0], 3, "Channel dimension should be 3");
                assert_eq!(dims[1], 128, "Height should be 4x input");
                assert_eq!(dims[2], 128, "Width should be 4x input");

                eprintln!("✓ dtype conversion and output shape validation passed!");
                eprintln!("\n✓✓✓ dtype conversion test PASSED ✓✓✓\n");
            }
            Err(e) => {
                panic!("✗ inference with dtype conversion failed: {}", e);
            }
        }
    }

    /// Helper to check if --cleanup flag is present in test arguments
    fn should_cleanup_cache() -> bool {
        std::env::args().any(|arg| arg == "--cleanup")
    }

    /// Test blend_balanced_frame function
    #[test]
    fn test_blend_balanced_frame() {
        eprintln!("\n============================================================");
        eprintln!("Testing blend_balanced_frame function");
        eprintln!("============================================================");

        let device = Device::Cpu;

        // Create two test frames [C, H, W] with f32 values [0.0, 1.0]
        let height = 32;
        let width = 32;
        let channels = 3;

        // Frame 1: filled with 0.2 (darker)
        let frame1_data: Vec<f32> = vec![0.2; channels * height * width];
        let frame1 = Tensor::from_vec(frame1_data, (channels, height, width), &device)
            .expect("Failed to create frame1");

        // Frame 2: filled with 0.8 (brighter)
        let frame2_data: Vec<f32> = vec![0.8; channels * height * width];
        let frame2 = Tensor::from_vec(frame2_data, (channels, height, width), &device)
            .expect("Failed to create frame2");

        eprintln!("Created test frames:");
        eprintln!("  Frame 1: all pixels = 0.2");
        eprintln!("  Frame 2: all pixels = 0.8");

        // Test case 1: factor = 0.0 (should return frame1)
        eprintln!("\nTest 1: factor = 0.0 (should return frame1)");
        let result = blend_balanced_frame(&frame1, &frame2, 0.0).expect("Failed to blend frames");
        let sample_value: f32 = result
            .get(0)
            .expect("Failed to get channel 0")
            .get(0)
            .expect("Failed to get row 0")
            .get(0)
            .expect("Failed to get col 0")
            .to_scalar::<f32>()
            .expect("Failed to convert to scalar");
        eprintln!("  Result sample value: {}", sample_value);
        assert!(
            (sample_value - 0.2).abs() < 1e-6,
            "Should be 0.2 when factor=0.0"
        );
        eprintln!("  ✓ Passed");

        // Test case 2: factor = 1.0 (should return frame2)
        eprintln!("\nTest 2: factor = 1.0 (should return frame2)");
        let result = blend_balanced_frame(&frame1, &frame2, 1.0).expect("Failed to blend frames");
        let sample_value: f32 = result
            .get(0)
            .expect("Failed to get channel 0")
            .get(0)
            .expect("Failed to get row 0")
            .get(0)
            .expect("Failed to get col 0")
            .to_scalar::<f32>()
            .expect("Failed to convert to scalar");
        eprintln!("  Result sample value: {}", sample_value);
        assert!(
            (sample_value - 0.8).abs() < 1e-6,
            "Should be 0.8 when factor=1.0"
        );
        eprintln!("  ✓ Passed");

        // Test case 3: factor = 0.5 (should be average)
        eprintln!("\nTest 3: factor = 0.5 (should be average)");
        let result = blend_balanced_frame(&frame1, &frame2, 0.5).expect("Failed to blend frames");
        let sample_value: f32 = result
            .get(0)
            .expect("Failed to get channel 0")
            .get(0)
            .expect("Failed to get row 0")
            .get(0)
            .expect("Failed to get col 0")
            .to_scalar::<f32>()
            .expect("Failed to convert to scalar");
        eprintln!("  Result sample value: {}", sample_value);
        assert!(
            (sample_value - 0.5).abs() < 1e-6,
            "Should be 0.5 when factor=0.5"
        );
        eprintln!("  ✓ Passed");

        // Test case 4: factor = 0.3 (30% of frame2, 70% of frame1)
        eprintln!("\nTest 4: factor = 0.3 (30% frame2 + 70% frame1)");
        let result = blend_balanced_frame(&frame1, &frame2, 0.3).expect("Failed to blend frames");
        let sample_value: f32 = result
            .get(0)
            .expect("Failed to get channel 0")
            .get(0)
            .expect("Failed to get row 0")
            .get(0)
            .expect("Failed to get col 0")
            .to_scalar::<f32>()
            .expect("Failed to convert to scalar");
        let expected = 0.2 * 0.7 + 0.8 * 0.3; // = 0.14 + 0.24 = 0.38
        eprintln!("  Result sample value: {}", sample_value);
        eprintln!("  Expected value: {}", expected);
        assert!(
            (sample_value - expected).abs() < 1e-6,
            "Blend calculation should be correct"
        );
        eprintln!("  ✓ Passed");

        // Test case 5: factor > 1.0 (should clamp to 1.0)
        eprintln!("\nTest 5: factor = 1.5 (should clamp to 1.0)");
        let result = blend_balanced_frame(&frame1, &frame2, 1.5).expect("Failed to blend frames");
        let sample_value: f32 = result
            .get(0)
            .expect("Failed to get channel 0")
            .get(0)
            .expect("Failed to get row 0")
            .get(0)
            .expect("Failed to get col 0")
            .to_scalar::<f32>()
            .expect("Failed to convert to scalar");
        eprintln!("  Result sample value: {}", sample_value);
        assert!(
            (sample_value - 0.8).abs() < 1e-6,
            "Should clamp to 0.8 when factor>1.0"
        );
        eprintln!("  ✓ Passed");

        // Test case 6: factor < 0.0 (should clamp to 0.0)
        eprintln!("\nTest 6: factor = -0.5 (should clamp to 0.0)");
        let result = blend_balanced_frame(&frame1, &frame2, -0.5).expect("Failed to blend frames");
        let sample_value: f32 = result
            .get(0)
            .expect("Failed to get channel 0")
            .get(0)
            .expect("Failed to get row 0")
            .get(0)
            .expect("Failed to get col 0")
            .to_scalar::<f32>()
            .expect("Failed to convert to scalar");
        eprintln!("  Result sample value: {}", sample_value);
        assert!(
            (sample_value - 0.2).abs() < 1e-6,
            "Should clamp to 0.2 when factor<0.0"
        );
        eprintln!("  ✓ Passed");

        eprintln!("\n✓✓✓ blend_balanced_frame test PASSED ✓✓✓\n");
    }

    #[test]
    fn test_all_models() {
        eprintln!("\n\n======================================================================");
        eprintln!("COMPREHENSIVE MODEL TESTING");
        eprintln!("Testing all Real-ESRGAN models (F32 and F16) with automatic download");
        eprintln!("======================================================================\n");

        let test_cases = vec![
            // F32 variants
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_fp32.safetensors",
                "RRDBNet 23-block (F32)",
                "RealESRGAN_x4plus_fp32.safetensors",
                DType::F32,
            ),
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp32.safetensors",
                "RRDBNet 6-block (anime) (F32)",
                "RealESRGAN_x4plus_anime_6B_fp32.safetensors",
                DType::F32,
            ),
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-animevideov3_fp32.safetensors",
                "SRVGG 16-conv (F32)",
                "realesr-animevideov3_fp32.safetensors",
                DType::F32,
            ),
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-x4v3_fp32.safetensors",
                "SRVGG 32-conv (F32)",
                "realesr-general-x4v3_fp32.safetensors",
                DType::F32,
            ),
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-wdn-x4v3_fp32.safetensors",
                "SRVGG 32-conv (denoise) (F32)",
                "realesr-general-wdn-x4v3_fp32.safetensors",
                DType::F32,
            ),
            // F16 variants
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_fp16.safetensors",
                "RRDBNet 23-block (F16)",
                "RealESRGAN_x4plus_fp16.safetensors",
                DType::F16,
            ),
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp16.safetensors",
                "RRDBNet 6-block (anime) (F16)",
                "RealESRGAN_x4plus_anime_6B_fp16.safetensors",
                DType::F16,
            ),
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-animevideov3_fp16.safetensors",
                "SRVGG 16-conv (F16)",
                "realesr-animevideov3_fp16.safetensors",
                DType::F16,
            ),
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-x4v3_fp16.safetensors",
                "SRVGG 32-conv (F16)",
                "realesr-general-x4v3_fp16.safetensors",
                DType::F16,
            ),
            (
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-wdn-x4v3_fp16.safetensors",
                "SRVGG 32-conv (denoise) (F16)",
                "realesr-general-wdn-x4v3_fp16.safetensors",
                DType::F16,
            ),
        ];

        let mut passed = 0;
        let mut failed = 0;

        for (url, name, filename, dtype) in &test_cases {
            match test_model_loading_and_inference(url, name, filename, None, *dtype) {
                Ok(_) => {
                    passed += 1;
                }
                Err(e) => {
                    eprintln!("✗✗✗ {} test FAILED: {}\n", name, e);
                    failed += 1;
                }
            }
        }

        eprintln!("\n======================================================================");
        eprintln!("TEST SUMMARY");
        eprintln!("======================================================================");
        eprintln!("Passed: {}/{}", passed, test_cases.len());
        eprintln!("Failed: {}/{}", failed, test_cases.len());
        eprintln!("======================================================================\n");

        // Clean up test cache directory only if --cleanup flag is passed
        if should_cleanup_cache() {
            if let Ok(cache_dir) = get_test_cache_dir() {
                eprintln!("Cleaning up test models from: {}", cache_dir.display());
                match fs::remove_dir_all(&cache_dir) {
                    Ok(_) => eprintln!("✓ Test cache cleaned up successfully"),
                    Err(e) => eprintln!("⚠ Warning: Failed to clean up test cache: {}", e),
                }
            }
        } else {
            if let Ok(cache_dir) = get_test_cache_dir() {
                eprintln!("ℹ Test models cached in: {}", cache_dir.display());
                eprintln!(
                    "  To clean up: cargo test --lib utils::realesrgan::tests -- --cleanup"
                );
            }
        }

        if failed > 0 {
            panic!("{} test(s) failed", failed);
        }
    }
}
