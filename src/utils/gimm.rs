use anyhow::{Result, anyhow, bail};
use candle_core::Device;
use candle_onnx::onnx::ModelProto;
use candle_onnx::{eval::Value as CValue, eval::simple_eval};
use half::f16;
use ndarray::stack;
use ndarray::{Array, Axis, Ix1, Ix3, Ix4, Ix5, IxDyn, s};
use std::collections::HashMap;
use std::path::Path;

pub struct GimmVfi {
    model: ModelProto,
    device: Device,
    use_fp16: bool,
}

impl GimmVfi {
    pub fn run(
        &self,
        frame_start: &Array<f32, Ix3>,
        frame_end: &Array<f32, Ix3>,
        num_interp: &usize,
    ) -> Result<Vec<Array<f32, Ix3>>> {
        let (orig_height, orig_width, channels) = frame_start.dim();
        if channels != 3 {
            bail!("Expected RGB frames with 3 channels, got {}", channels);
        }

        // Validate that both frames have the same dimensions
        if frame_end.dim() != (orig_height, orig_width, channels) {
            bail!("Frame dimensions mismatch");
        }

        // Calculate padding to make dimensions divisible by 16 (FlowFormer requirement)
        // FlowFormer uses patch_size=8 but has additional constraints requiring divisor=16
        const DIVISOR: usize = 16;
        let pad_h = ((orig_height + DIVISOR - 1) / DIVISOR) * DIVISOR - orig_height;
        let pad_w = ((orig_width + DIVISOR - 1) / DIVISOR) * DIVISOR - orig_width;
        let pad_top = pad_h / 2;
        let pad_bottom = pad_h - pad_top;
        let pad_left = pad_w / 2;
        let pad_right = pad_w - pad_left;

        let padded_height = orig_height + pad_h;
        let padded_width = orig_width + pad_w;

        // Pad frames using replication mode
        let frame_start_padded: Array<f32, Ix3> =
            self.pad_frame_replicate(frame_start, pad_top, pad_bottom, pad_left, pad_right)?;
        let frame_end_padded: Array<f32, Ix3> =
            self.pad_frame_replicate(frame_end, pad_top, pad_bottom, pad_left, pad_right)?;

        // Use padded dimensions for processing
        let (height, width) = (padded_height, padded_width);

        // Convert frames from [H, W, C] to [C, H, W] and normalize to [0, 1]
        let frame_start_chw: Array<f32, Ix3> = self.hwc_to_chw(&frame_start_padded)? / 255.0;
        let frame_end_chw: Array<f32, Ix3> = self.hwc_to_chw(&frame_end_padded)? / 255.0;

        // Stack frames to create input tensor [1, C, 2, H, W]
        let frame_start_batch = frame_start_chw.view().insert_axis(Axis(0));
        let frame_end_batch = frame_end_chw.view().insert_axis(Axis(0));
        let img_xs: Array<f32, Ix5> = stack(Axis(2), &[frame_start_batch, frame_end_batch])?;

        // Determine dtype based on model wrapper
        let img_xs_fp16 = self
            .use_fp16
            .then(|| img_xs.mapv(|value| f16::from_f32(value)));

        // Generate all interpolated frames
        let mut result_frames = Vec::with_capacity(*num_interp);

        for i in 0..*num_interp {
            // Calculate time value for this interpolation
            let t_value = (i + 1) as f32 / (num_interp + 1) as f32;

            // ================================================================
            // Generate all inputs based on model precision
            // This avoids unnecessary type conversions between fp16 and fp32
            // ================================================================

            // Note: ds_factor is now fixed at 1.0 inside the ONNX model
            // No need to pass it as an input anymore

            // Prepare inputs and run inference via GimmVfi wrapper
            let padded_frame = if self.use_fp16 {
                // FP16 path: img_xs and t are fp16, coord is ALWAYS fp32
                let coord_array = self
                    .generate_coord(1, height, width, t_value)
                    .map_err(|e| anyhow!("Failed to generate coord: {}", e))?;

                // Create t tensor in fp16
                let t_array = Array::from_shape_vec((1,), vec![f16::from_f32(t_value)])?;

                // Prepare owned arrays and convert to dynamic dims
                let img_xs_array = img_xs_fp16
                    .as_ref()
                    .expect("fp16 tensor available")
                    .view()
                    .to_owned();

                // Run model and get owned 4D output [1, C, H, W]
                let output_4d = self
                    .infer_fp16(img_xs_array, coord_array, t_array)
                    .map_err(|e| anyhow!("Failed to run inference: {}", e))?;

                let output_3d = output_4d.index_axis(Axis(0), 0);
                let hwc_view = output_3d.permuted_axes([1, 2, 0]);
                let hwc = hwc_view.as_standard_layout().into_owned();

                // Convert fp16 to f32 and scale to [0, 255]
                hwc.mapv(|value| (value.to_f32() * 255.0).clamp(0.0, 255.0))
            } else {
                // FP32 path
                let coord_array = self.generate_coord(1, height, width, t_value)?;
                let t_array = Array::from_shape_vec((1,), vec![t_value])?;

                let img_xs_array = img_xs.view().to_owned();

                let output_4d = self
                    .infer_fp32(img_xs_array, coord_array, t_array)
                    .map_err(|e| anyhow!("Failed to run inference: {}", e))?;

                let output_3d = output_4d.index_axis(Axis(0), 0);
                let hwc_view = output_3d.permuted_axes([1, 2, 0]);
                let hwc = hwc_view.as_standard_layout().into_owned();

                hwc.mapv(|value| (value * 255.0).clamp(0.0, 255.0))
            };

            // Unpad the output frame back to original dimensions
            let result_frame =
                self.unpad_frame(&padded_frame, pad_top, pad_left, orig_height, orig_width)?;

            result_frames.push(result_frame);
        }

        Ok(result_frames)
    }

    /// Create wrapper by loading an ONNX model via Candle/`candle-onnx`.
    ///
    /// Assumptions:
    /// - `candle-onnx::read_file` returns a `ModelProto` and `eval::simple_eval`
    ///   accepts a `&ModelProto` and a HashMap of named inputs.
    /// - Caller provides a `Device` (e.g. `Device::Cpu`) for tensor allocation.
    pub fn from_model<P: AsRef<Path>>(path: P, device: Device, use_fp16: bool) -> Result<Self> {
        let model = candle_onnx::read_file(path.as_ref())?;
        Ok(Self {
            model,
            device,
            use_fp16,
        })
    }

    /// Run inference for FP32 inputs and return an owned 4D array [1, C, H, W]
    fn infer_fp32(
        &self,
        img_xs: Array<f32, Ix5>,
        coord: Array<f32, Ix5>,
        t: Array<f32, Ix1>,
    ) -> Result<Array<f32, Ix4>> {
        // Build inputs map from model input names using the graph inputs
        let mut inputs: HashMap<String, CValue> = HashMap::new();

        for (idx, vi) in self
            .model
            .graph
            .as_ref()
            .and_then(|g| Some(&g.input))
            .into_iter()
            .flat_map(|i| i.iter())
            .enumerate()
            .take(3)
        {
            let name = vi.name.clone();
            let v = match idx {
                0 => self.make_value_f32(img_xs.clone().into_dyn())?,
                1 => self.make_value_f32(coord.clone().into_dyn())?,
                2 => self.make_value_f32(t.clone().into_dyn())?,
                _ => continue,
            };
            inputs.insert(name, v);
        }

        let outputs =
            simple_eval(&self.model, inputs).map_err(|e| anyhow!("Error evaluation: {}", e))?;
        let out_name = self
            .model
            .graph
            .as_ref()
            .and_then(|g| g.output.get(0))
            .map(|o| o.name.clone())
            .unwrap_or_else(|| "output".to_string());
        let out_val = outputs
            .get(&out_name)
            .ok_or_else(|| anyhow!("Candle eval returned no output named {}", out_name))?;

        let dims = out_val.dims();
        let flat: Vec<f32> = out_val.to_vec1()?; // flatten
        let shape_vec: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let arr = Array::from_shape_vec(shape_vec.clone(), flat)?;
        let arr4 = arr.into_dimensionality::<Ix4>()?;
        Ok(arr4)
    }

    /// Run inference for FP16 inputs and return an owned 4D array [1, C, H, W]
    fn infer_fp16(
        &self,
        img_xs: Array<f16, Ix5>,
        coord: Array<f32, Ix5>, // coord stays fp32 per model requirement
        t: Array<f16, Ix1>,
    ) -> Result<Array<f16, Ix4>> {
        // Build inputs map from model input names using the graph inputs
        let mut inputs: HashMap<String, CValue> = HashMap::new();

        for (idx, vi) in self
            .model
            .graph
            .as_ref()
            .and_then(|g| Some(&g.input))
            .into_iter()
            .flat_map(|i| i.iter())
            .enumerate()
            .take(3)
        {
            let name = vi.name.clone();
            let v = match idx {
                0 => self.make_value_f16(img_xs.clone().into_dyn())?,
                1 => self.make_value_f32(coord.clone().into_dyn())?,
                2 => self.make_value_f16(t.clone().into_dyn())?,
                _ => continue,
            };
            inputs.insert(name, v);
        }

        let outputs =
            simple_eval(&self.model, inputs).map_err(|e| anyhow!("Error evaluation: {}", e))?;
        let out_name = self
            .model
            .graph
            .as_ref()
            .and_then(|g| g.output.get(0))
            .map(|o| o.name.clone())
            .unwrap_or_else(|| "output".to_string());
        let out_val = outputs
            .get(&out_name)
            .ok_or_else(|| anyhow!("Candle eval returned no output named {}", out_name))?;

        let dims = out_val.dims();
        let flat: Vec<f16> = out_val.to_vec1()?; // flatten
        let shape_vec: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let arr = Array::from_shape_vec(shape_vec.clone(), flat)?;
        let arr4 = arr.into_dimensionality::<Ix4>()?;
        Ok(arr4)
    }

    // Convert ndarrays to flat Vec and build CValue (Tensor) for f32
    fn make_value_f32(&self, a: Array<f32, IxDyn>) -> Result<CValue> {
        let shape: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = a.into_iter().collect();
        Ok(CValue::from_vec(data, shape.as_slice(), &self.device)?)
    }

    // Convert ndarrays to flat Vec and build CValue (Tensor) for f16
    fn make_value_f16(&self, a: Array<f16, IxDyn>) -> Result<CValue> {
        let shape: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
        let data: Vec<f16> = a.into_iter().collect();
        Ok(CValue::from_vec(data, shape.as_slice(), &self.device)?)
    }

    fn pad_frame_replicate(
        &self,
        frame: &Array<f32, Ix3>,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
    ) -> Result<Array<f32, Ix3>> {
        let (height, width, channels) = frame.dim();
        let new_height = height + pad_top + pad_bottom;
        let new_width = width + pad_left + pad_right;
        Ok(Array::from_shape_fn(
            (new_height, new_width, channels),
            |(h, w, c)| {
                let src_h = if h < pad_top {
                    0
                } else if h >= pad_top + height {
                    height - 1
                } else {
                    h - pad_top
                };

                let src_w = if w < pad_left {
                    0
                } else if w >= pad_left + width {
                    width - 1
                } else {
                    w - pad_left
                };

                frame[[src_h, src_w, c]]
            },
        ))
    }

    fn hwc_to_chw(&self, frame: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>> {
        Ok(frame.view().permuted_axes([2, 0, 1]).to_owned())
    }

    /// Generate coordinate tensor for GIMMVFI INR sampling in fp32
    ///
    /// # Arguments
    /// * `batch_size` - Batch dimension size (typically 1)
    /// * `height` - Spatial height dimension
    /// * `width` - Spatial width dimension
    /// * `t_value` - Temporal coordinate value in range [0, 1]
    ///
    /// # Returns
    /// Coordinate tensor of shape [batch_size, 1, height, width, 3] in fp32
    fn generate_coord(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        t_value: f32,
    ) -> Result<Array<f32, ndarray::Dim<[usize; 5]>>> {
        // CRITICAL: Coordinate generation must match Python's CoordSampler3D.shape2coordinate
        // - t_value: NOT mapped to coord_range, used as-is (e.g., 0.5 for middle frame)
        // - spatial (h, w): pixel centers mapped to coord_range [-1, 1]
        //   Formula: coord = coord_range[0] + (coord_range[1] - coord_range[0]) * ((pixel + 0.5) / size)
        //   For coord_range=[-1, 1]: coord = -1 + 2 * ((pixel + 0.5) / size)
        Ok(Array::from_shape_fn(
            (batch_size, 1, height, width, 3),
            |(_, _, h, w, component)| match component {
                0 => t_value, // t: raw value in [0, 1], NOT mapped to [-1, 1]
                1 => -1.0 + 2.0 * ((h as f32 + 0.5) / height as f32), // y (h)
                2 => -1.0 + 2.0 * ((w as f32 + 0.5) / width as f32), // x (w)
                _ => unreachable!("coordinate component out of range"),
            },
        ))
    }

    /// Remove padding from a frame to restore original dimensions
    fn unpad_frame(
        &self,
        padded_frame: &Array<f32, Ix3>,
        pad_top: usize,
        pad_left: usize,
        orig_height: usize,
        orig_width: usize,
    ) -> Result<Array<f32, Ix3>> {
        Ok(padded_frame
            .slice(s![
                pad_top..pad_top + orig_height,
                pad_left..pad_left + orig_width,
                ..
            ])
            .to_owned())
    }
}
