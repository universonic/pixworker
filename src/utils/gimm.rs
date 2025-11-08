use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, Tensor};
use candle_onnx::onnx::ModelProto;
use candle_onnx::{eval::Value as CValue, eval::simple_eval};
use half::f16;
use std::collections::HashMap;
use std::path::Path;

pub struct GimmVfi {
    model: ModelProto,
    device: Device,
    dtype: DType,
}

impl GimmVfi {
    pub fn inference(
        &self,
        frame_start: &Tensor,
        frame_end: &Tensor,
        num_interp: usize,
    ) -> Result<Vec<Tensor>> {
        let (c_start, h_start, w_start) = frame_start.dims3()?;
        let (c_end, h_end, w_end) = frame_end.dims3()?;
        
        if c_start != 3 || c_end != 3 {
            bail!("Expected 3 channels, got {} and {}", c_start, c_end);
        }
        if (h_start, w_start) != (h_end, w_end) {
            bail!("Frame dimensions mismatch");
        }

        let orig_height = h_start as usize;
        let orig_width = w_start as usize;

        // Calculate padding for divisibility by 16
        const DIVISOR: usize = 16;
        let pad_h = ((orig_height + DIVISOR - 1) / DIVISOR) * DIVISOR - orig_height;
        let pad_w = ((orig_width + DIVISOR - 1) / DIVISOR) * DIVISOR - orig_width;
        let pad_top = pad_h / 2;
        let pad_bottom = pad_h - pad_top;
        let pad_left = pad_w / 2;
        let pad_right = pad_w - pad_left;

        let padded_height = orig_height + pad_h;
        let padded_width = orig_width + pad_w;

        // Pad frames
        let frame_start_padded = self.pad_tensor_replicate(frame_start, pad_top, pad_bottom, pad_left, pad_right)?;
        let frame_end_padded = self.pad_tensor_replicate(frame_end, pad_top, pad_bottom, pad_left, pad_right)?;

        // Stack frames to [1, C, 2, H, W]
        let img_xs = self.stack_frames_5d(&frame_start_padded, &frame_end_padded)?;

        let mut result_frames = Vec::with_capacity(num_interp);

        for i in 0..num_interp {
            let t_value = (i + 1) as f32 / (num_interp + 1) as f32;

            let padded_frame = if self.dtype == DType::F16 {
                let coord = self.generate_coord_tensor(1, padded_height, padded_width, t_value)?;
                let t = Tensor::full(f16::from_f32(t_value), 1, &self.device)?.to_dtype(DType::F16)?;
                let img_xs_f16 = img_xs.to_dtype(DType::F16)?;
                
                let output_4d = self.infer_fp16(&img_xs_f16, &coord, &t)?;
                output_4d.squeeze(0)?
            } else {
                let coord = self.generate_coord_tensor(1, padded_height, padded_width, t_value)?;
                let t = Tensor::full(t_value, 1, &self.device)?;
                
                let output_4d = self.infer_fp32(&img_xs, &coord, &t)?;
                output_4d.squeeze(0)?
            };

            let result_frame = self.unpad_tensor(&padded_frame, pad_top, pad_left, orig_height, orig_width)?;
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
            dtype: if use_fp16 { DType::F16 } else { DType::F32 },
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run inference for FP32 inputs and return Tensor output [1, C, H, W]
    fn infer_fp32(
        &self,
        img_xs: &Tensor,
        coord: &Tensor,
        t: &Tensor,
    ) -> Result<Tensor> {
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
                0 => self.tensor_to_cvalue_f32(img_xs)?,
                1 => self.tensor_to_cvalue_f32(coord)?,
                2 => self.tensor_to_cvalue_f32(t)?,
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
        let flat: Vec<f32> = out_val.to_vec1()?;
        let shape_vec: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let t_out = Tensor::from_vec(flat, shape_vec.as_slice(), &self.device)?;
        Ok(t_out)
    }

    /// Run inference for FP16 inputs and return Tensor output [1, C, H, W]
    fn infer_fp16(
        &self,
        img_xs: &Tensor,
        coord: &Tensor,
        t: &Tensor,
    ) -> Result<Tensor> {
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
                0 => self.tensor_to_cvalue_f16(img_xs)?,
                1 => self.tensor_to_cvalue_f32(coord)?,
                2 => self.tensor_to_cvalue_f16(t)?,
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
        let flat: Vec<f16> = out_val.to_vec1()?;
        let shape_vec: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let t_out = Tensor::from_vec(flat, shape_vec.as_slice(), &self.device)?;
        Ok(t_out)
    }


    // Tensor helper methods
    fn pad_tensor_replicate(
        &self,
        tensor: &Tensor,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
    ) -> Result<Tensor> {
        // tensor is [C, H, W], pad H and W dimensions
        let (c, h, w) = tensor.dims3()?;
        let new_h = h + pad_top + pad_bottom;
        let new_w = w + pad_left + pad_right;

        // Create output tensor filled with zeros, then copy values
        let mut out_vec = vec![0.0_f32; (c * new_h * new_w) as usize];
        
        // Get input data
        let input_data = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        
        // Copy with padding
        for ch in 0..c as usize {
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let in_idx = (ch * (h as usize) * (w as usize)) + (y * (w as usize)) + x;
                    let out_y = y + pad_top;
                    let out_x = x + pad_left;
                    let out_idx = (ch * (new_h as usize) * (new_w as usize)) + (out_y * (new_w as usize)) + out_x;
                    out_vec[out_idx] = input_data[in_idx];
                }
            }
        }
        
        let t = Tensor::from_vec(out_vec, (c as usize, new_h as usize, new_w as usize), &self.device)?;
        Ok(t.to_dtype(tensor.dtype())?)
    }

    fn stack_frames_5d(&self, frame1: &Tensor, frame2: &Tensor) -> Result<Tensor> {
        // frames are [C, H, W], create [1, C, 2, H, W]
        let (c, h, w) = frame1.dims3()?;
        
        let f1_data: Vec<f32> = frame1.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let f2_data: Vec<f32> = frame2.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        
        let mut out_vec = Vec::with_capacity(2 * (c * h * w) as usize);
        out_vec.extend_from_slice(&f1_data);
        out_vec.extend_from_slice(&f2_data);
        
        let t = Tensor::from_vec(out_vec, (1, c as usize, 2, h as usize, w as usize), &self.device)?;
        Ok(t.to_dtype(self.dtype)?)
    }

    fn generate_coord_tensor(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        t_value: f32,
    ) -> Result<Tensor> {
        let mut coord_vec = vec![0.0_f32; batch_size * 1 * height * width * 3];
        
        for b in 0..batch_size {
            for h in 0..height {
                for w in 0..width {
                    let idx = b * (1 * height * width * 3) + 0 * (height * width * 3) + (h * width * 3) + (w * 3);
                    coord_vec[idx] = t_value; // t
                    coord_vec[idx + 1] = -1.0 + 2.0 * ((h as f32 + 0.5) / height as f32); // y
                    coord_vec[idx + 2] = -1.0 + 2.0 * ((w as f32 + 0.5) / width as f32); // x
                }
            }
        }
        
        let t = Tensor::from_vec(coord_vec, (batch_size, 1, height, width, 3), &self.device)?;
        Ok(t)
    }

    fn unpad_tensor(
        &self,
        tensor: &Tensor,
        pad_top: usize,
        pad_left: usize,
        orig_height: usize,
        orig_width: usize,
    ) -> Result<Tensor> {
        // tensor is [C, H_padded, W_padded]
        let (c, h, w) = tensor.dims3()?;
        
        let data: Vec<f32> = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut out_vec = Vec::with_capacity((c * orig_height * orig_width) as usize);
        
        for ch in 0..c as usize {
            for y in 0..orig_height {
                for x in 0..orig_width {
                    let in_y = y + pad_top;
                    let in_x = x + pad_left;
                    let in_idx = (ch * (h as usize) * (w as usize)) + (in_y * (w as usize)) + in_x;
                    out_vec.push(data[in_idx]);
                }
            }
        }
        
        let t = Tensor::from_vec(out_vec, (c as usize, orig_height, orig_width), &self.device)?;
        Ok(t.to_dtype(tensor.dtype())?)
    }

    fn tensor_to_cvalue_f32(&self, tensor: &Tensor) -> Result<CValue> {
        let t = tensor.to_dtype(DType::F32)?;
        let flat = t.flatten_all()?;
        let data: Vec<f32> = flat.to_vec1()?;
        let shape: Vec<usize> = t.shape().dims().to_vec();
        Ok(CValue::from_vec(data, shape.as_slice(), &self.device)?)
    }

    fn tensor_to_cvalue_f16(&self, t: &Tensor) -> Result<CValue> {
        let t_f16 = if t.dtype() != DType::F16 { t.to_dtype(DType::F16)? } else { t.to_owned() };
        let flat: Vec<f16> = t_f16.flatten_all()?.to_vec1()?;
        let shape: Vec<usize> = t_f16.shape().dims().to_vec();
        Ok(CValue::from_vec(flat, shape.as_slice(), &self.device)?)
    }
}
