use anyhow::{Result, anyhow};
use candle_core::Device;
use candle_onnx::onnx::ModelProto;
use candle_onnx::{eval::Value as CValue, eval::simple_eval};
use half::f16;
use ndarray::{Array, Axis, Ix1, Ix3, Ix4, IxDyn};
use std::collections::HashMap;
use std::path::Path;

pub struct RealESRGAN {
    model: ModelProto,
    device: Device,
    supports_denoise: bool,
    use_fp16: bool,
}

impl RealESRGAN {
    pub fn run(
        &self,
        frame: &Array<f32, Ix3>,
        scale_factor: &f64,
        denoise_factor: &f32,
    ) -> Result<Array<f32, Ix3>> {
        // All Real-ESRGAN models are 4x upscale models
        const MODEL_SCALE: f64 = 4.0;
        // Determine how many times we need to apply 4x upscaling
        // For target_scale <= 4: apply once, then downscale if needed
        // For 4 < target_scale <= 16: apply twice (4x then 4x = 16x), then downscale
        // For 16 < target_scale: apply log4(target) times
        let num_upscale_passes = if *scale_factor <= 1.0 {
            // If target is smaller than input, just resize (no upscaling needed)
            0
        } else if *scale_factor <= MODEL_SCALE {
            // Single pass is sufficient
            1
        } else {
            // Multiple passes needed: calculate how many 4x passes to exceed target
            (scale_factor.log(MODEL_SCALE).ceil() as usize).max(1)
        };

        let mut result = frame.clone();
        // Apply upscaling multiple times if needed
        // Each pass applies 4x upscaling, so 2 passes = 16x total
        for _ in 0..num_upscale_passes {
            // Convert to CHW format [C, H, W] and normalize to [0, 1]
            // Real-ESRGAN expects normalized input in [0, 1] range
            let chw: Array<f32, Ix3> = self.hwc_to_chw(&frame)? / 255.0;

            // Add batch dimension: [1, C, H, W]
            let chw_batch: Array<f32, Ix4> = chw.view().insert_axis(Axis(0)).to_owned();

            // Run inference via RealESRGAN wrapper and convert to HWC [0,255]
            result = if self.use_fp16 {
                // FP16 path: convert input to fp16, run inference, convert output back
                let img_array: Array<f16, Ix4> = chw_batch.mapv(|v| f16::from_f32(v)).to_owned();

                // Prepare denoise tensor in fp16
                let denoise_array = if self.supports_denoise {
                    Array::from_shape_vec((1,), vec![f16::from_f32(*denoise_factor)])?
                } else {
                    Array::from_shape_vec((1,), vec![f16::from_f32(0.0)])?
                };

                let output_4d = self.infer_fp16(&img_array, &denoise_array)?;
                let output_3d = output_4d.index_axis(Axis(0), 0);
                let hwc_view = output_3d.permuted_axes([1, 2, 0]);
                let hwc = hwc_view.as_standard_layout().into_owned();

                // Convert fp16 to f32 and scale to [0, 255]
                hwc.mapv(|v| (v.to_f32() * 255.0).clamp(0.0, 255.0))
            } else {
                // FP32 path
                let img_array = chw_batch.to_owned();
                let denoise_array = if self.supports_denoise {
                    Array::from_shape_vec((1,), vec![*denoise_factor])?
                } else {
                    Array::from_shape_vec((1,), vec![0.0f32])?
                };

                let output_4d = self.infer_fp32(&img_array, &denoise_array)?;
                let output_3d = output_4d.index_axis(Axis(0), 0);
                let hwc_view = output_3d.permuted_axes([1, 2, 0]);
                let hwc = hwc_view.as_standard_layout().into_owned();

                hwc.mapv(|v| (v * 255.0).clamp(0.0, 255.0))
            };
        }

        Ok(result)
    }

    pub fn from_model<P: AsRef<Path>>(
        path: P,
        device: Device,
        use_fp16: bool,
        supports_denoise: bool,
    ) -> Result<Self> {
        let model = candle_onnx::read_file(path.as_ref())?;
        Ok(Self {
            model,
            device,
            use_fp16,
            supports_denoise,
        })
    }

    /// Run FP32 inference. `img` shape expected: [1, C, H, W]
    /// If `supports_denoise` is true, `denoise` should be shape [1]
    fn infer_fp32(
        &self,
        img: &Array<f32, Ix4>,
        denoise: &Array<f32, Ix1>,
    ) -> Result<Array<f32, Ix4>> {
        // Build inputs and call candle-onnx simple_eval
        let mut inputs: HashMap<String, CValue> = HashMap::new();
        // model.graph.input ordering assumed: img, (denoise?)
        for (idx, vi) in self
            .model
            .graph
            .as_ref()
            .and_then(|g| Some(&g.input))
            .into_iter()
            .flat_map(|i| i.iter())
            .enumerate()
        {
            let name = vi.name.clone();
            let v = if idx == 0 {
                self.make_value_f32(img.clone().into_dyn())?
            } else {
                self.make_value_f32(denoise.clone().into_dyn())?
            };
            inputs.insert(name, v);
        }
        let outputs = simple_eval(&self.model, inputs)?;
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
        let arr = Array::from_shape_vec(shape_vec.clone(), flat)?;
        let arr4 = arr.into_dimensionality::<Ix4>()?;
        Ok(arr4)
    }

    /// Run FP16 inference.
    fn infer_fp16(
        &self,
        img: &Array<f16, Ix4>,
        denoise: &Array<f16, Ix1>,
    ) -> Result<Array<f16, Ix4>> {
        // Build inputs and call candle-onnx simple_eval
        let mut inputs: HashMap<String, CValue> = HashMap::new();
        // model.graph.input ordering assumed: img, (denoise?)
        for (idx, vi) in self
            .model
            .graph
            .as_ref()
            .and_then(|g| Some(&g.input))
            .into_iter()
            .flat_map(|i| i.iter())
            .enumerate()
        {
            let name = vi.name.clone();
            let v = if idx == 0 {
                self.make_value_f16(img.clone().into_dyn())?
            } else {
                self.make_value_f16(denoise.clone().into_dyn())?
            };
            inputs.insert(name, v);
        }
        let outputs = simple_eval(&self.model, inputs)?;
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

    /// Convert frame from HWC to CHW layout
    fn hwc_to_chw(&self, frame: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>> {
        Ok(frame.view().permuted_axes([2, 0, 1]).to_owned())
    }
}
