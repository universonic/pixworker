use anyhow::{Result, anyhow};
use candle_core::Device;
use candle_onnx::{self as candle_onnx, eval::Value as CValue, eval::simple_eval};
use half::f16;
use ndarray::{Array, Ix4, IxDyn};
use std::collections::HashMap;
use std::path::Path;

pub struct RealESRGAN {
    model: candle_onnx::onnx::ModelProto,
    device: Device,
    pub use_fp16: bool,
    pub supports_denoise: bool,
}

impl RealESRGAN {
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
    pub fn infer_fp32(
        &self,
        img: Array<f32, IxDyn>,
        denoise: Array<f32, IxDyn>,
    ) -> Result<Array<f32, Ix4>> {
        // Build inputs and call candle-onnx simple_eval
        let mut inputs: HashMap<String, CValue> = HashMap::new();
        let make_value = |a: Array<f32, IxDyn>| -> Result<CValue> {
            let shape: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
            let data: Vec<f32> = a.into_iter().collect();
            Ok(CValue::from_vec(data, shape.as_slice(), &self.device)?)
        };
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
                make_value(img.clone().into_dyn())?
            } else {
                make_value(denoise.clone().into_dyn())?
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
    pub fn infer_fp16(
        &self,
        img: Array<f16, IxDyn>,
        denoise: Array<f16, IxDyn>,
    ) -> Result<Array<f16, Ix4>> {
        // Build inputs and call candle-onnx simple_eval
        let mut inputs: HashMap<String, CValue> = HashMap::new();
        let make_value = |a: Array<f16, IxDyn>| -> Result<CValue> {
            let shape: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
            let data: Vec<f16> = a.into_iter().collect();
            Ok(CValue::from_vec(data, shape.as_slice(), &self.device)?)
        };
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
                make_value(img.clone().into_dyn())?
            } else {
                make_value(denoise.clone().into_dyn())?
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
}
