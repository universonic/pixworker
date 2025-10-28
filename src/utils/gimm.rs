use anyhow::{Result, anyhow};
use candle_core::Device;
use candle_onnx::{self as candle_onnx, eval::Value as CValue, eval::simple_eval};
use half::f16;
use ndarray::{Array, Ix4, IxDyn};
use std::collections::HashMap;
use std::path::Path;

pub struct GimmVfi {
    model: candle_onnx::onnx::ModelProto,
    device: Device,
    pub use_fp16: bool,
}

impl GimmVfi {
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
    pub fn infer_fp32(
        &self,
        img_xs: Array<f32, IxDyn>,
        coord: Array<f32, IxDyn>,
        t: Array<f32, IxDyn>,
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
    pub fn infer_fp16(
        &self,
        img_xs: Array<f16, IxDyn>,
        coord: Array<f32, IxDyn>, // coord stays fp32 per model requirement
        t: Array<f16, IxDyn>,
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
        let device = &self.device;
        let shape: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = a.into_iter().collect();
        Ok(CValue::from_vec(data, shape.as_slice(), device)?)
    }

    // Convert ndarrays to flat Vec and build CValue (Tensor) for f16
    fn make_value_f16(&self, a: Array<f16, IxDyn>) -> Result<CValue> {
        let device = &self.device;
        let shape: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
        let data: Vec<f16> = a.into_iter().collect();
        Ok(CValue::from_vec(data, shape.as_slice(), device)?)
    }
}
