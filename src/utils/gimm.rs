use anyhow::Result;
use half::f16;
use ndarray::{Array, CowArray, Ix4, IxDyn};
use ort::Session;

pub struct GimmVfi {
    session: Session,
    pub use_fp16: bool,
}

impl GimmVfi {
    /// Create wrapper from an existing ONNX Runtime `Session`.
    /// `use_fp16` should be true if the model expects fp16 inputs.
    pub fn new(session: Session, use_fp16: bool) -> Self {
        Self { session, use_fp16 }
    }

    /// Run inference for FP32 inputs and return an owned 4D array [1, C, H, W]
    pub fn infer_fp32(
        &self,
        img_xs: Array<f32, IxDyn>,
        coord: Array<f32, IxDyn>,
        t: Array<f32, IxDyn>,
    ) -> Result<Array<f32, Ix4>> {
    let allocator = self.session.allocator();
    // Convert owned arrays into CowArray so Value::from_array accepts them
    let img_cow = CowArray::from(img_xs.into_dyn());
    let coord_cow = CowArray::from(coord.into_dyn());
    let t_cow = CowArray::from(t.into_dyn());
    let img_val = ort::Value::from_array(allocator, &img_cow)?;
    let coord_val = ort::Value::from_array(allocator, &coord_cow)?;
    let t_val = ort::Value::from_array(allocator, &t_cow)?;
        let outputs = self.session.run(vec![img_val, coord_val, t_val])?;
        let output = &outputs[0];
        let output_array = output.try_extract::<f32>()?;
        let owned = output_array.view().to_owned();
        let arr4 = owned.into_dimensionality::<Ix4>()?;
        Ok(arr4)
    }

    /// Run inference for FP16 inputs and return an owned 4D array [1, C, H, W]
    pub fn infer_fp16(
        &self,
        img_xs: Array<f16, IxDyn>,
        coord: Array<f32, IxDyn>, // coord stays fp32 per model requirement
        t: Array<f16, IxDyn>,
    ) -> Result<Array<f16, Ix4>> {
    let allocator = self.session.allocator();
    let img_cow = CowArray::from(img_xs.into_dyn());
    let coord_cow = CowArray::from(coord.into_dyn());
    let t_cow = CowArray::from(t.into_dyn());
    let img_val = ort::Value::from_array(allocator, &img_cow)?;
    let coord_val = ort::Value::from_array(allocator, &coord_cow)?;
    let t_val = ort::Value::from_array(allocator, &t_cow)?;
        let outputs = self.session.run(vec![img_val, coord_val, t_val])?;
        let output = &outputs[0];
        let output_array = output.try_extract::<f16>()?;
        let owned = output_array.view().to_owned();
        let arr4 = owned.into_dimensionality::<Ix4>()?;
        Ok(arr4)
    }
}
