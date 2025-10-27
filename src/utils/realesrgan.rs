use anyhow::Result;
use half::f16;
use ndarray::{Array, CowArray, Ix4, IxDyn};
use ort::Session;

pub struct RealESRGAN {
    session: Session,
    pub use_fp16: bool,
    pub supports_denoise: bool,
}

impl RealESRGAN {
    pub fn new(session: Session, use_fp16: bool, supports_denoise: bool) -> Self {
        Self { session, use_fp16, supports_denoise }
    }

    /// Run FP32 inference. `img` shape expected: [1, C, H, W]
    /// If `supports_denoise` is true, `denoise` should be shape [1]
    pub fn infer_fp32(&self, img: Array<f32, IxDyn>, denoise: Array<f32, IxDyn>) -> Result<Array<f32, Ix4>> {
        let allocator = self.session.allocator();
        let img_cow = CowArray::from(img.into_dyn());
        let denoise_cow = CowArray::from(denoise.into_dyn());
        let img_val = ort::Value::from_array(allocator, &img_cow)?;
        let denoise_val = ort::Value::from_array(allocator, &denoise_cow)?;

        let outputs = if self.supports_denoise {
            self.session.run(vec![img_val, denoise_val])?
        } else {
            self.session.run(vec![img_val])?
        };

        let output = &outputs[0];
        let output_array = output.try_extract::<f32>()?;
        let owned = output_array.view().to_owned();
        let arr4 = owned.into_dimensionality::<Ix4>()?;
        Ok(arr4)
    }

    /// Run FP16 inference.
    pub fn infer_fp16(&self, img: Array<f16, IxDyn>, denoise: Array<f16, IxDyn>) -> Result<Array<f16, Ix4>> {
        let allocator = self.session.allocator();
        let img_cow = CowArray::from(img.into_dyn());
        let denoise_cow = CowArray::from(denoise.into_dyn());
        let img_val = ort::Value::from_array(allocator, &img_cow)?;
        let denoise_val = ort::Value::from_array(allocator, &denoise_cow)?;

        let outputs = if self.supports_denoise {
            self.session.run(vec![img_val, denoise_val])?
        } else {
            self.session.run(vec![img_val])?
        };

        let output = &outputs[0];
        let output_array = output.try_extract::<f16>()?;
        let owned = output_array.view().to_owned();
        let arr4 = owned.into_dimensionality::<Ix4>()?;
        Ok(arr4)
    }
}
