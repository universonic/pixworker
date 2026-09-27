use anyhow::{Context, Result, ensure};
use image::RgbImage;
use std::{ffi::CString, path::Path};

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/ncnn.rs"));
}

pub const OUTPUT_SHAPE: [usize; 4] = [1, 3, 2880, 5120];

pub struct Net {
    ptr: ffi::ncnn_net_t,
    input: i32,
    output: i32,
}

impl Drop for Net {
    fn drop(&mut self) {
        unsafe { ffi::ncnn_net_destroy(self.ptr) };
    }
}

pub struct Mat(ffi::ncnn_mat_t);

impl Drop for Mat {
    fn drop(&mut self) {
        unsafe { ffi::ncnn_mat_destroy(self.0) };
    }
}

struct Extractor(ffi::ncnn_extractor_t);

impl Drop for Extractor {
    fn drop(&mut self) {
        unsafe { ffi::ncnn_extractor_destroy(self.0) };
    }
}

pub fn load_model(param: &Path, bin: &Path, vulkan: bool) -> Result<Net> {
    if vulkan {
        ensure!(
            std::env::var_os("NCNN_VULKAN_DRIVER").is_some(),
            "set NCNN_VULKAN_DRIVER to a verified MoltenVK dylib"
        );
    }
    let ptr = unsafe { ffi::ncnn_net_create() };
    ensure!(!ptr.is_null(), "ncnn_net_create failed");
    let mut net = Net {
        ptr,
        input: -1,
        output: -1,
    };
    let opt = unsafe { ffi::ncnn_net_get_option(ptr) };
    ensure!(!opt.is_null(), "ncnn_net_get_option failed");
    unsafe {
        ffi::ncnn_option_set_num_threads(opt, 1);
        ffi::ncnn_option_set_use_fp16_packed(opt, 0);
        ffi::ncnn_option_set_use_fp16_storage(opt, 0);
        ffi::ncnn_option_set_use_fp16_arithmetic(opt, 0);
        ffi::ncnn_option_set_use_bf16_packed(opt, 0);
        ffi::ncnn_option_set_use_bf16_storage(opt, 0);
        ffi::ncnn_option_set_use_int8_packed(opt, 0);
        ffi::ncnn_option_set_use_int8_storage(opt, 0);
        ffi::ncnn_option_set_use_int8_arithmetic(opt, 0);
        ffi::ncnn_option_set_use_vulkan_compute(opt, i32::from(vulkan));
        if vulkan {
            ffi::ncnn_net_set_vulkan_device(ptr, 0);
        }
    }
    let param = CString::new(param.to_string_lossy().as_bytes())?;
    let bin = CString::new(bin.to_string_lossy().as_bytes())?;
    ensure!(
        unsafe { ffi::ncnn_net_load_param(ptr, param.as_ptr()) } == 0,
        "load_param failed"
    );
    ensure!(
        unsafe { ffi::ncnn_net_load_model(ptr, bin.as_ptr()) } == 0,
        "load_model failed"
    );
    ensure!(
        unsafe { ffi::ncnn_net_get_input_count(ptr) } == 1,
        "expected one input"
    );
    ensure!(
        unsafe { ffi::ncnn_net_get_output_count(ptr) } == 1,
        "expected one output"
    );
    net.input = unsafe { ffi::ncnn_net_get_input_index(ptr, 0) };
    net.output = unsafe { ffi::ncnn_net_get_output_index(ptr, 0) };
    ensure!(
        net.input >= 0 && net.output >= 0 && net.input != net.output,
        "invalid blob indices"
    );
    ensure!(
        unsafe { ffi::ncnn_option_get_use_vulkan_compute(opt) } == i32::from(vulkan),
        "ncnn fell back from Vulkan to CPU"
    );
    let precision = unsafe {
        (
            ffi::ncnn_option_get_use_fp16_packed(opt),
            ffi::ncnn_option_get_use_fp16_storage(opt),
            ffi::ncnn_option_get_use_fp16_arithmetic(opt),
            ffi::ncnn_option_get_use_int8_storage(opt),
            ffi::ncnn_option_get_use_int8_arithmetic(opt),
        )
    };
    ensure!(
        precision == (0, 0, 0, 0, 0),
        "low-precision option enabled: {precision:?}"
    );
    println!(
        "ncnn={} backend={} fp16=off input_blob={} output_blob={}",
        unsafe { std::ffi::CStr::from_ptr(ffi::ncnn_version()).to_string_lossy() },
        if vulkan { "Vulkan" } else { "CPU" },
        net.input,
        net.output
    );
    Ok(net)
}

pub fn prepare_frame(image: &RgbImage) -> Result<Mat> {
    ensure!(image.dimensions() == (1280, 720), "expected 1280x720 RGB");
    let raw = unsafe { ffi::ncnn_mat_create_3d(1280, 720, 3, std::ptr::null_mut()) };
    ensure!(!raw.is_null(), "allocate input Mat failed");
    let input = Mat(raw);
    ensure!(
        unsafe { ffi::ncnn_mat_get_elemsize(raw) } == 4,
        "input Mat not FP32"
    );
    ensure!(
        unsafe { ffi::ncnn_mat_get_elempack(raw) } == 1,
        "input Mat not planar"
    );
    let channels = (0..3)
        .map(|c| unsafe { ffi::ncnn_mat_get_channel_data(raw, c) as *mut f32 })
        .collect::<Vec<_>>();
    ensure!(
        channels.iter().all(|p| !p.is_null()),
        "input channel missing"
    );
    for (i, pixel) in image.as_raw().chunks_exact(3).enumerate() {
        for c in 0..3 {
            unsafe { channels[c].add(i).write(pixel[c] as f32 / 255.0) };
        }
    }
    Ok(input)
}

pub fn infer(net: &Net, input: &Mat, output: &mut [f32]) -> Result<()> {
    ensure!(
        output.len() == OUTPUT_SHAPE.iter().product::<usize>(),
        "wrong readback length"
    );
    let ex = unsafe { ffi::ncnn_extractor_create(net.ptr) };
    ensure!(!ex.is_null(), "create extractor failed");
    let ex = Extractor(ex);
    ensure!(
        unsafe { ffi::ncnn_extractor_input_index(ex.0, net.input, input.0) } == 0,
        "input failed"
    );
    let mut raw_output = std::ptr::null_mut();
    let status = unsafe { ffi::ncnn_extractor_extract_index(ex.0, net.output, &mut raw_output) };
    ensure!(!raw_output.is_null(), "missing output Mat");
    let mat = Mat(raw_output);
    ensure!(status == 0, "extract failed: {status}");
    let shape = unsafe {
        (
            ffi::ncnn_mat_get_dims(raw_output),
            ffi::ncnn_mat_get_w(raw_output),
            ffi::ncnn_mat_get_h(raw_output),
            ffi::ncnn_mat_get_d(raw_output),
            ffi::ncnn_mat_get_c(raw_output),
        )
    };
    ensure!(
        shape == (3, 5120, 2880, 1, 3),
        "unexpected output shape {shape:?}"
    );
    ensure!(
        unsafe { ffi::ncnn_mat_get_elemsize(raw_output) } == 4,
        "output not FP32"
    );
    ensure!(
        unsafe { ffi::ncnn_mat_get_elempack(raw_output) } == 1,
        "output not planar"
    );
    ensure!(
        unsafe { ffi::ncnn_mat_get_cstep(raw_output) } >= 5120 * 2880,
        "short output plane"
    );
    for c in 0..3usize {
        let plane = unsafe { ffi::ncnn_mat_get_channel_data(mat.0, c as i32) as *const f32 };
        ensure!(!plane.is_null(), "output channel missing");
        unsafe {
            std::ptr::copy_nonoverlapping(
                plane,
                output[c * 5120 * 2880..].as_mut_ptr(),
                5120 * 2880,
            )
        };
    }
    Ok(())
}

pub fn check_reference(output: &[f32], path: &Path) -> Result<()> {
    let reference = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        reference.len() == output.len() * 4,
        "reference length mismatch"
    );
    let mut sum = 0.0_f64;
    let mut max = 0.0_f64;
    for (bytes, &value) in reference.chunks_exact(4).zip(output) {
        let error = (f32::from_le_bytes(bytes.try_into()?) - value).abs() as f64;
        ensure!(error.is_finite(), "non-finite output");
        sum += error * error;
        max = max.max(error);
    }
    let mse = sum / output.len() as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    };
    println!("float_mse={mse:.12e} max_abs={max:.12e} psnr_0_1_db={psnr:.3}");
    ensure!(
        psnr >= 100.0 && max <= 0.001,
        "not equivalent to ORT CPU; timing disqualified"
    );
    Ok(())
}
