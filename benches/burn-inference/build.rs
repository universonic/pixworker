use std::path::PathBuf;

fn main() {
    let model = PathBuf::from(std::env::var_os("HOME").expect("HOME is required"))
        .join(".cache/pixworker/models/upscale/realesr-animevideov3_fp32.onnx");
    assert!(
        model.is_file(),
        "missing local ONNX model: {}",
        model.display()
    );
    println!("cargo:rerun-if-changed={}", model.display());

    burn_onnx::ModelGen::new()
        .input(model.to_str().expect("non-UTF-8 model path"))
        .out_dir("model")
        .run_from_script();
}
