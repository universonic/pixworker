use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=NCNN_SDK_DIR");
    let sdk = PathBuf::from(env::var_os("NCNN_SDK_DIR").expect("set NCNN_SDK_DIR"));
    let include = sdk.join("ncnn.framework/Headers/ncnn");
    println!(
        "cargo:rerun-if-changed={}",
        include.join("c_api.h").display()
    );
    println!("cargo:rustc-link-search=framework={}", sdk.display());
    for framework in ["ncnn", "glslang", "openmp"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    let moltenvk = sdk.join("MoltenVK/MoltenVK/dynamic/dylib/macOS");
    println!("cargo:rustc-link-search=native={}", moltenvk.display());
    println!("cargo:rustc-link-lib=dylib=MoltenVK");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", moltenvk.display());
    bindgen::Builder::default()
        .header(include.join("c_api.h").to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_function("ncnn_.*")
        .allowlist_type("ncnn_.*")
        .allowlist_var("NCNN_.*")
        .layout_tests(false)
        .generate()
        .expect("generate ncnn C API bindings")
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("ncnn.rs"))
        .expect("write ncnn C API bindings");
}
