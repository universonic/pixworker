fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download the canonical ONNX proto from the Candle repo on each build so we
    // always compile against the latest definition.
    let url = "https://raw.githubusercontent.com/huggingface/candle/refs/heads/main/candle-onnx/src/onnx.proto3";
    println!("Downloading ONNX proto from {}", url);

    // Fetch the proto file
    let mut resp = ureq::get(url).call()?;
    if resp.status() != 200 {
        return Err(format!("failed to download onnx proto: HTTP {}", resp.status()).into());
    }
    let body = resp.body_mut().read_to_string()?;
    let proto_text = body.as_str();

    // Ensure proto directory exists and write the fetched proto as onnx.proto
    std::fs::create_dir_all("src/proto/onnx")?;
    std::fs::write("src/proto/onnx/onnx.proto", &proto_text)?;

    // Compile the fetched proto into Rust types (write to OUT_DIR so the project
    // can include them with concat!(env!("OUT_DIR"), "/onnx.rs")).
    let protos = ["src/proto/onnx/onnx.proto"];
    let includes = ["src/proto/onnx"];
    prost_build::Config::new().compile_protos(&protos, &includes)?;
    println!("cargo:rerun-if-changed=src/proto/onnx/onnx.proto");
    Ok(())
}
