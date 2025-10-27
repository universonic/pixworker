// Include the prost-generated ONNX types from OUT_DIR at build time.
// build.rs compiles the proto into $OUT_DIR/onnx.rs, so include it here.
include!(concat!(env!("OUT_DIR"), "/onnx.rs"));