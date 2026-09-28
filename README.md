# pixworker

[中文文档](README_zh_CN.md)

A streaming video enhancement tool using native tch/libtorch inference and FFmpeg 9 shared libraries for decoding and encoding.

## Features

- **Frame Interpolation (VFI)**: Practical-RIFE v4.25 (MIT), supporting arbitrary intermediate timestamps
- **Upscaling**: Eight Real-ESRGAN safetensors variants: `realesr-animevideov3`, `realesr-animevideov3-hf`, `realesr-generalx4v3`, `realesr-generalx4v3-hf`, `realesrgan-x4plus`, `realesrgan-x4plus-hf`, `realesrgan-x4plus-anime`, `realesrgan-x4plus-anime-hf`
- **Native platforms**: macOS arm64 (MPS), Linux x86_64 and Windows x86_64 (CUDA when available, otherwise CPU)

## Build Requirements

- Rust 1.88 or newer (install via [rustup](https://rustup.rs/))
- Make and Bash; on Windows, use MSYS2 Make/Bash with the MSVC Rust toolchain and Visual Studio C++ build tools
- libclang (for bindgen) and FFmpeg 9.x shared development libraries/headers; on macOS/Linux, make them visible to pkg-config
- Official libtorch **2.11.0** matching the host architecture (and CUDA driver, if using CUDA). Set `LIBTORCH` to its extracted directory, not its `lib/` subdirectory. Windows also requires `FFMPEG_DIR` pointing to an FFmpeg 9 shared development package.
- At runtime, FFmpeg 9.x system shared libraries with libx265 are required; FFmpeg is not bundled. `make dist` packages the dynamic libtorch libraries with the binary.

The tch 0.26/libtorch 2.13 spike measured **200.567 ms** median MPS inference against a **110.4 ms** baseline. The fallback to tch 0.24/libtorch 2.11.0 measured **176.583 ms** and **154.261 ms** medians; it is still slower than the baseline.

## Building

Build on the target host only; cross-compilation and Linux/Windows ARM64 are not supported. Install the host Rust target with rustup if needed.

```bash
export LIBTORCH="/path/to/libtorch-2.11.0"
# Windows (MSYS2 Bash): also export FFMPEG_DIR="/path/to/ffmpeg-9-shared-dev"
make                 # native release build
make dist            # package in dist/<target-triple>/
make help            # show native platform targets
```

The named targets `make macos-arm64`, `make linux-x64`, and `make windows-x64` build and package only on their matching hosts. The binary is in `target/<target-triple>/release/pixworker` (or `pixworker.exe` on Windows); the package is in `dist/<target-triple>/`. Packaging does not include FFmpeg or remove existing `dist` contents.

For unbundled local builds, set `DYLD_LIBRARY_PATH="$LIBTORCH/lib"` on macOS or `LD_LIBRARY_PATH="$LIBTORCH/lib"` on Linux when running the binary or `cargo test`. On Windows, add `%LIBTORCH%/lib` to `PATH`. `PYTORCH_ENABLE_MPS_FALLBACK=0` prevents silent CPU fallback during MPS validation.

## Usage

### Models

Models are `.safetensors` files, cached in `~/.cache/pixworker/models/upscale/` and `~/.cache/pixworker/models/vfi/`. Missing Real-ESRGAN weights are downloaded after license confirmation. RIFE uses `rife-v4.25_fp32.safetensors` in the VFI cache; its Hugging Face URL currently returns **404** until the user uploads the converted weights to `universonic/RIFE`. Until then, place that file in the VFI cache locally before using interpolation.

### Basic Commands

```bash
# Interpolate to 60 fps (RIFE v4.25 is the default VFI model)
pixworker enhance --vfi 60fps --upscale 1.0 -i input.mp4 -o output.mp4

# Upscale to 4x resolution
pixworker enhance --upscale 4.0 --upscale-model realesr-animevideov3-hf -i input.mp4 -o output.mp4

pixworker --help
```

### Model Parity (Development)

`examples/parity.rs` dumps raw RGB24 frames and checks the eight upscale models and RIFE against CPU FP32 `.f32` and quantized `.u8` references from the official PyTorch implementations. Generate those references once in a temporary Python environment under `tmp/`; Python is not a product dependency. Place the reference files next to the dumped frames, then run:

```bash
DYLD_LIBRARY_PATH="$LIBTORCH/lib" cargo run --example parity -- dump input.mp4 tmp/ref
DYLD_LIBRARY_PATH="$LIBTORCH/lib" cargo run --example parity -- check tmp/ref cpu
# On macOS arm64 with MPS and the quantized reference files:
DYLD_LIBRARY_PATH="$LIBTORCH/lib" PYTORCH_ENABLE_MPS_FALLBACK=0 cargo run --example parity -- check tmp/ref mps
```

## License

The pixworker source code is licensed under the **MIT License**. See [LICENSE](LICENSE).

- **Practical-RIFE v4.25** (frame interpolation): MIT, [license](https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE). Commercial use is allowed subject to the license notice.
- **Real-ESRGAN** (upscaling): BSD 3-Clause, [license](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE). Commercial use is allowed subject to the copyright and license notices.

The program prompts for the applicable model license before downloading missing weights.
