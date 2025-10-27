# pixworker

[中文文档](README_zh_CN.md)

> Note: This project is in early development. Features are experimental and may be incomplete or unstable. Use for learning and research purposes only.

A video enhancement tool powered by ONNX Runtime, supporting frame interpolation and upscaling.

## Features

- **Frame Interpolation (VFI)**: Enhance video frame rate using GIMM-VFI models for smoother playback
- **Upscaling**: 4x or higher resolution enhancement using Real-ESRGAN models
- **Hardware Acceleration**: 
  - macOS: CoreML acceleration
  - Linux: CUDA / TensorRT acceleration
  - Windows: CUDA / TensorRT acceleration
- **NTSC Video Processing**: Deinterlacing support for interlaced video content

## Build Requirements

### Prerequisites
- **Rust**: Latest stable version (install via [rustup](https://rustup.rs/))
- **Make**: For build automation

### Platform-Specific Requirements

#### macOS
```bash
xcode-select --install
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt install build-essential

# Optional: CUDA and TensorRT for GPU acceleration
```

#### Windows
- Visual Studio 2019 or later (with C++ build tools)
- Or MinGW-w64 via [MSYS2](https://www.msys2.org/)

## Building

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd pixworker

# Build release version (recommended)
make

# Or build debug version
make debug
```

### Cross-Platform Compilation

```bash
# View all available targets
make help

# Build for specific platforms
make macos-x64      # macOS x86_64
make macos-arm64    # macOS Apple Silicon
make linux-x64      # Linux x86_64
make linux-arm64    # Linux ARM64

# Build all platforms and package
make dist
```

**Note**: 
- Windows binaries cannot be cross-compiled from macOS/Linux (due to `ring` crate limitations)
- Windows builds must be done on a Windows machine
- Cross-compiling Linux targets may require additional linker configuration

### Binary Locations

- Release build: `target/release/pixworker`
- Debug build: `target/debug/pixworker`
- Multi-platform package: `dist/<target-triple>/pixworker`

## Usage

### Model Download

On first run, you need to download ONNX model files. Place models in the following directories:

```
~/.cache/pixworker/models/vfi/       # Frame interpolation models
~/.cache/pixworker/models/upscale/   # Upscaling models
```

Or the program will automatically download them from huggingface.co.

### Basic Commands

```bash
# Frame interpolation - increase video frame rate to 60fps
pixworker enhance --vfi 60fps --vfi-model gimm-vfi-f-p-hf -i input.mp4 -o output.mp4

# Video upscaling - 4x resolution
pixworker enhance --upscale 4.0 --upscale-model realesr-animevideov3-hf -i input.mp4 -o output.mp4

# Show help
pixworker --help
```

## Development

```bash
# Run build
make

# Clean build artifacts
make clean
```

## License

### Source Code

The pixworker source code is licensed under the **MIT License**. See [LICENSE](LICENSE) file for details.

### AI Models

This project uses AI models with different licenses:

#### GIMM-VFI (Frame Interpolation)
- **License**: S-Lab License 1.0 (Non-Commercial)
- **Link**: https://github.com/GSeanCDAT/GIMM-VFI/blob/main/LICENSE
- **Restrictions**: 
  - ✅ Personal, non-commercial use
  - ✅ Academic research
  - ✅ Educational purposes
  - ❌ Commercial products or services
  - ❌ Profit-generating activities
- **Commercial Use**: Contact the contributors for permission

#### Real-ESRGAN (Upscaling)
- **License**: BSD 3-Clause License
- **Link**: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
- **Restrictions**:
  - ✅ Commercial use allowed
  - ✅ Modification allowed
  - ✅ Distribution allowed
  - ⚠️ Must include copyright notice and license text

### Usage Guidelines

**Important**: Due to GIMM-VFI's non-commercial license:
- **If you use VFI (frame interpolation)**: The software **cannot be used for commercial purposes**
- **If you only use upscaling**: Commercial use is permitted under BSD 3-Clause terms

The software will prompt you to accept the respective model license when downloading models for the first time.
