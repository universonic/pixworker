# pixworker

[English Documentation](README.md)

基于原生 tch/libtorch 推理与 FFmpeg 9 共享库编解码的流式视频增强工具。

## 功能特性

- **帧插值 (VFI)**：Practical-RIFE v4.25（MIT），支持任意中间时间点
- **分辨率放大 (Upscale)**：8 个 Real-ESRGAN safetensors 变体：`realesr-animevideov3`、`realesr-animevideov3-hf`、`realesr-generalx4v3`、`realesr-generalx4v3-hf`、`realesrgan-x4plus`、`realesrgan-x4plus-hf`、`realesrgan-x4plus-anime`、`realesrgan-x4plus-anime-hf`
- **本机平台**：macOS arm64（MPS）、Linux x86_64 和 Windows x86_64（可用时使用 CUDA，否则使用 CPU）

## 编译环境要求

- Rust 1.88 或更新版本（通过 [rustup](https://rustup.rs/) 安装）
- Make 和 Bash；Windows 使用 MSYS2 Make/Bash、MSVC Rust 工具链及 Visual Studio C++ 构建工具
- libclang（供 bindgen 使用）和 FFmpeg 9.x 共享开发库及头文件；macOS/Linux 上需确保 pkg-config 能找到它们
- 与本机架构（如使用 CUDA，还需与驱动）匹配的官方 libtorch **2.11.0**；`LIBTORCH` 指向解压目录，而不是其中的 `lib/`。Windows 还需将 `FFMPEG_DIR` 指向 FFmpeg 9 的 shared 开发包。
- 运行时需要系统提供含 libx265 的 FFmpeg 9.x 共享库；FFmpeg 不随程序打包。`make dist` 会将 libtorch 动态库与二进制文件一起打包。

tch 0.26/libtorch 2.13 在 MPS 上的试验推理中位数为 **200.567 ms**，基准为 **110.4 ms**。回退到 tch 0.24/libtorch 2.11.0 后测得 **176.583 ms** 和 **154.261 ms**，仍慢于基准。

## 编译

只能在目标主机上原生构建；不支持交叉编译及 Linux/Windows ARM64。如有需要，先用 rustup 安装本机目标。

```bash
export LIBTORCH="/path/to/libtorch-2.11.0"
# Windows（MSYS2 Bash）：还需 export FFMPEG_DIR="/path/to/ffmpeg-9-shared-dev"
make                 # 本机发布版本
make dist            # 打包到 dist/<target-triple>/
make help            # 查看本机平台目标
```

`make macos-arm64`、`make linux-x64` 和 `make windows-x64` 仅能在对应主机上构建并打包。二进制文件位于 `target/<target-triple>/release/pixworker`（Windows 为 `pixworker.exe`），打包目录为 `dist/<target-triple>/`。打包不包含 FFmpeg，也不会删除 `dist` 中的已有内容。

直接运行未打包的本机构建时，macOS 使用 `DYLD_LIBRARY_PATH="$LIBTORCH/lib"`，Linux 使用 `LD_LIBRARY_PATH="$LIBTORCH/lib"`，Windows 将 `%LIBTORCH%/lib` 加入 `PATH`；运行 `cargo test` 时同样需要设置。MPS 验证时设置 `PYTORCH_ENABLE_MPS_FALLBACK=0`，避免静默回退到 CPU。

## 使用

### 模型

模型为 `.safetensors` 文件，缓存于 `~/.cache/pixworker/models/upscale/` 和 `~/.cache/pixworker/models/vfi/`。缺少 Real-ESRGAN 权重时，程序会在许可确认后下载。RIFE 使用 VFI 缓存中的 `rife-v4.25_fp32.safetensors`；在用户将转换后的权重上传到 `universonic/RIFE` 前，Hugging Face 下载地址目前返回 **404**。插帧前需暂时将该文件放入本地 VFI 缓存。

### 基本命令

```bash
# 插帧到 60 fps（默认插帧模型为 RIFE v4.25）
pixworker enhance --vfi 60fps --upscale 1.0 -i input.mp4 -o output.mp4

# 放大 4 倍
pixworker enhance --upscale 4.0 --upscale-model realesr-animevideov3-hf -i input.mp4 -o output.mp4

pixworker --help
```

### 模型等价验证（开发）

`examples/parity.rs` 导出原始 RGB24 帧，并将 8 个放大模型与 RIFE 的输出同官方 PyTorch 实现生成的 CPU FP32 `.f32` 和量化 `.u8` 参考比较。参考文件只需在 `tmp/` 下的临时 Python 环境中生成一次；产品运行不依赖 Python。将参考文件放在导出帧旁，然后运行：

```bash
DYLD_LIBRARY_PATH="$LIBTORCH/lib" cargo run --example parity -- dump input.mp4 tmp/ref
DYLD_LIBRARY_PATH="$LIBTORCH/lib" cargo run --example parity -- check tmp/ref cpu
# 在有 MPS 和量化参考文件的 macOS arm64 上：
DYLD_LIBRARY_PATH="$LIBTORCH/lib" PYTORCH_ENABLE_MPS_FALLBACK=0 cargo run --example parity -- check tmp/ref mps
```

## 许可证

pixworker 源代码采用 **MIT 许可证**。详见 [LICENSE](LICENSE)。

- **Practical-RIFE v4.25**（帧插值）：MIT，[许可证](https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE)。遵守许可声明即可商用。
- **Real-ESRGAN**（放大）：BSD 3-Clause，[许可证](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)。遵守版权及许可声明即可商用。

缺少模型权重时，程序会在下载前提示确认相应的模型许可。
