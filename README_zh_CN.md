# pixworker

[English Documentation](README.md)

> 注意：本项目处于早期开发阶段，功能仍在测试与开发中，可能存在未完成或不稳定的行为。仅限学习交流使用。

基于 ONNX Runtime 的视频增强工具，支持帧插值和分辨率放大。

## 功能特性

- **帧插值 (VFI)**: 使用 GIMM-VFI 模型提升视频帧率，让视频更流畅
- **分辨率放大 (Upscale)**: 使用 Real-ESRGAN 模型进行 4x 或更高倍数的视频放大
- **多平台加速**: 
  - macOS: CoreML 加速
  - Linux: CUDA / TensorRT 加速
  - Windows: CUDA / TensorRT acceleration
- **NTSC 视频处理**: 支持隔行扫描视频的去隔行处理

## 编译环境要求

### 基础要求
- **Rust**: 最新稳定版 (通过 [rustup](https://rustup.rs/) 安装)
- **Make**: 用于构建脚本

### 平台特定要求

#### macOS
```bash
xcode-select --install
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt install build-essential

# 可选: CUDA 和 TensorRT (用于 GPU 加速)
```

#### Windows
- Visual Studio 2019 或更新版本 (带 C++ 构建工具)
- 或使用 MinGW-w64 (通过 [MSYS2](https://www.msys2.org/) 安装)

## 编译

### 快速开始

```bash
# 克隆仓库
git clone <repository-url>
cd pixworker

# 编译发布版本 (推荐)
make

# 或编译调试版本
make debug
```

### 跨平台编译

```bash
# 查看所有可用目标
make help

# 编译特定平台
make macos-x64      # macOS x86_64
make macos-arm64    # macOS Apple Silicon
make linux-x64      # Linux x86_64
make linux-arm64    # Linux ARM64

# 编译所有平台并打包
make dist
```

**注意**: 
- 在 macOS/Linux 上无法交叉编译 Windows 版本 (由于 `ring` 加密库依赖限制)
- Windows 版本需要在 Windows 系统上编译
- 跨平台编译 Linux 目标可能需要额外的链接器配置

### 编译后的二进制文件位置

- 发布版本: `target/release/pixworker`
- 调试版本: `target/debug/pixworker`
- 多平台打包: `dist/<target-triple>/pixworker`

## 使用

### 模型下载

首次运行时，需要下载 ONNX 模型文件。将模型放置在以下目录：

```
~/.cache/pixworker/models/vfi/       # 帧插值模型
~/.cache/pixworker/models/upscale/   # 放大模型
```

或者程序会自动从 huggingface.co 下载。

### 基本命令

```bash
# 帧插值 - 将视频帧率提升到 60fps
pixworker enhance --vfi 60fps --vfi-model gimm-vfi-f-p-hf -i input.mp4 -o output.mp4

# 视频放大 - 4倍放大
pixworker enhance --upscale 4.0 --upscale-model realesr-animevideov3-hf -i input.mp4 -o output.mp4

# 查看帮助
pixworker --help
```

## 开发

```bash
# 运行编译
make

# 清理构建产物
make clean
```

## 许可证

### 源代码

pixworker 源代码采用 **MIT 许可证**。详见 [LICENSE](LICENSE) 文件。

### AI 模型

本项目使用的 AI 模型具有不同的许可证：

#### GIMM-VFI（帧插值）
- **许可证**: S-Lab License 1.0（非商业）
- **链接**: https://github.com/GSeanCDAT/GIMM-VFI/blob/main/LICENSE
- **限制**: 
  - ✅ 个人非商业用途
  - ✅ 学术研究
  - ✅ 教育目的
  - ❌ 商业产品或服务
  - ❌ 盈利性活动
- **商业使用**: 需联系贡献者获得许可

#### Real-ESRGAN（放大）
- **许可证**: BSD 3-Clause License
- **链接**: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
- **限制**:
  - ✅ 允许商业使用
  - ✅ 允许修改
  - ✅ 允许分发
  - ⚠️ 必须包含版权声明和许可证文本

### 使用指南

**重要提示**：由于 GIMM-VFI 的非商业许可证：
- **如果使用 VFI（帧插值）功能**：软件**不能用于商业目的**
- **如果仅使用放大功能**：允许在 BSD 3-Clause 条款下商业使用

首次下载模型时，软件会提示您接受相应的模型许可证。
