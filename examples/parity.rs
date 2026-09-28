//! Development-only parity harness. Run `cargo run --example parity --` for usage.
use anyhow::{Context, Result, bail, ensure};
use ffmpeg_next as ffmpeg;
use pixworker::utils::{ffmpeg::VideoDecoder, vfi::Rife};
use std::{
    env, fs,
    io::{self, Write},
    path::Path,
};
use tch::{Device, Kind, Tensor};

// The production API returns u8 only. Compile the same source here to expose
// the pre-quantization output without changing production modules.
#[allow(dead_code)]
mod float_upscale {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/utils/upscale.rs"));

    impl Upscaler {
        pub fn float(&self, input: Tensor) -> anyhow::Result<Tensor> {
            let _guard = tch::no_grad_guard();
            let mut x = input.f_to_kind(self.kind)?;
            for _ in 0..self.passes {
                x = match &self.network {
                    Network::Srvgg(model) => model.forward(&x)?,
                    Network::Rrdb(model) => model.forward(&x)?,
                };
            }
            Ok(x.f_to_kind(Kind::Float)?)
        }
    }
}
use float_upscale::Upscaler;

const FRAMES: [usize; 6] = [0, 1, 60, 61, 118, 119];
const PAIRS: [(usize, usize); 3] = [(0, 1), (60, 61), (118, 119)];
const TIMES: [f32; 3] = [0.4, 0.5, 0.8];
const KEYS: [&str; 8] = [
    "realesr-animevideov3",
    "realesr-animevideov3-hf",
    "realesr-generalx4v3",
    "realesr-generalx4v3-hf",
    "realesrgan-x4plus",
    "realesrgan-x4plus-hf",
    "realesrgan-x4plus-anime",
    "realesrgan-x4plus-anime-hf",
];

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [_, "dump", video, dir] => dump(Path::new(video), Path::new(dir)),
        [_, "raw", video] => raw(Path::new(video)),
        [_, "check", dir] => check(Path::new(dir), "all"),
        [_, "check", dir, mode @ ("all" | "cpu" | "mps")] => check(Path::new(dir), mode),
        _ => bail!(
            "usage: parity dump <video> <tmp/ref> | parity raw <video> | parity check <tmp/ref> [all|cpu|mps]"
        ),
    }
}

fn raw(video: &Path) -> Result<()> {
    let mut output = io::stdout().lock();
    let mut count = 0;
    for result in VideoDecoder::new(video)? {
        let frame = result?;
        let row = frame.width() as usize * 3;
        for line in frame
            .data(0)
            .chunks(frame.stride(0))
            .take(frame.height() as usize)
        {
            output.write_all(&line[..row])?;
        }
        count += 1;
    }
    eprintln!("decoded {count} RGB24 frames");
    Ok(())
}

fn dump(video: &Path, dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)?;
    let mut decoder = VideoDecoder::new(video)?;
    let mut size = None;
    let mut found = 0;
    for (index, result) in (&mut decoder).enumerate() {
        let frame = result?;
        if index > FRAMES[5] {
            break;
        }
        if !FRAMES.contains(&index) {
            continue;
        }
        ensure!(
            frame.format() == ffmpeg::format::Pixel::RGB24,
            "expected RGB24"
        );
        let dimensions = (frame.width(), frame.height());
        ensure!(
            size.is_none_or(|previous| previous == dimensions),
            "frame size changed"
        );
        size = Some(dimensions);
        let row = frame.width() as usize * 3;
        let bytes: Vec<u8> = frame
            .data(0)
            .chunks(frame.stride(0))
            .take(frame.height() as usize)
            .flat_map(|line| line[..row].iter().copied())
            .collect();
        let path = dir.join(format!("frame-{index:03}.rgb"));
        fs::write(&path, bytes)?;
        println!("{}", path.display());
        found += 1;
    }
    let (width, height) = size.context("video ended before frame 119")?;
    ensure!(found == FRAMES.len(), "video ended before frame 119");
    fs::write(dir.join("dimensions.txt"), format!("{width} {height}\n"))?;
    Ok(())
}

fn input(dir: &Path, index: usize, width: usize, height: usize, device: Device) -> Result<Tensor> {
    let path = dir.join(format!("frame-{index:03}.rgb"));
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() == width * height * 3,
        "{}: wrong RGB24 size",
        path.display()
    );
    Ok(Tensor::f_from_slice(&bytes)?
        .f_view([height as i64, width as i64, 3])?
        .f_to_device(device)?
        .f_to_kind(Kind::Float)?
        .f_div_scalar(255.0)?
        .f_permute([2, 0, 1])?
        .f_unsqueeze(0)?)
}

fn floats(tensor: &Tensor) -> Result<Vec<f32>> {
    let cpu = tensor
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Float)?
        .f_contiguous()?;
    let mut out = vec![0f32; cpu.numel()];
    cpu.f_copy_data(&mut out, cpu.numel())?;
    Ok(out)
}

fn bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    let cpu = tensor.f_to_device(Device::Cpu)?.f_contiguous()?;
    ensure!(cpu.kind() == Kind::Uint8, "expected u8 output");
    let mut out = vec![0u8; cpu.numel()];
    cpu.f_copy_data(&mut out, cpu.numel())?;
    Ok(out)
}

fn reference_f32(path: &Path, len: usize) -> Result<Vec<f32>> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(raw.len() == len * 4, "{}: wrong float size", path.display());
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn psnr(mse: f64, peak: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    }
}

fn strict(dir: &Path, name: &str, actual: &Tensor, len: usize) -> Result<()> {
    let expected = reference_f32(&dir.join(format!("{name}.f32")), len)?;
    let observed = floats(actual)?;
    ensure!(observed.len() == len, "{name}: wrong output shape");
    let mut max_abs = 0.0f64;
    let mut sum = 0.0f64;
    for (a, b) in observed.iter().zip(expected) {
        ensure!(a.is_finite() && b.is_finite(), "{name}: non-finite output");
        let error = (*a as f64 - b as f64).abs();
        max_abs = max_abs.max(error);
        sum += error * error;
    }
    let db = psnr(sum / len as f64, 1.0);
    println!("{name} float: PSNR={db:.3} dB max_abs={max_abs:.7}");
    ensure!(
        db >= 100.0 && max_abs <= 1e-3,
        "{name}: float parity failed"
    );
    Ok(())
}

// Non-overlapping 8x8 RGB windows, uniform weights, population covariance.
fn ssim(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    let (mut score, mut windows) = (0.0, 0);
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            for channel in 0..3 {
                let (mut sa, mut sb, mut aa, mut bb, mut ab, mut n) =
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                for yy in y..(y + 8).min(height) {
                    for xx in x..(x + 8).min(width) {
                        let i = (yy * width + xx) * 3 + channel;
                        let (v, w) = (a[i] as f64, b[i] as f64);
                        sa += v;
                        sb += w;
                        aa += v * v;
                        bb += w * w;
                        ab += v * w;
                        n += 1.0;
                    }
                }
                let (ma, mb) = (sa / n, sb / n);
                let (va, vb, cov) = (
                    (aa / n - ma * ma).max(0.0),
                    (bb / n - mb * mb).max(0.0),
                    ab / n - ma * mb,
                );
                score += (2.0 * ma * mb + 6.5025) * (2.0 * cov + 58.5225)
                    / ((ma * ma + mb * mb + 6.5025) * (va + vb + 58.5225));
                windows += 1;
            }
        }
    }
    score / windows as f64
}

fn quantized(dir: &Path, name: &str, actual: &Tensor, width: usize, height: usize) -> Result<()> {
    let path = dir.join(format!("{name}.u8"));
    let expected = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let observed = bytes(actual)?;
    ensure!(
        expected.len() == width * height * 3 && observed.len() == expected.len(),
        "{name}: wrong u8 size"
    );
    let mse = observed
        .iter()
        .zip(&expected)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / expected.len() as f64;
    let db = psnr(mse, 255.0);
    let structural = ssim(&observed, &expected, width, height);
    println!("{name} u8: PSNR={db:.3} dB SSIM={structural:.6}");
    ensure!(
        db >= 45.0 && structural >= 0.995,
        "{name}: MPS u8 parity failed"
    );
    Ok(())
}

fn require(dir: &Path, name: &str, ext: &str, size: usize) -> Result<()> {
    let path = dir.join(format!("{name}.{ext}"));
    ensure!(
        fs::metadata(&path)
            .with_context(|| format!("missing {}", path.display()))?
            .len()
            == size as u64,
        "{}: wrong reference size",
        path.display()
    );
    Ok(())
}

fn check(dir: &Path, mode: &str) -> Result<()> {
    let dims = fs::read_to_string(dir.join("dimensions.txt"))?;
    let mut parts = dims.split_whitespace();
    let width: usize = parts.next().context("missing width")?.parse()?;
    let height: usize = parts.next().context("missing height")?.parse()?;
    ensure!(
        width > 0 && height > 0 && parts.next().is_none(),
        "invalid dimensions.txt"
    );
    for index in FRAMES {
        require(dir, &format!("frame-{index:03}"), "rgb", width * height * 3)?;
    }
    for key in KEYS {
        for index in [0, 60, 119] {
            let name = format!("{key}-frame-{index:03}");
            if mode != "mps" || key == "realesr-animevideov3" {
                require(dir, &name, "f32", width * height * 3 * 16 * 4)?;
            }
            if mode != "cpu" {
                require(dir, &name, "u8", width * height * 3 * 16)?;
            }
        }
    }
    for (a, b) in PAIRS {
        for t in TIMES {
            let name = format!("rife-{a:03}-{b:03}-t{t:.1}");
            if mode != "mps" {
                require(dir, &name, "f32", width * height * 3 * 4)?;
            }
            if mode != "cpu" {
                require(dir, &name, "u8", width * height * 3)?;
            }
        }
    }
    let root = env::var("PIXWORKER_UPSCALE_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or(
            dirs::home_dir()
                .context("no home directory")?
                .join(".cache/pixworker/models/upscale"),
        );
    let rife_path = env::var("PIXWORKER_VFI_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or(
            dirs::home_dir()
                .context("no home directory")?
                .join(".cache/pixworker/models/vfi/rife-v4.25_fp32.safetensors"),
        );
    let devices: &[Device] = match mode {
        "cpu" => &[Device::Cpu],
        "mps" => &[Device::Mps],
        _ => &[Device::Cpu, Device::Mps],
    };
    for &device in devices {
        ensure!(
            device != Device::Mps || tch::utils::has_mps(),
            "MPS unavailable; use cpu mode to check CPU only"
        );
        println!("device: {device:?}");
        let _guard = tch::no_grad_guard();
        for key in KEYS {
            let (file, wdn) = Upscaler::files(key)?;
            let model = Upscaler::new(
                key,
                &root.join(file),
                wdn.map(|name| root.join(name)).as_deref(),
                1,
                device,
            )?;
            for index in [0, 60, 119] {
                let name = format!("{key}-frame-{index:03}");
                let source = input(dir, index, width, height, device)?;
                if device == Device::Cpu || key == "realesr-animevideov3" {
                    strict(
                        dir,
                        &name,
                        &model.float(source.shallow_clone())?,
                        width * height * 3 * 16,
                    )?;
                }
                if device == Device::Mps {
                    quantized(dir, &name, &model.run(source)?, width * 4, height * 4)?;
                }
            }
        }
        let rife = Rife::new(&rife_path, device)?;
        for (a, b) in PAIRS {
            let first = input(dir, a, width, height, device)?;
            let second = input(dir, b, width, height, device)?;
            let results = rife.interpolate(&first, &second, &TIMES)?;
            for (t, result) in TIMES.iter().zip(results) {
                let name = format!("rife-{a:03}-{b:03}-t{t:.1}");
                if device == Device::Cpu {
                    strict(dir, &name, &result, width * height * 3)?;
                } else {
                    let output = result
                        .f_mul_scalar(255.0)?
                        .f_clamp(0.0, 255.0)?
                        .f_floor()?
                        .f_to_kind(Kind::Uint8)?
                        .f_squeeze_dim(0)?
                        .f_permute([1, 2, 0])?
                        .f_contiguous()?;
                    quantized(dir, &name, &output, width, height)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_metrics() {
        assert!(psnr(0.0, 1.0).is_infinite());
        assert!((psnr(1e-10, 1.0) - 100.0).abs() < 1e-9);
        assert!(psnr(1e-8, 1.0) < 100.0);
        let expected = vec![0u8; 8 * 8 * 3];
        let mut changed = expected.clone();
        changed.fill(255);
        assert!((ssim(&expected, &expected, 8, 8) - 1.0).abs() < 1e-9);
        assert!(ssim(&expected, &changed, 8, 8) < 0.995);
    }
}
