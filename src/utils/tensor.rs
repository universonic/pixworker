use crate::utils::ffmpeg::{VideoDecoder, VideoEncoder, VideoInfo};
use crate::utils::ntsc::NTSC;
use crate::utils::upscale::Upscaler;
use crate::utils::vfi::Rife;
use anyhow::{Context, Result, bail, ensure};
use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, change_type_of_pixel_components,
    images,
};
use ffmpeg_next as ffmpeg;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread;
use std::time::{Duration, Instant};
use tch::{Device, Kind, Tensor};

pub fn enhance(
    input: &PathBuf,
    output: &Option<PathBuf>,
    upscale: &Option<String>,
    upscale_model: &Option<String>,
    vfi: &Option<String>,
    vfi_model: &Option<String>,
    silent: &Option<bool>,
) -> Result<()> {
    EnhanceOptions::try_new(
        input,
        output,
        upscale,
        upscale_model,
        vfi,
        vfi_model,
        silent,
    )?
    .process()
}

pub struct EnhanceOptions {
    input: PathBuf,
    output: PathBuf,
    upscale: Upscale,
    upscale_model: UpscaleModel,
    vfi: VFI,
    silent: bool,
    frame_estimate: Option<u64>,
}

impl EnhanceOptions {
    pub fn try_new(
        input: &PathBuf,
        output: &Option<PathBuf>,
        upscale: &Option<String>,
        upscale_model: &Option<String>,
        vfi: &Option<String>,
        vfi_model: &Option<String>,
        silent: &Option<bool>,
    ) -> Result<Self> {
        ensure!(input.exists(), "Specified input video does not exist.");
        println!("Input video: {}", input.display());
        let info = VideoInfo::open(input)?;
        let output = match output {
            Some(path) => path.clone(),
            None => {
                let stem = input
                    .file_stem()
                    .context("Input has no file name")?
                    .to_string_lossy();
                let ext = input
                    .extension()
                    .context("Input has no extension")?
                    .to_string_lossy();
                let mut path = input.with_file_name(format!("{stem}_enhanced.{ext}"));
                let mut i = 0;
                while path.exists() {
                    path = input.with_file_name(format!("{stem}_enhanced_{i}.{ext}"));
                    i += 1;
                }
                path
            }
        };
        let (width, height) = match upscale {
            None => (info.width as u64 * 2, info.height as u64 * 2),
            Some(value) if value.contains('x') => {
                let (w, h) = value.split_once('x').context("Invalid resolution format")?;
                (
                    w.parse()
                        .with_context(|| format!("Invalid width in resolution format: {w}"))?,
                    h.parse()
                        .with_context(|| format!("Invalid height in resolution format: {h}"))?,
                )
            }
            Some(value) if value.ends_with('p') => match value.as_str() {
                "2160p" => (3840, 2160),
                "1440p" => (2560, 1440),
                "1080p" => (1920, 1080),
                "720p" => (1280, 720),
                "480p" => (720, 480),
                _ => bail!("Unsupported resolution preset: {value}"),
            },
            Some(value) => {
                let factor: f64 = value
                    .parse()
                    .with_context(|| format!("Invalid upscale factor: {value}"))?;
                ensure!(
                    factor.is_finite() && factor > 0.0,
                    "Invalid upscale factor: {value}"
                );
                (
                    (info.width as f64 * factor) as u64,
                    (info.height as f64 * factor) as u64,
                )
            }
        };
        ensure!(
            width > 0 && height > 0,
            "Output dimensions must be positive"
        );
        if upscale
            .as_ref()
            .is_some_and(|s| s.contains('x') || s.ends_with('p'))
        {
            ensure!(
                width / info.width as u64 == height / info.height as u64,
                "Aspect ratio must be maintained when specifying resolution directly."
            );
        }
        let upscale = Upscale {
            old_width: info.width as u64,
            old_height: info.height as u64,
            width,
            height,
        };
        let upscale_model =
            UpscaleModel::parse(upscale_model.as_deref().unwrap_or("realesr-animevideov3"))?;
        let target = match vfi {
            None => info.rate,
            Some(value) if value.to_lowercase().ends_with("fps") => {
                let lower = value.to_lowercase();
                let fps: u64 = lower[..lower.len() - 3]
                    .parse()
                    .with_context(|| format!("Invalid fps format: {value}"))?;
                info.rate
                    .scaled(fps, info.rate.to_fps().round() as u64)
                    .context("Invalid target FPS")?
            }
            Some(value) => {
                let (whole, decimal) = value.split_once('.').unwrap_or((value, ""));
                let den = 10u64
                    .checked_pow(decimal.len().try_into()?)
                    .context("Invalid VFI factor")?;
                let num = whole
                    .parse::<u64>()?
                    .checked_mul(den)
                    .and_then(|n| {
                        if decimal.is_empty() {
                            Some(n)
                        } else {
                            decimal.parse::<u64>().ok().and_then(|d| n.checked_add(d))
                        }
                    })
                    .with_context(|| format!("Invalid VFI factor: {value}"))?;
                info.rate
                    .scaled(num, den)
                    .with_context(|| format!("Invalid VFI factor: {value}"))?
            }
        };
        ensure!(
            target == info.rate
                || target.num as u128 * info.rate.den as u128
                    >= 2 * info.rate.num as u128 * target.den as u128,
            "Target FPS must be at least 2x the original FPS for interpolation"
        );
        match vfi_model.as_deref().unwrap_or("rife-v4.25") {
            "rife-v4.25" => (),
            old if old.starts_with("gimm-") => {
                bail!("Model {old} is no longer supported; use rife-v4.25")
            }
            other => bail!("Unsupported VFI model: {other}"),
        };
        Ok(Self {
            input: input.clone(),
            output,
            upscale,
            upscale_model,
            vfi: VFI {
                old_fps: info.rate,
                fps: target,
            },
            silent: silent.unwrap_or(false),
            frame_estimate: info.frame_estimate,
        })
    }

    pub fn process(&self) -> Result<()> {
        let start = Instant::now();
        let decoder = VideoDecoder::new(&self.input)?;
        let mut encoder = VideoEncoder::new(
            &self.output,
            &self.input,
            self.upscale.width.try_into()?,
            self.upscale.height.try_into()?,
            self.vfi.fps,
            self.silent,
        )?;
        let upscale = self.upscale.width != self.upscale.old_width
            || self.upscale.height != self.upscale.old_height;
        let scale = self.upscale.width as f64 / self.upscale.old_width as f64;
        let passes = if !upscale || scale <= 1.0 {
            0
        } else {
            scale.log(4.0).ceil().max(1.0) as usize
        };
        let intermediate = 4u64
            .checked_pow(passes.try_into()?)
            .and_then(|n| {
                self.upscale
                    .old_width
                    .checked_mul(n)
                    .zip(self.upscale.old_height.checked_mul(n))
            })
            .context("Upscale dimensions exceed supported range")?;
        let resize = intermediate != (self.upscale.width, self.upscale.height);
        let (decoded_tx, decoded_rx) = sync_channel(2);
        let (gpu_tx, gpu_rx) = sync_channel(2);
        let (resize_tx, resize_rx) = sync_channel(2);
        let mut encoded = 0usize;
        let mut encode_busy = Duration::ZERO;
        let estimate = self
            .frame_estimate
            .and_then(|n| usize::try_from(n).ok())
            .and_then(|n| output_frame_count(n, self.vfi.old_fps, self.vfi.fps).ok());
        let (decode, gpu, resized, encoding) = thread::scope(|scope| {
            let decode = scope.spawn(move || decode_frames(decoder, decoded_tx));
            let gpu = scope.spawn(move || self.gpu_frames(decoded_rx, gpu_tx, passes, start));
            let (resize_thread, rx) = if resize {
                let width = self.upscale.width as u32;
                let height = self.upscale.height as u32;
                (
                    Some(scope.spawn(move || resize_frames(gpu_rx, resize_tx, width, height))),
                    resize_rx,
                )
            } else {
                drop(resize_tx);
                (None, gpu_rx)
            };
            let encoding = (|| -> Result<()> {
                for frame in &rx {
                    let busy = Instant::now();
                    encoder.write(&frame)?;
                    encode_busy += busy.elapsed();
                    encoded += 1;
                    if !self.silent && encoded.is_multiple_of(10) {
                        eprintln!(
                            "Encoded {encoded}/{} frames ({:.2} fps)",
                            estimate
                                .map(|n| n.to_string())
                                .unwrap_or_else(|| "?".into()),
                            encoded as f64 / start.elapsed().as_secs_f64()
                        );
                    }
                }
                Ok(())
            })();
            drop(rx);
            (
                decode.join().unwrap(),
                gpu.join().unwrap(),
                resize_thread.map(|worker| worker.join().unwrap()),
                encoding,
            )
        });
        let decode_busy = decode?;
        let gpu = gpu?;
        let resize_busy = resized.transpose()?.unwrap_or(Duration::ZERO);
        encoding?;
        let busy = Instant::now();
        encoder.finish()?;
        encode_busy += busy.elapsed();
        let first_packet = encoder
            .first_video_packet()
            .map(|at| at.duration_since(start));
        if std::env::var("PIXWORKER_TIMING").as_deref() == Ok("1") {
            eprintln!(
                "model load: {:?}, decode busy: {:?}, gpu busy: {:?}, resize busy: {:?}, encode busy: {:?}, RIFE: {:?}, upscale: {:?}, readback: {:?}, first gpu frame: {:?}, first video packet: {:?}, total: {:?}",
                gpu.load,
                decode_busy,
                gpu.busy,
                resize_busy,
                encode_busy,
                gpu.rife,
                gpu.upscale,
                gpu.readback,
                gpu.first_frame,
                first_packet,
                start.elapsed()
            );
        }
        Ok(())
    }

    fn gpu_frames(
        &self,
        rx: Receiver<ffmpeg::frame::Video>,
        tx: SyncSender<ffmpeg::frame::Video>,
        passes: usize,
        start: Instant,
    ) -> Result<GpuTimes> {
        let mut times = GpuTimes::default();
        let load = Instant::now();
        let device = Upscaler::default_device();
        let _guard = tch::no_grad_guard();
        let upscale = if passes > 0 {
            let (file, wdn) = Upscaler::files(self.upscale_model.key())?;
            let files = self.models(
                "upscale",
                &[file].into_iter().chain(wdn).collect::<Vec<_>>(),
                "Real-ESRGAN",
                "BSD 3-Clause",
                "https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE",
                "RealESRGAN",
            )?;
            Some(Upscaler::new(
                self.upscale_model.key(),
                &files[0],
                files.get(1).map(PathBuf::as_path),
                passes,
                device,
            )?)
        } else {
            None
        };
        let rife = if self.vfi.fps != self.vfi.old_fps {
            let path = self.models(
                "vfi",
                &["rife-v4.25_fp32.safetensors"],
                "Practical-RIFE v4.25",
                "MIT",
                "https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE",
                "RIFE",
            )?;
            Some(Rife::new(&path[0], device)?)
        } else {
            None
        };
        times.load = load.elapsed();
        let mut previous: Option<(ffmpeg::frame::Video, Option<Tensor>)> = None;
        let mut count = 0usize;
        let mut next = 0usize;
        for frame in rx {
            times.first_frame.get_or_insert_with(|| start.elapsed());
            let busy = Instant::now();
            let mut accounted = Duration::ZERO;
            let tensor = if upscale.is_some() || rife.is_some() {
                Some(upload(&frame, device)?)
            } else {
                None
            };
            count += 1;
            if let Some((old, old_tensor)) = previous.take() {
                let mut old = Some(old);
                let ready = ready_frames(&mut next, count, false, self.vfi.old_fps, self.vfi.fps)?;
                let ts: Vec<_> = ready
                    .iter()
                    .filter_map(|&(_, _, t)| (t > 0.0).then_some(t))
                    .collect();
                let mut interpolated = if ts.is_empty() {
                    Vec::new()
                } else {
                    let run = Instant::now();
                    let result = rife.as_ref().unwrap().interpolate(
                        old_tensor.as_ref().unwrap(),
                        tensor.as_ref().unwrap(),
                        &ts,
                    )?;
                    times.rife += run.elapsed();
                    result
                }
                .into_iter();
                for (_, source, t) in ready {
                    debug_assert_eq!(source, count - 2);
                    let result = if t == 0.0 {
                        if let Some(model) = &upscale {
                            let run = Instant::now();
                            let output = model.run(old_tensor.as_ref().unwrap().shallow_clone())?;
                            times.upscale += run.elapsed();
                            let read = Instant::now();
                            let result = readback(&output)?;
                            times.readback += read.elapsed();
                            result
                        } else {
                            old.take().context("Duplicate source frame timestamp")?
                        }
                    } else {
                        let image = interpolated.next().context("Missing RIFE output")?;
                        let quantized = image
                            .f_mul_scalar(255.0)?
                            .f_clamp(0., 255.)?
                            .f_floor()?
                            .f_to_kind(Kind::Uint8)?;
                        let output = if let Some(model) = &upscale {
                            let run = Instant::now();
                            let output = model
                                .run(quantized.f_to_kind(Kind::Float)?.f_div_scalar(255.0)?)?;
                            times.upscale += run.elapsed();
                            output
                        } else {
                            quantized
                                .f_squeeze_dim(0)?
                                .f_permute([1, 2, 0])?
                                .f_contiguous()?
                        };
                        let read = Instant::now();
                        let result = readback(&output)?;
                        times.readback += read.elapsed();
                        result
                    };
                    times.busy += busy.elapsed().saturating_sub(accounted);
                    if tx.send(result).is_err() {
                        return Ok(times);
                    }
                    accounted = busy.elapsed();
                }
            }
            previous = Some((frame, tensor));
            times.busy += busy.elapsed().saturating_sub(accounted);
        }
        ensure!(count > 0, "No video frames were decoded");
        if rife.is_some() {
            ensure!(
                count >= 2,
                "Need at least 2 frames for interpolation, but only found {count} frames"
            );
        }
        let (last, last_tensor) = previous.unwrap();
        let busy = Instant::now();
        let mut accounted = Duration::ZERO;
        let remaining = ready_frames(&mut next, count, true, self.vfi.old_fps, self.vfi.fps)?;
        if !remaining.is_empty() {
            let tail = if let Some(model) = &upscale {
                let run = Instant::now();
                let output = model.run(last_tensor.as_ref().unwrap().shallow_clone())?;
                times.upscale += run.elapsed();
                let read = Instant::now();
                let frame = readback(&output)?;
                times.readback += read.elapsed();
                frame
            } else {
                last
            };
            let mut tail = Some(tail);
            for index in 0..remaining.len() {
                let result = if index + 1 == remaining.len() {
                    tail.take().unwrap()
                } else {
                    tail.as_ref().unwrap().clone()
                };
                times.busy += busy.elapsed().saturating_sub(accounted);
                if tx.send(result).is_err() {
                    return Ok(times);
                }
                accounted = busy.elapsed();
            }
        }
        Ok(times)
    }

    fn models(
        &self,
        category: &str,
        names: &[&str],
        title: &str,
        license: &str,
        link: &str,
        repo: &str,
    ) -> Result<Vec<PathBuf>> {
        let root = dirs::home_dir()
            .context("Cannot locate home directory for model cache")?
            .join(".cache/pixworker/models")
            .join(category);
        let paths: Vec<_> = names.iter().map(|name| root.join(name)).collect();
        if paths.iter().any(|path| !path.exists()) {
            println!(
                "\n=== {title} Model License Agreement ===\nLicensed under {license}.\nLicense: {link}\nBy downloading this model, you agree to its license.\n"
            );
            if category == "upscale" {
                println!("You may use, modify, and distribute the model, including commercially.");
                println!(
                    "You must include the copyright notice and license text, and may not use the authors' names for endorsement."
                );
            } else {
                println!(
                    "You may use, modify, and distribute the model, including commercially, provided the copyright and permission notice is included."
                );
            }
            print!("Do you agree to the license terms? (y/N): ");
            io::stdout().flush()?;
            let mut answer = String::new();
            io::stdin().read_line(&mut answer)?;
            ensure!(
                matches!(answer.trim().to_lowercase().as_str(), "y" | "yes"),
                "Model download cancelled. You must agree to the license to use {title} models."
            );
            fs::create_dir_all(&root)?;
            for (name, path) in names.iter().zip(&paths) {
                if path.exists() {
                    continue;
                }
                let url = format!("https://huggingface.co/universonic/{repo}/resolve/main/{name}");
                println!("Downloading model from {url}...");
                let mut response = reqwest::blocking::get(&url)?.error_for_status()?;
                let mut temp = tempfile::NamedTempFile::new_in(&root)?;
                io::copy(&mut response, &mut temp)?;
                temp.persist(path)?;
            }
        }
        Ok(paths)
    }
}

#[derive(Default)]
struct GpuTimes {
    load: Duration,
    busy: Duration,
    rife: Duration,
    upscale: Duration,
    readback: Duration,
    first_frame: Option<Duration>,
}

fn decode_frames(
    mut decoder: VideoDecoder,
    tx: SyncSender<ffmpeg::frame::Video>,
) -> Result<Duration> {
    let mut busy = Duration::ZERO;
    loop {
        let start = Instant::now();
        let next = decoder.next_frame()?;
        busy += start.elapsed();
        let Some(frame) = next else { break };
        if tx.send(frame).is_err() {
            break;
        }
    }
    Ok(busy)
}

fn upload(frame: &ffmpeg::frame::Video, device: Device) -> Result<Tensor> {
    ensure!(
        frame.format() == ffmpeg::format::Pixel::RGB24,
        "Expected RGB24 decoded frame"
    );
    let (width, height, stride) = (
        frame.width() as usize,
        frame.height() as usize,
        frame.stride(0),
    );
    let row = width.checked_mul(3).context("Frame width overflow")?;
    ensure!(
        stride >= row && frame.data(0).len() >= (height - 1) * stride + row,
        "Invalid RGB frame stride"
    );
    // from_blob borrows AVFrame storage. Float conversion owns its bytes even on CPU.
    let bytes = unsafe {
        Tensor::f_from_blob(
            frame.data(0).as_ptr(),
            &[height as i64, width as i64, 3],
            &[stride as i64, 3, 1],
            Kind::Uint8,
            Device::Cpu,
        )?
    };
    Ok(bytes
        .f_to_device(device)?
        .f_to_kind(Kind::Float)?
        .f_div_scalar(255.0)?
        .f_permute([2, 0, 1])?
        .f_unsqueeze(0)?)
}

fn readback(tensor: &Tensor) -> Result<ffmpeg::frame::Video> {
    let shape = tensor.size();
    ensure!(
        shape.len() == 3 && shape[2] == 3 && tensor.kind() == Kind::Uint8,
        "Expected HWC RGB u8 tensor"
    );
    let (height, width) = (u32::try_from(shape[0])?, u32::try_from(shape[1])?);
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, width, height);
    let cpu = tensor.f_to_device(Device::Cpu)?.f_contiguous()?;
    let mut bytes = vec![0; cpu.numel()];
    cpu.f_copy_data(&mut bytes, cpu.numel())?;
    let row = width as usize * 3;
    let stride = frame.stride(0);
    for (src, dst) in bytes
        .chunks_exact(row)
        .zip(frame.data_mut(0).chunks_mut(stride))
    {
        dst[..row].copy_from_slice(src);
    }
    Ok(frame)
}

fn resize_frames(
    rx: Receiver<ffmpeg::frame::Video>,
    tx: SyncSender<ffmpeg::frame::Video>,
    width: u32,
    height: u32,
) -> Result<Duration> {
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    let mut resizer = Resizer::new();
    let mut busy = Duration::ZERO;
    for frame in rx {
        let start = Instant::now();
        let row = frame.width() as usize * 3;
        let source: Vec<u8> = frame
            .data(0)
            .chunks(frame.stride(0))
            .take(frame.height() as usize)
            .flat_map(|line| line[..row].iter().copied())
            .collect();
        let src = images::ImageRef::new(frame.width(), frame.height(), &source, PixelType::U8x3)?;
        // Keep the Lanczos intermediate in float, as in the old image resize.
        let mut float_src = images::Image::new(frame.width(), frame.height(), PixelType::F32x3);
        change_type_of_pixel_components(&src, &mut float_src)?;
        let mut float_dst = images::Image::new(width, height, PixelType::F32x3);
        resizer.resize(&float_src, &mut float_dst, &options)?;
        let mut dst = images::Image::new(width, height, PixelType::U8x3);
        change_type_of_pixel_components(&float_dst, &mut dst)?;
        let mut output = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, width, height);
        let stride = output.stride(0);
        for (src, dst) in dst
            .buffer()
            .chunks_exact(width as usize * 3)
            .zip(output.data_mut(0).chunks_mut(stride))
        {
            dst[..width as usize * 3].copy_from_slice(src);
        }
        busy += start.elapsed();
        if tx.send(output).is_err() {
            break;
        }
    }
    Ok(busy)
}

pub struct Upscale {
    pub old_width: u64,
    pub old_height: u64,
    pub width: u64,
    pub height: u64,
}

pub enum UpscaleModel {
    RealESRAnimeVideoV3,
    RealESRAnimeVideoV3Hf,
    RealESRGeneralx4v3,
    RealESRGeneralx4v3Hf,
    RealESRGANx4Plus,
    RealESRGANx4PlusHf,
    RealESRGANx4PlusAnime,
    RealESRGANx4PlusAnimeHf,
}

impl UpscaleModel {
    fn parse(key: &str) -> Result<Self> {
        Ok(match key {
            "realesr-animevideov3" => Self::RealESRAnimeVideoV3,
            "realesr-animevideov3-hf" => Self::RealESRAnimeVideoV3Hf,
            "realesr-generalx4v3" => Self::RealESRGeneralx4v3,
            "realesr-generalx4v3-hf" => Self::RealESRGeneralx4v3Hf,
            "realesrganx4plus" | "realesrgan-x4plus" => Self::RealESRGANx4Plus,
            "realesrganx4plus-hf" | "realesrgan-x4plus-hf" => Self::RealESRGANx4PlusHf,
            "realesrganx4plus-anime" | "realesrgan-x4plus-anime" => Self::RealESRGANx4PlusAnime,
            "realesrganx4plus-anime-hf" | "realesrgan-x4plus-anime-hf" => {
                Self::RealESRGANx4PlusAnimeHf
            }
            _ => bail!("Unsupported upscale model: {key}"),
        })
    }

    fn key(&self) -> &'static str {
        match self {
            Self::RealESRAnimeVideoV3 => "realesr-animevideov3",
            Self::RealESRAnimeVideoV3Hf => "realesr-animevideov3-hf",
            Self::RealESRGeneralx4v3 => "realesr-generalx4v3",
            Self::RealESRGeneralx4v3Hf => "realesr-generalx4v3-hf",
            Self::RealESRGANx4Plus => "realesrgan-x4plus",
            Self::RealESRGANx4PlusHf => "realesrgan-x4plus-hf",
            Self::RealESRGANx4PlusAnime => "realesrgan-x4plus-anime",
            Self::RealESRGANx4PlusAnimeHf => "realesrgan-x4plus-anime-hf",
        }
    }
}

pub struct VFI {
    pub old_fps: NTSC,
    pub fps: NTSC,
}

fn output_frame_count(source_count: usize, source: NTSC, target: NTSC) -> Result<usize> {
    let num = (source_count as u128)
        .checked_mul(target.num as u128)
        .and_then(|n| n.checked_mul(source.den as u128))
        .context("Output frame count exceeds supported range")?;
    let den = target.den as u128 * source.num as u128;
    Ok(num.div_ceil(den).try_into()?)
}

fn frame_position(output_idx: usize, source: NTSC, target: NTSC) -> Result<(usize, f32)> {
    let num = (output_idx as u128)
        .checked_mul(target.den as u128)
        .and_then(|n| n.checked_mul(source.num as u128))
        .context("Frame timestamp exceeds supported range")?;
    let den = target.num as u128 * source.den as u128;
    Ok(((num / den).try_into()?, (num % den) as f32 / den as f32))
}

// A pair becomes available only when its following source frame has arrived.
fn ready_frames(
    next: &mut usize,
    received: usize,
    eof: bool,
    source: NTSC,
    target: NTSC,
) -> Result<Vec<(usize, usize, f32)>> {
    let limit = if eof {
        output_frame_count(received, source, target)?
    } else {
        usize::MAX
    };
    let mut ready = Vec::new();
    while *next < limit {
        let (index, t) = frame_position(*next, source, target)?;
        if !eof && index >= received - 1 {
            break;
        }
        ready.push((
            *next,
            index.min(received - 1),
            if eof && index >= received - 1 { 0.0 } else { t },
        ));
        *next = next.checked_add(1).context("Output frame count overflow")?;
    }
    Ok(ready)
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    #[test]
    fn preserves_rational_timestamps_and_frame_boundaries() {
        for source in [NTSC::new(&24, &1), NTSC::new(&24000, &1001)] {
            let target = source.scaled(5, 2).unwrap();
            assert_eq!(output_frame_count(4, source, target).unwrap(), 10);
            let positions: Vec<_> = (0..10)
                .map(|j| frame_position(j, source, target).unwrap())
                .collect();
            assert_eq!(
                positions.iter().map(|p| p.0).collect::<Vec<_>>(),
                [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
            );
            assert_eq!(positions[0], (0, 0.0));
            assert_eq!(positions[5], (2, 0.0));
            assert_eq!(
                target,
                if source.den == 1 {
                    NTSC::new(&60, &1)
                } else {
                    NTSC::new(&60000, &1001)
                }
            );
            let doubled = source.scaled(2, 1).unwrap();
            assert_eq!(output_frame_count(4, source, doubled).unwrap(), 8);
            assert_eq!(frame_position(7, source, doubled).unwrap(), (3, 0.5));
            assert_eq!(output_frame_count(4, source, source).unwrap(), 4);
        }
        assert_eq!(NTSC::from_strict_fps(&30), NTSC::new(&30, &1));
        assert!(
            output_frame_count(usize::MAX, NTSC::new(&1, &1), NTSC::new(&u64::MAX, &1)).is_err()
        );
    }

    #[test]
    fn streaming_matches_batch_at_boundaries() {
        for source in [NTSC::new(&24, &1), NTSC::new(&24000, &1001)] {
            for target in [
                source,
                source.scaled(2, 1).unwrap(),
                source.scaled(5, 2).unwrap(),
            ] {
                let mut next = 0;
                let mut emitted = Vec::new();
                for n in 1..=4 {
                    emitted.extend(ready_frames(&mut next, n, false, source, target).unwrap());
                }
                emitted.extend(ready_frames(&mut next, 4, true, source, target).unwrap());
                let expected: Vec<_> = (0..output_frame_count(4, source, target).unwrap())
                    .map(|j| {
                        let (i, t) = frame_position(j, source, target).unwrap();
                        (j, i.min(3), if i >= 3 { 0.0 } else { t })
                    })
                    .collect();
                assert_eq!(emitted, expected);
            }
        }
    }

    #[test]
    fn rgb_stride_round_trip() -> Result<()> {
        let mut input = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, 3, 2);
        let stride = input.stride(0);
        assert!(stride >= 9);
        for (y, row) in input.data_mut(0).chunks_mut(stride).take(2).enumerate() {
            row[..9].copy_from_slice(&[y as u8 + 1; 9]);
        }
        let tensor = upload(&input, Device::Cpu)?;
        let bytes = tensor
            .f_mul_scalar(255.0)?
            .f_round()?
            .f_to_kind(Kind::Uint8)?
            .f_squeeze_dim(0)?
            .f_permute([1, 2, 0])?
            .f_contiguous()?;
        let output = readback(&bytes)?;
        for (y, row) in output.data(0).chunks(output.stride(0)).take(2).enumerate() {
            assert_eq!(&row[..9], &[y as u8 + 1; 9]);
        }
        Ok(())
    }

    #[test]
    fn one_frame_without_interpolation_passes_through() -> Result<()> {
        let rate = NTSC::new(&24, &1);
        let options = EnhanceOptions {
            input: PathBuf::new(),
            output: PathBuf::new(),
            upscale: Upscale {
                old_width: 2,
                old_height: 2,
                width: 2,
                height: 2,
            },
            upscale_model: UpscaleModel::RealESRAnimeVideoV3,
            vfi: VFI {
                old_fps: rate,
                fps: rate,
            },
            silent: true,
            frame_estimate: None,
        };
        let (input, rx) = sync_channel(2);
        input.send(ffmpeg::frame::Video::new(
            ffmpeg::format::Pixel::RGB24,
            2,
            2,
        ))?;
        drop(input);
        let (output, frames) = sync_channel(2);
        options.gpu_frames(rx, output, 0, Instant::now())?;
        assert_eq!(frames.into_iter().count(), 1);
        Ok(())
    }
}
