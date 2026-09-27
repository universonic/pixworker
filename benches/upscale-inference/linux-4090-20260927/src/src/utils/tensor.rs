use crate::utils::ffmpeg::ArchiveOptions;
use crate::utils::ffmpeg::{ExtractOptions, FFProbe};
use crate::utils::ntsc::NTSC;
use anyhow::{Result, bail};
use half::f16;
use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array, Axis, Ix3, Ix4, s, stack};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use ort::ep::CoreML as CoreMLExecutionProvider;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(
        any(target_os = "linux", target_os = "windows"),
        target_arch = "x86_64"
    )
))]
use ort::ep::ExecutionProvider;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use ort::ep::coreml::ComputeUnits;
#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    target_arch = "x86_64"
))]
use ort::ep::{CUDA as CUDAExecutionProvider, TensorRT as TensorRTExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::{
    session::Session,
    value::{Tensor, TensorRef, Value},
};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

pub fn enhance(
    input: &PathBuf,
    output: &Option<PathBuf>,
    upscale: &Option<String>,
    upscale_model: &Option<String>,
    vfi: &Option<String>,
    vfi_model: &Option<String>,
    silent: &Option<bool>,
) -> Result<()> {
    let options = EnhanceOptions::try_new(
        input,
        output,
        upscale,
        upscale_model,
        vfi,
        vfi_model,
        silent,
    )?;
    options.process()?;
    Ok(())
}

pub struct EnhanceOptions {
    input: PathBuf,
    output: PathBuf,
    upscale: Upscale,
    upscale_model: UpscaleModel,
    vfi: VFI,
    vfi_model: VFIModel,
    silent: bool,
}

impl EnhanceOptions {
    pub fn new(
        input: PathBuf,
        output: PathBuf,
        upscale: Upscale,
        upscale_model: UpscaleModel,
        vfi: VFI,
        vfi_model: VFIModel,
        silent: bool,
    ) -> Self {
        Self {
            input,
            output,
            upscale,
            upscale_model,
            vfi,
            vfi_model,
            silent,
        }
    }

    pub fn try_new(
        input: &PathBuf,
        output: &Option<PathBuf>,
        upscale: &Option<String>,
        upscale_model: &Option<String>,
        vfi: &Option<String>,
        vfi_model: &Option<String>,
        silent: &Option<bool>,
    ) -> Result<Self> {
        let input = input.as_path();
        if !input.exists() {
            bail!("Specified input video does not exist.");
        }
        println!("Input video: {}", input.display());

        let info = FFProbe::new(&input.to_path_buf()).inspect_video()?;
        if info.width.is_none() || info.height.is_none() || info.r_frame_rate.is_none() {
            bail!("Failed to retrieve video information from input file.");
        }

        let output = match output {
            Some(path) => path.clone(),
            None => {
                // By default, it appends "_enhanced" to the file name.
                // If a file with that name exists, it will attempt versions suffixed with "_enhanced_0", "_enhanced_1", etc.
                let mut output = input.with_file_name(format!(
                    "{}_enhanced.{}",
                    input.file_stem().unwrap().display(),
                    input.extension().unwrap().display()
                ));
                let mut i = 0;
                while output.exists() {
                    let new_file_name = format!(
                        "{}_enhanced_{}.{}",
                        input.file_stem().unwrap().display(),
                        i,
                        input.extension().unwrap().display()
                    );
                    output = input.with_file_name(new_file_name);
                    i += 1;
                }
                output
            }
        };

        let upscale = match upscale {
            Some(upscale_str) => {
                let lowercase = upscale_str.to_lowercase();
                // Check if it's in "WIDTHxHEIGHT" format.
                if upscale_str.contains("x") {
                    let values = lowercase.split("x").collect::<Vec<&str>>();
                    if values.len() != 2 {
                        bail!("Invalid resolution format: {}", upscale_str);
                    }

                    let width = values[0].parse::<u64>().map_err(|_| {
                        anyhow::anyhow!("Invalid width in resolution format: {}", values[0])
                    })?;
                    let height = values[1].parse::<u64>().map_err(|_| {
                        anyhow::anyhow!("Invalid height in resolution format: {}", values[1])
                    })?;

                    if width / info.width.unwrap() != height / info.height.unwrap() {
                        bail!(
                            "Aspect ratio must be maintained when specifying resolution directly."
                        );
                    }

                    Upscale {
                        old_width: info.width.unwrap(),
                        old_height: info.height.unwrap(),
                        width,
                        height,
                    }
                // Check if it's a preset like "1080p".
                } else if upscale_str.ends_with("p") {
                    let (width, height) = match upscale_str.as_str() {
                        "2160p" => (3840, 2160),
                        "1440p" => (2560, 1440),
                        "1080p" => (1920, 1080),
                        "720p" => (1280, 720),
                        "480p" => (720, 480),
                        _ => {
                            bail!("Unsupported resolution preset: {}", upscale_str);
                        }
                    };

                    if width / info.width.unwrap() != height / info.height.unwrap() {
                        bail!(
                            "Aspect ratio must be maintained when specifying resolution directly."
                        );
                    }

                    Upscale {
                        old_width: info.width.unwrap(),
                        old_height: info.height.unwrap(),
                        width,
                        height,
                    }
                // Otherwise, treat it as a scaling factor.
                } else {
                    Upscale {
                        old_width: info.width.unwrap(),
                        old_height: info.height.unwrap(),
                        width: (info.width.unwrap() as f64
                            * upscale_str.parse::<f64>().map_err(|_| {
                                anyhow::anyhow!("Invalid upscale factor: {}", upscale_str)
                            })?) as u64,
                        height: (info.height.unwrap() as f64
                            * upscale_str.parse::<f64>().map_err(|_| {
                                anyhow::anyhow!("Invalid upscale factor: {}", upscale_str)
                            })?) as u64,
                    }
                }
            }
            None => Upscale {
                old_width: info.width.unwrap(),
                old_height: info.height.unwrap(),
                width: info.width.unwrap() * 2,   // Default width
                height: info.height.unwrap() * 2, // Default height
            },
        };

        let upscale_model = match upscale_model {
            Some(model_name) => {
                let model: UpscaleModel = match model_name.as_str() {
                    "realesr-animevideov3" => UpscaleModel::RealESRAnimeVideoV3,
                    "realesr-animevideov3-hf" => UpscaleModel::RealESRAnimeVideoV3Hf,
                    "realesr-generalx4v3" => UpscaleModel::RealESRGeneralx4v3,
                    "realesr-generalx4v3-hf" => UpscaleModel::RealESRGeneralx4v3Hf,
                    "realesrganx4plus" => UpscaleModel::RealESRGANx4Plus,
                    "realesrganx4plus-hf" => UpscaleModel::RealESRGANx4PlusHf,
                    "realesrganx4plus-anime" => UpscaleModel::RealESRGANx4PlusAnime,
                    "realesrganx4plus-anime-hf" => UpscaleModel::RealESRGANx4PlusAnimeHf,
                    _ => {
                        bail!("Unsupported upscale model: {}", model_name);
                    }
                };
                model
            }
            None => UpscaleModel::RealESRAnimeVideoV3,
        };

        let old_fps = info.r_frame_rate.unwrap();
        let vfi = match vfi {
            Some(vfi_str) => {
                // Check if it's in "XXfps" format.
                let vfi_str = vfi_str.to_lowercase();
                if vfi_str.ends_with("fps") {
                    let fps_value = &vfi_str[..vfi_str.len() - 3];
                    let fps = fps_value
                        .parse::<u64>()
                        .map_err(|_| anyhow::anyhow!("Invalid fps format: {}", vfi_str))?;

                    let nominal = old_fps.to_fps().round() as u64;
                    let target = old_fps
                        .scaled(fps, nominal)
                        .ok_or_else(|| anyhow::anyhow!("Invalid target FPS"))?;
                    VFI {
                        old_fps,
                        fps: target,
                    }
                // Otherwise, treat it as a scaling factor.
                } else {
                    let (whole, decimal) = vfi_str.split_once('.').unwrap_or((&vfi_str, ""));
                    let denominator = 10u64
                        .checked_pow(decimal.len().try_into()?)
                        .ok_or_else(|| anyhow::anyhow!("Invalid VFI factor: {}", vfi_str))?;
                    let numerator = whole
                        .parse::<u64>()?
                        .checked_mul(denominator)
                        .and_then(|v| {
                            if decimal.is_empty() {
                                Some(v)
                            } else {
                                decimal
                                    .parse::<u64>()
                                    .ok()
                                    .and_then(|fraction| v.checked_add(fraction))
                            }
                        })
                        .ok_or_else(|| anyhow::anyhow!("Invalid VFI factor: {}", vfi_str))?;
                    let target = old_fps
                        .scaled(numerator, denominator)
                        .ok_or_else(|| anyhow::anyhow!("Invalid VFI factor: {}", vfi_str))?;
                    VFI {
                        old_fps,
                        fps: target,
                    }
                }
            }
            None => VFI {
                old_fps,
                fps: old_fps,
            },
        };
        if vfi.fps != vfi.old_fps
            && (vfi.fps.num as u128 * (vfi.old_fps.den as u128)
                < 2 * vfi.old_fps.num as u128 * vfi.fps.den as u128)
        {
            bail!("Target FPS must be at least 2x the original FPS for interpolation");
        }

        let vfi_model = match vfi_model {
            Some(model_name) => {
                let model: VFIModel = match model_name.as_str() {
                    "gimm-vfi-f-p" => VFIModel::GimmVfiFP,
                    "gimm-vfi-f-p-hf" => VFIModel::GimmVfiFPHf,
                    "gimm-vfi-r-p" => VFIModel::GimmVfiRP,
                    "gimm-vfi-r-p-hf" => VFIModel::GimmVfiRPHf,
                    _ => {
                        bail!("Unsupported VFI model: {}", model_name);
                    }
                };
                model
            }
            None => VFIModel::GimmVfiFP,
        };

        let silent = silent.unwrap_or(false);

        Ok(Self::new(
            input.to_path_buf(),
            output,
            upscale,
            upscale_model,
            vfi,
            vfi_model,
            silent,
        ))
    }

    pub fn process(&self) -> Result<()> {
        if !self.silent {
            tracing_subscriber::fmt::init();
        }

        let tempdir = TempDir::new()?;
        let tempdir_path = tempdir.path().to_path_buf();

        let tempdir_extracted = tempdir_path.join("extracted");
        let tempdir_orig_frames = tempdir_path.join("frames_orig");
        let tempdir_frames = tempdir_path.join("frames");
        let tempdir_vfi = tempdir_path.join("vfi");

        let stage_start = Instant::now();
        let extract = ExtractOptions::new(
            self.input.to_path_buf(),
            tempdir_extracted.clone(),
            0,
            0,
            "pcm_s16le".to_string(),
            self.silent,
        );
        extract.process()?;

        // Rename extracted frames directory to avoid conflicts
        fs::rename(tempdir_extracted.join("frames"), &tempdir_orig_frames)?;
        fs::rename(tempdir_extracted.join("audio"), tempdir_path.join("audio"))?;
        if std::env::var_os("PIXWORKER_TIMING").is_some() {
            eprintln!("extract: {:?}", stage_start.elapsed());
        }

        // Process VFI and upscaling
        let stage_start = Instant::now();
        if self.vfi.fps == self.vfi.old_fps {
            fs::rename(&tempdir_orig_frames, &tempdir_vfi)?;
        } else {
            self.process_vfi(&tempdir_orig_frames, &tempdir_vfi)?;
        }
        if std::env::var_os("PIXWORKER_TIMING").is_some() {
            eprintln!("vfi: {:?}", stage_start.elapsed());
        }
        let stage_start = Instant::now();
        if self.upscale.width == self.upscale.old_width
            && self.upscale.height == self.upscale.old_height
        {
            fs::rename(&tempdir_vfi, &tempdir_frames)?;
        } else {
            self.process_upscale(&tempdir_vfi, &tempdir_frames)?;
        }
        if std::env::var_os("PIXWORKER_TIMING").is_some() {
            eprintln!("upscale: {:?}", stage_start.elapsed());
        }

        // Encode all frames (original + interpolated) to output video
        let stage_start = Instant::now();
        let archive = ArchiveOptions::new(
            tempdir_path.to_path_buf(),
            self.output.to_path_buf(),
            self.vfi.fps,
            1.0,
            "pcm_s16le".to_string(),
            self.silent,
        );
        archive.process()?;
        if std::env::var_os("PIXWORKER_TIMING").is_some() {
            eprintln!("archive: {:?}", stage_start.elapsed());
        }
        Ok(())
    }

    fn process_vfi(&self, input_dir: &Path, output_dir: &Path) -> Result<()> {
        // Determine model path based on VFI model type
        let model_filename = match self.vfi_model {
            VFIModel::GimmVfiFP => "gimmvfi_f_arb_lpips_fp32.onnx",
            VFIModel::GimmVfiFPHf => "gimmvfi_f_arb_lpips_fp16.onnx",
            VFIModel::GimmVfiRP => "gimmvfi_r_arb_lpips_fp32.onnx",
            VFIModel::GimmVfiRPHf => "gimmvfi_r_arb_lpips_fp16.onnx",
        };

        // Try to find model in GIMM-VFI workspace or default model directory
        let model_path = self.find_vfi_model(model_filename)?;

        if !self.silent {
            println!("Loading VFI model: {}", model_path.display());
        }

        let session_start = Instant::now();
        let mut _session = new_session(&model_path, self.silent)?;
        if std::env::var_os("PIXWORKER_TIMING").is_some() {
            eprintln!("vfi session: {:?}", session_start.elapsed());
        }
        let session = &mut _session;

        if !self.silent {
            println!("VFI model loaded successfully");
            println!(
                "Model type: {}",
                match self.vfi_model {
                    VFIModel::GimmVfiFP => "GIMMVFI_F (FlowFormer-based)",
                    VFIModel::GimmVfiFPHf => "GIMMVFI_F (FlowFormer-based, Half-Precision)",
                    VFIModel::GimmVfiRP => "GIMMVFI_R (RAFT-based)",
                    VFIModel::GimmVfiRPHf => "GIMMVFI_R (RAFT-based, Half-Precision)",
                }
            );
            println!(
                "Processing video interpolation from {}fps to {}fps",
                self.vfi.old_fps.to_fps(),
                self.vfi.fps.to_fps()
            );
        }

        // Frame files are extracted to tempdir_path/frames directory
        if !input_dir.exists() {
            bail!("Frames directory does not exist. Frame extraction may have failed.");
        }

        // Collect and sort frame file paths (only paths, not loading images yet)
        let mut frame_files: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(&input_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "png" {
                        frame_files.push(path);
                    }
                }
            }
        }

        frame_files.sort_by(|a, b| {
            let x = a
                .file_stem()
                .unwrap()
                .to_str()
                .and_then(|num_str| num_str.parse::<f64>().ok())
                .unwrap_or(0.0);
            let y = b
                .file_stem()
                .unwrap()
                .to_str()
                .and_then(|num_str| num_str.parse::<f64>().ok())
                .unwrap_or(0.0);
            x.partial_cmp(&y).unwrap()
        });

        if !self.silent {
            println!(
                "Processing {} frames with {}x interpolation...",
                frame_files.len(),
                self.vfi.fps.to_fps() / self.vfi.old_fps.to_fps()
            );
        }

        // Check if we have enough frames to interpolate
        if frame_files.len() < 2 {
            bail!(
                "Need at least 2 frames for interpolation, but only found {} frames",
                frame_files.len()
            );
        }

        // Create output directory for interpolated frames
        let output_path = output_dir.to_path_buf();
        fs::create_dir_all(&output_path)?;

        // Process frame interpolation in streaming fashion
        let mut output_frame_idx = 0;
        let output_count = output_frame_count(frame_files.len(), self.vfi.old_fps, self.vfi.fps)?;
        let mut io_elapsed = Duration::ZERO;
        let io_start = Instant::now();
        let mut frame_start = load_frame(&frame_files[0])?;
        io_elapsed += io_start.elapsed();

        for i in 0..frame_files.len() - 1 {
            if !self.silent && i % 10 == 0 {
                println!("Processing frame pair {}/{}", i + 1, frame_files.len() - 1);
            }

            let mut times = Vec::new();
            while output_frame_idx < output_count {
                let (source_idx, t) =
                    frame_position(output_frame_idx, self.vfi.old_fps, self.vfi.fps)?;
                if source_idx != i {
                    break;
                }
                if t == 0.0 {
                    let io_start = Instant::now();
                    fs::copy(
                        &frame_files[i],
                        output_path.join(format!("{}.png", output_frame_idx)),
                    )?;
                    io_elapsed += io_start.elapsed();
                } else {
                    times.push((output_frame_idx, t));
                }
                output_frame_idx += 1;
            }
            let io_start = Instant::now();
            let frame_end = load_frame(&frame_files[i + 1])?;
            io_elapsed += io_start.elapsed();
            if !times.is_empty() {
                let interpolated_frames = self.interpolate_frames(
                    session,
                    &frame_start,
                    &frame_end,
                    &times.iter().map(|(_, t)| *t).collect::<Vec<_>>(),
                )?;
                for ((index, _), interp_frame) in times.into_iter().zip(interpolated_frames) {
                    let io_start = Instant::now();
                    save_frame(&interp_frame, &output_path, index)?;
                    io_elapsed += io_start.elapsed();
                }
            }
            frame_start = frame_end;
        }

        while output_frame_idx < output_count {
            let io_start = Instant::now();
            fs::copy(
                frame_files.last().unwrap(),
                output_path.join(format!("{}.png", output_frame_idx)),
            )?;
            io_elapsed += io_start.elapsed();
            output_frame_idx += 1;
        }
        if std::env::var_os("PIXWORKER_TIMING").is_some() {
            eprintln!("vfi PNG I/O: {io_elapsed:?}");
        }

        if !self.silent {
            println!(
                "Interpolation complete! Generated {} frames total.",
                output_frame_idx
            );
        }
        Ok(())
    }

    fn process_upscale(&self, input_dir: &Path, output_dir: &Path) -> Result<()> {
        if !input_dir.exists() {
            bail!("Upscale input directory does not exist");
        }

        fs::create_dir_all(output_dir)?;

        // All Real-ESRGAN models are 4x upscale models
        const MODEL_SCALE: f64 = 4.0;

        // Calculate upscale factor needed
        let target_scale = self.upscale.width as f64 / self.upscale.old_width as f64;

        // Determine how many times we need to apply 4x upscaling
        // For target_scale <= 4: apply once, then downscale if needed
        // For 4 < target_scale <= 16: apply twice (4x then 4x = 16x), then downscale
        // For 16 < target_scale: apply log4(target) times
        let num_upscale_passes = if target_scale <= 1.0 {
            // If target is smaller than input, just resize (no upscaling needed)
            0
        } else if target_scale <= MODEL_SCALE {
            // Single pass is sufficient
            1
        } else {
            // Multiple passes needed: calculate how many 4x passes to exceed target
            (target_scale.log(MODEL_SCALE).ceil() as usize).max(1)
        };

        // Determine model filename and whether it uses FP16
        let (model_filename, use_fp16) = match self.upscale_model {
            UpscaleModel::RealESRAnimeVideoV3 => ("realesr-animevideov3_fp32.onnx", false),
            UpscaleModel::RealESRAnimeVideoV3Hf => ("realesr-animevideov3_fp16.onnx", true),
            UpscaleModel::RealESRGeneralx4v3 => ("realesr-general-x4v3_fp32.onnx", false),
            UpscaleModel::RealESRGeneralx4v3Hf => ("realesr-general-x4v3_fp16.onnx", true),
            UpscaleModel::RealESRGANx4Plus => ("RealESRGAN_x4plus_fp32.onnx", false),
            UpscaleModel::RealESRGANx4PlusHf => ("RealESRGAN_x4plus_fp16.onnx", true),
            UpscaleModel::RealESRGANx4PlusAnime => ("RealESRGAN_x4plus_anime_6B_fp32.onnx", false),
            UpscaleModel::RealESRGANx4PlusAnimeHf => ("RealESRGAN_x4plus_anime_6B_fp16.onnx", true),
        };

        let model_path = if num_upscale_passes > 0 {
            Some(self.find_upscale_model(model_filename)?)
        } else {
            None
        };

        if !self.silent {
            if let Some(path) = &model_path {
                println!("Loading upscaler: {}", path.display());
            }
        }

        let session_start = Instant::now();
        let mut session = model_path
            .as_ref()
            .map(|path| new_session(path, self.silent))
            .transpose()?;
        if std::env::var_os("PIXWORKER_TIMING").is_some() {
            eprintln!("upscale session: {:?}", session_start.elapsed());
        }

        if !self.silent {
            println!(
                "Model type: {}",
                match self.upscale_model {
                    UpscaleModel::RealESRAnimeVideoV3 => "Real-ESRAnimeVideoV3",
                    UpscaleModel::RealESRAnimeVideoV3Hf => "Real-ESRAnimeVideoV3 (Half-Precision)",
                    UpscaleModel::RealESRGeneralx4v3 => "Real-ESRGeneralx4v3",
                    UpscaleModel::RealESRGeneralx4v3Hf => "Real-ESRGeneralx4v3 (Half-Precision)",
                    UpscaleModel::RealESRGANx4Plus => "Real-ESRGANx4Plus",
                    UpscaleModel::RealESRGANx4PlusHf => "Real-ESRGANx4Plus (Half-Precision)",
                    UpscaleModel::RealESRGANx4PlusAnime => "Real-ESRGANx4PlusAnime",
                    UpscaleModel::RealESRGANx4PlusAnimeHf =>
                        "Real-ESRGANx4PlusAnime (Half-Precision)",
                }
            );
            println!(
                "Processing video upscaling from {}x{} to {}x{}",
                self.upscale.old_width,
                self.upscale.old_height,
                self.upscale.width,
                self.upscale.height
            );

            if num_upscale_passes == 0 {
                println!("Target is smaller than input, will only resize");
            } else if num_upscale_passes > 1 {
                println!(
                    "Will apply {}x upscaling {} times (total {}x), then resize to target resolution",
                    MODEL_SCALE as u32,
                    num_upscale_passes,
                    MODEL_SCALE.powi(num_upscale_passes as i32) as u32
                );
            }
        }

        // Collect all valid frame files first
        let mut frame_files: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(input_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let Some(ext) = path.extension() else {
                continue;
            };
            if ext == "png" {
                frame_files.push(path);
            }
        }

        // Sort frames to maintain order
        frame_files.sort_by(|a, b| {
            let x = a
                .file_stem()
                .unwrap()
                .to_str()
                .and_then(|num_str| num_str.parse::<f64>().ok())
                .unwrap_or(0.0);
            let y = b
                .file_stem()
                .unwrap()
                .to_str()
                .and_then(|num_str| num_str.parse::<f64>().ok())
                .unwrap_or(0.0);
            x.partial_cmp(&y).unwrap()
        });

        if frame_files.is_empty() {
            bail!("No valid image files found in upscale input directory");
        }

        if !self.silent {
            println!("Processing {} frames for upscaling...", frame_files.len());
        }

        let mut io_elapsed = Duration::ZERO;
        let mut run_elapsed = Duration::ZERO;
        let mut post_elapsed = Duration::ZERO;
        let total_start = Instant::now();
        for (idx, path) in frame_files.iter().enumerate() {
            if !self.silent && idx % 10 == 0 {
                println!("Upscaling frame {}/{}", idx + 1, frame_files.len());
            }

            // Load frame [H, W, C] with values in [0, 255]
            let io_start = Instant::now();
            let mut current_frame = load_frame(&path)?;
            io_elapsed += io_start.elapsed();

            // Apply upscaling multiple times if needed
            // Each pass applies 4x upscaling, so 2 passes = 16x total
            for pass in 0..num_upscale_passes {
                if !self.silent && num_upscale_passes > 1 && idx == 0 {
                    // Only show pass info for first frame to avoid spam
                    let (h, w, _) = current_frame.dim();
                    println!(
                        "  Pass {}/{}: upscaling {}x{} → {}x{}",
                        pass + 1,
                        num_upscale_passes,
                        w,
                        h,
                        w * 4,
                        h * 4
                    );
                }

                // Convert to CHW format [C, H, W] and normalize to [0, 1]
                // Real-ESRGAN expects normalized input in [0, 1] range
                let chw = self.hwc_to_chw(&current_frame)? / 255.0;

                // Add batch dimension: [1, C, H, W]
                let chw_batch = chw.view().insert_axis(Axis(0));

                // Check if model supports denoise_strength input (only realesr-general-x4v3)
                let supports_denoise = matches!(
                    self.upscale_model,
                    UpscaleModel::RealESRGeneralx4v3 | UpscaleModel::RealESRGeneralx4v3Hf
                );

                // Denoise strength: 1.0 favors detail, 0.0 favors denoise
                let denoise_strength = 0.1f32; // Balanced default

                // Run inference via RealESRGAN wrapper and convert to HWC [0,255]
                current_frame = if use_fp16 {
                    // FP16 path: convert input to fp16, run inference, convert output back
                    let chw_fp16 = chw_batch.mapv(f16::from_f32);

                    // Prepare denoise tensor in fp16
                    let denoise_tensor = Tensor::from_array(Array::from_shape_vec(
                        (1,),
                        vec![f16::from_f32(denoise_strength)],
                    )?)?;

                    let img_tensor = Tensor::from_array(chw_fp16)?;

                    let run_start = Instant::now();
                    let outputs = if supports_denoise {
                        session
                            .as_mut()
                            .unwrap()
                            .run(ort::inputs![img_tensor, denoise_tensor])?
                    } else {
                        session.as_mut().unwrap().run(ort::inputs![img_tensor])?
                    };
                    run_elapsed += run_start.elapsed();
                    let post_start = Instant::now();

                    let output = &outputs[0];

                    // Extract and process FP16 output directly to avoid extra conversions
                    let output_array = output.try_extract_array::<f16>()?;
                    let output_4d = output_array.into_dimensionality::<Ix4>()?;
                    let output_3d = output_4d.index_axis(Axis(0), 0);
                    // Convert fp16 to f32 and scale to [0, 255] in one operation
                    let frame = output_3d
                        .permuted_axes([1, 2, 0])
                        .as_standard_layout()
                        .mapv(|v| (v.to_f32() * 255.0).clamp(0.0, 255.0));
                    post_elapsed += post_start.elapsed();
                    frame
                } else {
                    // FP32 path: no type conversion needed
                    let chw_owned = chw_batch.to_owned();

                    // Prepare denoise tensor in fp32
                    let denoise_tensor =
                        Tensor::from_array(Array::from_shape_vec((1,), vec![denoise_strength])?)?;

                    let img_tensor = Tensor::from_array(chw_owned)?;

                    let run_start = Instant::now();
                    let outputs = if supports_denoise {
                        session
                            .as_mut()
                            .unwrap()
                            .run(ort::inputs![img_tensor, denoise_tensor])?
                    } else {
                        session.as_mut().unwrap().run(ort::inputs![img_tensor])?
                    };
                    run_elapsed += run_start.elapsed();
                    let post_start = Instant::now();

                    let output = &outputs[0];
                    let output_array = output.try_extract_array::<f32>()?;
                    let output_4d = output_array.to_owned().into_dimensionality::<Ix4>()?;
                    let output_3d = output_4d.index_axis(Axis(0), 0);
                    let hwc = output_3d
                        .permuted_axes([1, 2, 0])
                        .as_standard_layout()
                        .to_owned();

                    // Scale to [0, 255] directly
                    let frame = hwc.mapv(|v| (v * 255.0).clamp(0.0, 255.0));
                    post_elapsed += post_start.elapsed();
                    frame
                };
            }

            // Final resize to exact target dimensions
            let post_start = Instant::now();
            let final_frame = self.resize_to_target(
                &current_frame,
                self.upscale.width as usize,
                self.upscale.height as usize,
            )?;
            post_elapsed += post_start.elapsed();

            let io_start = Instant::now();
            save_frame(&final_frame, output_dir, idx)?;
            io_elapsed += io_start.elapsed();
        }
        if std::env::var_os("PIXWORKER_TIMING").is_some() {
            eprintln!(
                "upscale preprocessing: {:?}, Session::run: {:?}, postprocessing: {:?}, PNG I/O: {:?}",
                total_start
                    .elapsed()
                    .saturating_sub(run_elapsed + post_elapsed + io_elapsed),
                run_elapsed,
                post_elapsed,
                io_elapsed
            );
        }

        if !self.silent {
            println!(
                "Upscaling complete! Processed {} frames.",
                frame_files.len()
            );
        }

        Ok(())
    }

    /// Interpolate between two frames using the GIMMVFI model
    ///
    /// # Arguments
    /// * `session` - ONNX Runtime session with loaded model
    /// * `frame0` - First frame as ndarray [H, W, C] in RGB format, values [0, 255]
    /// * `frame1` - Second frame as ndarray [H, W, C] in RGB format, values [0, 255]
    /// * `times` - Temporal positions between frame0 and frame1
    ///
    /// # Returns
    /// Vector of interpolated frames as ndarray [H, W, C], values [0, 255]
    fn interpolate_frames(
        &self,
        session: &mut Session,
        frame0: &Array<f32, Ix3>,
        frame1: &Array<f32, Ix3>,
        times: &[f32],
    ) -> Result<Vec<Array<f32, Ix3>>> {
        let (orig_height, orig_width, channels) = frame0.dim();
        if channels != 3 {
            bail!("Expected RGB frames with 3 channels, got {}", channels);
        }

        // Validate that both frames have the same dimensions
        if frame1.dim() != (orig_height, orig_width, channels) {
            bail!("Frame dimensions mismatch");
        }
        let total_start = Instant::now();

        // Calculate padding to make dimensions divisible by 16 (FlowFormer requirement)
        // FlowFormer uses patch_size=8 but has additional constraints requiring divisor=16
        let divisor = 16;
        let pad_h = ((orig_height + divisor - 1) / divisor) * divisor - orig_height;
        let pad_w = ((orig_width + divisor - 1) / divisor) * divisor - orig_width;
        let pad_top = pad_h / 2;
        let pad_bottom = pad_h - pad_top;
        let pad_left = pad_w / 2;
        let pad_right = pad_w - pad_left;

        let padded_height = orig_height + pad_h;
        let padded_width = orig_width + pad_w;

        // Pad frames using replication mode
        let frame0_padded =
            self.pad_frame_replicate(frame0, pad_top, pad_bottom, pad_left, pad_right)?;
        let frame1_padded =
            self.pad_frame_replicate(frame1, pad_top, pad_bottom, pad_left, pad_right)?;

        // Use padded dimensions for processing
        let (height, width) = (padded_height, padded_width);

        // Convert frames from [H, W, C] to [C, H, W] and normalize to [0, 1]
        let frame0_chw = self.hwc_to_chw(&frame0_padded)? / 255.0;
        let frame1_chw = self.hwc_to_chw(&frame1_padded)? / 255.0;

        // Stack frames to create input tensor [1, C, 2, H, W]
        let frame0_batch = frame0_chw.view().insert_axis(Axis(0));
        let frame1_batch = frame1_chw.view().insert_axis(Axis(0));
        let img_xs = stack(Axis(2), &[frame0_batch, frame1_batch])?;

        // Determine dtype based on model wrapper
        let use_fp16 = matches!(
            self.vfi_model,
            VFIModel::GimmVfiFPHf | VFIModel::GimmVfiRPHf
        );
        let img_input_fp16 = if use_fp16 {
            Some(Tensor::from_array(img_xs.mapv(f16::from_f32))?)
        } else {
            None
        };
        let img_input_fp32 = if use_fp16 {
            None
        } else {
            Some(Tensor::from_array(img_xs)?)
        };
        let mut coord_array = self.generate_coord(1, height, width, 0.0)?;

        // Generate all interpolated frames
        let mut result_frames = Vec::with_capacity(times.len());
        let mut run_elapsed = Duration::ZERO;
        let mut post_elapsed = Duration::ZERO;
        let mut unpad_elapsed = Duration::ZERO;

        let mut infer_frame = |t_value: f32| -> Result<Array<f32, Ix3>> {
            // Generate coordinate tensor (always fp32)
            coord_array.slice_mut(s![.., .., .., .., 0]).fill(t_value);

            if use_fp16 {
                // FP16 path: img_xs and t are fp16, coord is always fp32
                let t_array = Array::from_shape_vec((1,), vec![f16::from_f32(t_value)])?;

                let coord_input = TensorRef::from_array_view(coord_array.view())?;
                let t_input = Tensor::from_array(t_array)?;

                let run_start = Instant::now();
                let outputs = session.run(ort::inputs![
                    img_input_fp16.as_ref().unwrap(),
                    coord_input,
                    t_input
                ])?;
                run_elapsed += run_start.elapsed();
                let post_start = Instant::now();
                let frame = self.extract_padded_frame(&outputs[0], true)?;
                post_elapsed += post_start.elapsed();
                Ok(frame)
            } else {
                // FP32 path: all tensors in fp32
                let t_array = Array::from_shape_vec((1,), vec![t_value])?;

                let coord_input = TensorRef::from_array_view(coord_array.view())?;
                let t_input = Tensor::from_array(t_array)?;

                let run_start = Instant::now();
                let outputs = session.run(ort::inputs![
                    img_input_fp32.as_ref().unwrap(),
                    coord_input,
                    t_input
                ])?;
                run_elapsed += run_start.elapsed();
                let post_start = Instant::now();
                let frame = self.extract_padded_frame(&outputs[0], false)?;
                post_elapsed += post_start.elapsed();
                Ok(frame)
            }
        };

        for &t_value in times {
            let padded_frame = infer_frame(t_value)?;
            let post_start = Instant::now();
            let result_frame =
                self.unpad_frame(padded_frame, pad_top, pad_left, orig_height, orig_width)?;
            unpad_elapsed += post_start.elapsed();
            result_frames.push(result_frame);
        }

        post_elapsed += unpad_elapsed;

        if std::env::var_os("PIXWORKER_TIMING").is_some() {
            eprintln!(
                "vfi preprocessing: {:?}, Session::run: {:?}, postprocessing: {:?}",
                total_start
                    .elapsed()
                    .saturating_sub(run_elapsed + post_elapsed),
                run_elapsed,
                post_elapsed
            );
        }
        Ok(result_frames)
    }

    /// Generate coordinate tensor for GIMMVFI INR sampling in fp32
    ///
    /// # Arguments
    /// * `batch_size` - Batch dimension size (typically 1)
    /// * `height` - Spatial height dimension
    /// * `width` - Spatial width dimension
    /// * `t_value` - Temporal coordinate value in range [0, 1]
    ///
    /// # Returns
    /// Coordinate tensor of shape [batch_size, 1, height, width, 3] in fp32
    fn generate_coord(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        t_value: f32,
    ) -> Result<Array<f32, ndarray::Dim<[usize; 5]>>> {
        // CRITICAL: Coordinate generation must match Python's CoordSampler3D.shape2coordinate
        // - t_value: NOT mapped to coord_range, used as-is (e.g., 0.5 for middle frame)
        // - spatial (h, w): pixel centers mapped to coord_range [-1, 1]
        //   Formula: coord = coord_range[0] + (coord_range[1] - coord_range[0]) * ((pixel + 0.5) / size)
        //   For coord_range=[-1, 1]: coord = -1 + 2 * ((pixel + 0.5) / size)
        Ok(Array::from_shape_fn(
            (batch_size, 1, height, width, 3),
            |(_, _, h, w, component)| match component {
                0 => t_value, // t: raw value in [0, 1], NOT mapped to [-1, 1]
                1 => -1.0 + 2.0 * ((h as f32 + 0.5) / height as f32), // y (h)
                2 => -1.0 + 2.0 * ((w as f32 + 0.5) / width as f32), // x (w)
                _ => unreachable!("coordinate component out of range"),
            },
        ))
    }

    /// Convert frame from HWC to CHW layout
    fn hwc_to_chw(&self, frame: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>> {
        Ok(frame.view().permuted_axes([2, 0, 1]).to_owned())
    }

    /// Pad a frame using edge replication (similar to PyTorch's F.pad with mode='replicate')
    fn pad_frame_replicate(
        &self,
        frame: &Array<f32, Ix3>,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
    ) -> Result<Array<f32, Ix3>> {
        let (height, width, channels) = frame.dim();
        let new_height = height + pad_top + pad_bottom;
        let new_width = width + pad_left + pad_right;
        Ok(Array::from_shape_fn(
            (new_height, new_width, channels),
            |(h, w, c)| {
                let src_h = if h < pad_top {
                    0
                } else if h >= pad_top + height {
                    height - 1
                } else {
                    h - pad_top
                };

                let src_w = if w < pad_left {
                    0
                } else if w >= pad_left + width {
                    width - 1
                } else {
                    w - pad_left
                };

                frame[[src_h, src_w, c]]
            },
        ))
    }

    /// Remove padding from a frame to restore original dimensions
    fn unpad_frame(
        &self,
        padded_frame: Array<f32, Ix3>,
        pad_top: usize,
        pad_left: usize,
        orig_height: usize,
        orig_width: usize,
    ) -> Result<Array<f32, Ix3>> {
        if padded_frame.dim().0 == orig_height && padded_frame.dim().1 == orig_width {
            return Ok(padded_frame);
        }
        Ok(padded_frame
            .slice(s![
                pad_top..pad_top + orig_height,
                pad_left..pad_left + orig_width,
                ..
            ])
            .to_owned())
    }

    fn extract_padded_frame(&self, output: &Value, use_fp16: bool) -> Result<Array<f32, Ix3>> {
        if use_fp16 {
            let output_array = output.try_extract_array::<f16>()?;
            let chw_4d = output_array.into_dimensionality::<Ix4>()?;
            let hwc = chw_4d.index_axis(Axis(0), 0).permuted_axes([1, 2, 0]);
            Ok(hwc
                .as_standard_layout()
                .mapv(|value| (value.to_f32() * 255.0).clamp(0.0, 255.0)))
        } else {
            let output_array = output.try_extract_array::<f32>()?;
            let chw_4d = output_array.into_dimensionality::<Ix4>()?;
            let hwc = chw_4d.index_axis(Axis(0), 0).permuted_axes([1, 2, 0]);
            Ok(hwc
                .as_standard_layout()
                .mapv(|value| (value * 255.0).clamp(0.0, 255.0)))
        }
    }

    /// Find VFI model file in workspace or model directory
    fn find_vfi_model(&self, filename: &str) -> Result<PathBuf> {
        // try user's home directory model path
        if let Some(home_dir) = dirs::home_dir() {
            let model_path = home_dir
                .join(".cache")
                .join("pixworker")
                .join("models")
                .join("vfi")
                .join(filename);

            if model_path.exists() {
                return Ok(model_path);
            }

            // If not found, ask user to agree to license before downloading
            println!("\n=== GIMM-VFI Model License Agreement ===");
            println!("The GIMM-VFI model is licensed under S-Lab License 1.0.");
            println!("License: https://github.com/GSeanCDAT/GIMM-VFI/blob/main/LICENSE");
            println!("\nThis license PROHIBITS commercial use.");
            println!("You may use this model for:");
            println!("  - Personal, non-commercial projects");
            println!("  - Academic research");
            println!("  - Educational purposes");
            println!("\nYou may NOT use this model for:");
            println!("  - Commercial products or services");
            println!("  - Any profit-generating activities");
            println!("\nFor commercial use, you must contact the contributors.");
            println!(
                "\nBy downloading this model, you agree to comply with the S-Lab License 1.0."
            );
            println!("========================================\n");

            print!("Do you agree to the license terms? (y/N): ");
            io::Write::flush(&mut io::stdout())?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim().to_lowercase();

            if input != "y" && input != "yes" {
                bail!(
                    "Model download cancelled. You must agree to the license to use GIMM-VFI models."
                );
            }

            // If not found, try download from huggingface.co
            let url = format!(
                "https://huggingface.co/universonic/GIMM-VFI/resolve/main/{}",
                filename
            );
            println!("Downloading VFI model from {}...", url);
            let response = reqwest::blocking::get(&url)?;
            if response.status().is_success() {
                let bytes = response.bytes()?;
                fs::create_dir_all(model_path.parent().unwrap())?;
                fs::write(&model_path, &bytes)?;
                println!("Model downloaded and saved to {}", model_path.display());
                return Ok(model_path);
            } else {
                println!(
                    "Failed to download model from {}: HTTP {}",
                    url,
                    response.status()
                );
            }
        }

        bail!(
            "VFI model '{}' not found. Please ensure the model is available in: ~/.cache/pixworker/models/vfi/",
            filename
        )
    }

    fn find_upscale_model(&self, filename: &str) -> Result<PathBuf> {
        if let Some(home_dir) = dirs::home_dir() {
            let model_path = home_dir
                .join(".cache")
                .join("pixworker")
                .join("models")
                .join("upscale")
                .join(filename);

            if model_path.exists() {
                return Ok(model_path);
            }

            // If not found, ask user to agree to license before downloading
            println!("\n=== Real-ESRGAN Model License Agreement ===");
            println!("The Real-ESRGAN model is licensed under BSD 3-Clause License.");
            println!(
                "License: https://raw.githubusercontent.com/xinntao/Real-ESRGAN/refs/heads/master/LICENSE"
            );
            println!("\nThis is a permissive open-source license that allows:");
            println!("  - Commercial use");
            println!("  - Modification");
            println!("  - Distribution");
            println!("  - Private use");
            println!("\nYou must:");
            println!("  - Include the copyright notice");
            println!("  - Include the license text");
            println!("  - Not use author's name for endorsement");
            println!(
                "\nBy downloading this model, you agree to comply with the BSD 3-Clause License."
            );
            println!("===========================================\n");

            print!("Do you agree to the license terms? (y/N): ");
            io::Write::flush(&mut io::stdout())?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim().to_lowercase();

            if input != "y" && input != "yes" {
                bail!(
                    "Model download cancelled. You must agree to the license to use Real-ESRGAN models."
                );
            }

            // If not found, try download from huggingface.co
            let url = format!(
                "https://huggingface.co/universonic/RealESRGAN/resolve/main/{}",
                filename
            );
            println!("Downloading RealESRGAN model from {}...", url);
            let response = reqwest::blocking::get(&url)?;
            if response.status().is_success() {
                let bytes = response.bytes()?;
                fs::create_dir_all(model_path.parent().unwrap())?;
                fs::write(&model_path, &bytes)?;
                println!("Model downloaded and saved to {}", model_path.display());
                return Ok(model_path);
            } else {
                println!(
                    "Failed to download model from {}: HTTP {}",
                    url,
                    response.status()
                );
            }
        }

        bail!(
            "Upscale model '{}' not found. Please place it in ~/.cache/pixworker/models/upscale/",
            filename
        )
    }

    fn resize_to_target(
        &self,
        frame: &Array<f32, Ix3>,
        width: usize,
        height: usize,
    ) -> Result<Array<f32, Ix3>> {
        let (current_h, current_w, _) = frame.dim();
        if current_h == height && current_w == width {
            return Ok(frame.clone());
        }

        let frame_u8 = frame
            .mapv(|v| v.clamp(0.0, 255.0) as u8)
            .into_raw_vec_and_offset()
            .0;
        let image =
            ImageBuffer::<Rgb<u8>, _>::from_raw(current_w as u32, current_h as u32, frame_u8)
                .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer for resizing"))?;

        let resized = DynamicImage::ImageRgb8(image)
            .resize_exact(width as u32, height as u32, FilterType::Lanczos3)
            .to_rgb8();

        let data: Vec<f32> = resized.into_raw().into_iter().map(|v| v as f32).collect();
        Ok(Array::from_shape_vec((height, width, 3), data)?)
    }
}

pub struct Upscale {
    pub old_width: u64,
    pub old_height: u64,
    pub width: u64,
    pub height: u64,
}

pub enum UpscaleModel {
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-animevideov3_fp32.onnx
    RealESRAnimeVideoV3,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-animevideov3_fp16.onnx
    RealESRAnimeVideoV3Hf,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-x4v3_fp32.onnx
    RealESRGeneralx4v3,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-x4v3_fp16.onnx
    RealESRGeneralx4v3Hf,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_fp32.onnx
    RealESRGANx4Plus,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_fp16.onnx
    RealESRGANx4PlusHf,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp32.onnx
    RealESRGANx4PlusAnime,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp16.onnx
    RealESRGANx4PlusAnimeHf,
}

pub struct VFI {
    pub old_fps: NTSC,
    pub fps: NTSC,
}

fn output_frame_count(source_count: usize, source: NTSC, target: NTSC) -> Result<usize> {
    let num = (source_count as u128)
        .checked_mul(target.num as u128)
        .and_then(|value| value.checked_mul(source.den as u128))
        .ok_or_else(|| anyhow::anyhow!("Output frame count exceeds supported range"))?;
    let den = target.den as u128 * source.num as u128;
    Ok(num.div_ceil(den).try_into()?)
}

fn frame_position(output_idx: usize, source: NTSC, target: NTSC) -> Result<(usize, f32)> {
    let num = (output_idx as u128)
        .checked_mul(target.den as u128)
        .and_then(|value| value.checked_mul(source.num as u128))
        .ok_or_else(|| anyhow::anyhow!("Frame timestamp exceeds supported range"))?;
    let den = target.num as u128 * source.den as u128;
    Ok(((num / den).try_into()?, (num % den) as f32 / den as f32))
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
}

pub enum VFIModel {
    // https://huggingface.co/universonic/GIMM-VFI/resolve/main/gimmvfi_f_arb_lpips_fp32.onnx
    GimmVfiFP,
    // https://huggingface.co/universonic/GIMM-VFI/resolve/main/gimmvfi_f_arb_lpips_fp16.onnx
    GimmVfiFPHf,
    // https://huggingface.co/universonic/GIMM-VFI/resolve/main/gimmvfi_r_arb_lpips_fp32.onnx
    GimmVfiRP,
    // https://huggingface.co/universonic/GIMM-VFI/resolve/main/gimmvfi_r_arb_lpips_fp16.onnx
    GimmVfiRPHf,
}

// Format bytes to a human readable string, e.g. 1024 -> "1.00 KiB"
#[allow(dead_code)]
fn human(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut i = 0usize;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{} {}", bytes, UNITS[i])
    } else {
        format!("{:.2} {}", v, UNITS[i])
    }
}

fn new_session(model_path: &Path, silent: bool) -> Result<Session> {
    // Optimize ONNX Runtime for maximum CPU utilization
    let num_threads_intra = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let num_threads_inter = if num_threads_intra > 8 { 4 } else { 2 };

    // Configure execution providers based on platform
    // Try hardware acceleration first, fall back to CPU if unavailable
    let mut _builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(num_threads_intra)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_threads(num_threads_inter)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    #[cfg(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(
            any(target_os = "linux", target_os = "windows"),
            target_arch = "x86_64"
        )
    ))]
    let builder = &mut _builder;

    // Register execution providers in order of preference
    // ONNX Runtime will try each in order and fall back if unavailable
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let mut coreml_registered = false;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let coreml = CoreMLExecutionProvider::default().with_subgraphs(true);
        let coreml = match std::env::var("PIXWORKER_COREML_UNITS").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("default") => coreml,
            Ok("gpu") => coreml.with_compute_units(ComputeUnits::CPUAndGPU),
            Ok("ane") => coreml.with_compute_units(ComputeUnits::CPUAndNeuralEngine),
            _ => bail!("PIXWORKER_COREML_UNITS must be default, gpu or ane"),
        };
        if std::env::var_os("PIXWORKER_CPU_ONLY").is_none() && coreml.is_available()? {
            match coreml.register(builder) {
                Ok(_) => {
                    coreml_registered = true;
                    if !silent {
                        println!("✓ Enabled CoreML Execution Provider for inference.");
                    }
                }
                Err(e) => {
                    if !silent {
                        eprintln!("⚠️ CoreML Execution Provider failed to register: {}", e);
                    }
                }
            }
        } else {
            if !silent {
                println!("⚠️ CoreML Execution Provider not available.");
            }
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if std::env::var_os("PIXWORKER_CPU_ONLY").is_none() {
        let tensorrt = TensorRTExecutionProvider::default();
        if tensorrt.is_available()? {
            match tensorrt.register(builder) {
                Ok(_) => {
                    if !silent {
                        println!("✓ Enabled TensorRT Execution Provider for inference.");
                    }
                }
                Err(e) => {
                    if !silent {
                        eprintln!("⚠️ TensorRT Execution Provider failed to register: {}", e);
                    }
                }
            }
        } else {
            if !silent {
                println!("⚠️ TensorRT Execution Provider not available.");
            }
        }
        let cuda = CUDAExecutionProvider::default();
        if cuda.is_available()? {
            match cuda.register(builder) {
                Ok(_) => {
                    if !silent {
                        println!("✓ Enabled CUDA Execution Provider for inference.");
                    }
                }
                Err(e) => {
                    if !silent {
                        eprintln!("⚠️ CUDA Execution Provider failed to register: {}", e);
                    }
                }
            }
        } else {
            if !silent {
                println!("⚠️ CUDA Execution Provider not available.");
            }
        }
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    if std::env::var_os("PIXWORKER_CPU_ONLY").is_none() {
        let tensorrt = TensorRTExecutionProvider::default();
        if tensorrt.is_available()? {
            match tensorrt.register(builder) {
                Ok(_) => {
                    if !silent {
                        println!("✓ Enabled TensorRT Execution Provider for inference.");
                    }
                }
                Err(e) => {
                    if !silent {
                        eprintln!("⚠️ TensorRT Execution Provider failed to register: {}", e);
                    }
                }
            }
        } else {
            if !silent {
                println!("⚠️ TensorRT Execution Provider not available.");
            }
        }
        let cuda = CUDAExecutionProvider::default();
        if cuda.is_available()? {
            match cuda.register(builder) {
                Ok(_) => {
                    if !silent {
                        println!("✓ Enabled CUDA Execution Provider for inference.");
                    }
                }
                Err(e) => {
                    if !silent {
                        eprintln!("⚠️ CUDA Execution Provider failed to register: {}", e);
                    }
                }
            }
        } else {
            if !silent {
                println!("⚠️ CUDA Execution Provider not available.");
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let session = match _builder.commit_from_file(model_path) {
        Ok(session) => session,
        Err(error) if coreml_registered => {
            if !silent {
                eprintln!("CoreML model load failed ({error}); retrying on CPU.");
            }
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_intra_threads(num_threads_intra)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_inter_threads(num_threads_inter)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .commit_from_file(model_path)?
        }
        Err(error) => return Err(error.into()),
    };
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let session = _builder.commit_from_file(model_path)?;

    if !silent {
        println!(
            "✓ ONNX model loaded successfully from: {}",
            model_path.display()
        );
    }

    Ok(session)
}

/// Load a single frame from disk into memory as ndarray [H, W, C] with f32 values [0, 255]
fn load_frame(path: &PathBuf) -> Result<Array<f32, Ix3>> {
    let img = image::open(path)?.to_rgb8();
    let (width, height) = img.dimensions();

    let data: Vec<f32> = img
        .into_raw()
        .into_iter()
        .map(|value| value as f32)
        .collect();
    Ok(Array::from_shape_vec(
        (height as usize, width as usize, 3),
        data,
    )?)
}

/// Save a frame to disk as PNG
fn save_frame(frame: &Array<f32, Ix3>, output_dir: &Path, frame_idx: usize) -> Result<()> {
    let (height, width, _channels) = frame.dim();

    let frame_owned = frame.to_owned();
    let frame_u8 = frame_owned.mapv(|x| x.clamp(0.0, 255.0) as u8);
    let rgb_buffer = frame_u8.into_raw_vec_and_offset().0;

    let output_path = output_dir.join(format!("{}.png", frame_idx));
    image::save_buffer(
        output_path,
        &rgb_buffer,
        width as u32,
        height as u32,
        image::ColorType::Rgb8,
    )?;

    Ok(())
}
