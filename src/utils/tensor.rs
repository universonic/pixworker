use crate::utils::ffmpeg::ArchiveOptions;
use crate::utils::ffmpeg::{ExtractOptions, FFProbe};
use anyhow::{bail, Result};
use half::f16;
use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{s, Array, Axis, Ix3};
use crate::utils::gimm::GimmVfi;
use crate::utils::realesrgan::RealESRGAN;
use ndarray::stack;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, execution_providers::ExecutionProvider};
use scopeguard::defer;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::sync::Arc;
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

        let vfi = match vfi {
            Some(vfi_str) => {
                // Check if it's in "XXfps" format.
                let vfi_str = vfi_str.to_lowercase();
                if vfi_str.ends_with("fps") {
                    let fps_value = &vfi_str[..vfi_str.len() - 3];
                    let fps = fps_value
                        .parse::<u64>()
                        .map_err(|_| anyhow::anyhow!("Invalid fps format: {}", vfi_str))?;

                    let old_fps = info.r_frame_rate.unwrap().as_strict_fps();
                    if fps != old_fps && fps < old_fps * 2 {
                        bail!("Target FPS must be at least 2x the original FPS for interpolation");
                    }

                    VFI {
                        old_fps: old_fps,
                        fps,
                    }
                // Otherwise, treat it as a scaling factor.
                } else {
                    let factor = vfi_str
                        .parse::<f64>()
                        .map_err(|_| anyhow::anyhow!("Invalid VFI factor: {}", vfi_str))?;

                    if factor != 1.0 && factor < 2.0 {
                        bail!("Target FPS must be at least 2x the original FPS for interpolation");
                    }

                    let old_fps = info.r_frame_rate.unwrap().as_strict_fps();

                    VFI {
                        old_fps: old_fps,
                        fps: (old_fps as f64 * factor) as u64,
                    }
                }
            }
            None => VFI {
                old_fps: info.r_frame_rate.unwrap().as_strict_fps(),
                fps: 30,
            },
        };

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
        let mut tempdir = TempDir::new()?;
        tempdir.disable_cleanup(true);
        let tempdir_path = tempdir.path().to_path_buf();
        defer! {
            let _ = tempdir.close();
        };

        let tempdir_orig_frames = tempdir_path.join("frames_orig");
        let tempdir_frames = tempdir_path.join("frames");
        let tempdir_vfi = tempdir_path.join("vfi");

        let extract = ExtractOptions::new(
            self.input.to_path_buf(),
            tempdir_path.to_path_buf(),
            0,
            0,
            "pcm_s16le".to_string(),
            self.silent,
        );
        extract.process()?;

        // Rename extracted frames directory to avoid conflicts
        fs::rename(tempdir_frames.as_path(), tempdir_orig_frames.as_path())?;

        // Create ONNX session with optimizations
        let environment = Environment::builder()
            .with_name("pixworker")
            .build()?
            .into_arc();

        self.process_vfi(&environment, &tempdir_orig_frames, &tempdir_vfi)?;
        self.process_upscale(&environment, &tempdir_vfi, &tempdir_frames)?;

        // Encode all frames (original + interpolated) to output video
        let archive = ArchiveOptions::new(
            tempdir_path.to_path_buf(),
            self.output.to_path_buf(),
            self.vfi.fps,
            1.0,
            "pcm_s16le".to_string(),
            self.silent,
        );
        archive.process()?;
        Ok(())
    }

    fn process_vfi(&self, onnx_env: &Arc<Environment>, input_dir: &Path, output_dir: &Path) -> Result<()> {
        if self.vfi.fps <= self.vfi.old_fps {
            // no need to interpolate, just copy input to output
            fs::create_dir_all(output_dir)?;
            for entry in fs::read_dir(input_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    let filename = path.file_name().unwrap();
                    fs::copy(&path, output_dir.join(filename))?;
                }
            }
            return Ok(());
        }
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

        let session = new_builder(onnx_env, self.silent)?
            .with_model_from_file(&model_path)?;

        let use_fp16 = matches!(
            self.vfi_model,
            VFIModel::GimmVfiFPHf | VFIModel::GimmVfiRPHf
        );
        let gimm = GimmVfi::new(session, use_fp16);

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
                self.vfi.old_fps, self.vfi.fps
            );
        }

        // Calculate interpolation parameters
        let frame_multiplier = self.vfi.fps / self.vfi.old_fps;
        if frame_multiplier < 2 {
            bail!("Target FPS must be at least 2x the original FPS for interpolation");
        }

        if !self.silent {
            println!("Frame multiplier: {}x", frame_multiplier);
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
                frame_multiplier
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

        for i in 0..frame_files.len() - 1 {
            // Load only current frame pair (memory efficient)
            let frame_start: ndarray::ArrayBase<
                ndarray::OwnedRepr<f32>,
                ndarray::Dim<[usize; 3]>,
            > = load_frame(&frame_files[i])?;
            let frame_end = load_frame(&frame_files[i + 1])?;

            if !self.silent && i % 10 == 0 {
                println!("Processing frame pair {}/{}", i + 1, frame_files.len() - 1);
            }

            // Save original frame
            save_frame(&frame_start, &output_path, output_frame_idx)?;
            output_frame_idx += 1;

            // Generate interpolated frames using GIMM wrapper
            let interpolated_frames = self.interpolate_frames(
                &gimm,
                &frame_start,
                &frame_end,
                frame_multiplier as usize - 1,
            )?;

            // Save interpolated frames and immediately release memory
            for interp_frame in interpolated_frames {
                save_frame(&interp_frame, &output_path, output_frame_idx)?;
                output_frame_idx += 1;
                // interp_frame is dropped here, freeing memory
            }
        }

        // Save last frame
        let last_frame = load_frame(frame_files.last().unwrap())?;
        save_frame(&last_frame, &output_path, output_frame_idx)?;

        if !self.silent {
            println!(
                "Interpolation complete! Generated {} frames total.",
                output_frame_idx + 1
            );
        }
        Ok(())
    }

    fn process_upscale(&self, onnx_env: &Arc<Environment>, input_dir: &Path, output_dir: &Path) -> Result<()> {
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

        let model_path = self.find_upscale_model(model_filename)?;

        if !self.silent {
            println!("Loading upscaler: {}", model_path.display());
        }

        let session = new_builder(&onnx_env, self.silent)?
            .with_model_from_file(&model_path)?;

        if !self.silent {
            println!("Upscaler model loaded successfully");
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
                    UpscaleModel::RealESRGANx4PlusAnimeHf => "Real-ESRGANx4PlusAnime (Half-Precision)",
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

            let Some(ext) = path.extension() else { continue; };
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

        // Wrap session in RealESRGAN helper
        let supports_denoise = matches!(
            self.upscale_model,
            UpscaleModel::RealESRGeneralx4v3 | UpscaleModel::RealESRGeneralx4v3Hf
        );
        let model = RealESRGAN::new(session, use_fp16, supports_denoise);

        for (idx, path) in frame_files.iter().enumerate() {
            if !self.silent && idx % 10 == 0 {
                println!("Upscaling frame {}/{}", idx + 1, frame_files.len());
            }

            // Load frame [H, W, C] with values in [0, 255]
            let mut current_frame = load_frame(&path)?;
            
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
                        w, h,
                        w * 4, h * 4
                    );
                }
                
                // Convert to CHW format [C, H, W] and normalize to [0, 1]
                // Real-ESRGAN expects normalized input in [0, 1] range
                let chw = self.hwc_to_chw(&current_frame)? / 255.0;
                
                // Add batch dimension: [1, C, H, W]
                let chw_batch = chw.view().insert_axis(Axis(0));

                
                // Denoise strength: 1.0 favors detail, 0.0 favors denoise
                let denoise_strength = 0.1f32; // Balanced default

                // Run inference via RealESRGAN wrapper and convert to HWC [0,255]
                current_frame = if use_fp16 {
                    // FP16 path: convert input to fp16, run inference, convert output back
                    let chw_fp16 = chw_batch.mapv(|v| f16::from_f32(v));
                    let input_arr = chw_fp16.to_owned().into_dyn();

                    // Prepare denoise tensor in fp16
                    let denoise_array = Array::from_shape_vec((1,), vec![f16::from_f32(denoise_strength)])?;
                    let denoise_arr = denoise_array.into_dyn();

                    let output_4d = model.infer_fp16(input_arr, denoise_arr)?;
                    let output_3d = output_4d.index_axis(Axis(0), 0);
                    let hwc_view = output_3d.permuted_axes([1, 2, 0]);
                    let hwc = hwc_view.as_standard_layout().into_owned();

                    // Convert fp16 to f32 and scale to [0, 255]
                    hwc.mapv(|v| (v.to_f32() * 255.0).clamp(0.0, 255.0))
                } else {
                    // FP32 path
                    let input_arr = chw_batch.to_owned().into_dyn();
                    let denoise_array = Array::from_shape_vec((1,), vec![denoise_strength])?;
                    let denoise_arr = denoise_array.into_dyn();

                    let output_4d = model.infer_fp32(input_arr, denoise_arr)?;
                    let output_3d = output_4d.index_axis(Axis(0), 0);
                    let hwc_view = output_3d.permuted_axes([1, 2, 0]);
                    let hwc = hwc_view.as_standard_layout().into_owned();

                    hwc.mapv(|v| (v * 255.0).clamp(0.0, 255.0))
                };
            }
            
            // Final resize to exact target dimensions
            let final_frame = self.resize_to_target(
                &current_frame,
                self.upscale.width as usize,
                self.upscale.height as usize,
            )?;

            save_frame(&final_frame, &output_dir.to_path_buf(), idx)?;
        }

        if !self.silent {
            println!("Upscaling complete! Processed {} frames.", frame_files.len());
        }

        Ok(())
    }

    /// Interpolate between two frames using the GIMMVFI model
    ///
    /// # Arguments
    /// * `session` - ONNX Runtime session with loaded model
    /// * `frame0` - First frame as ndarray [H, W, C] in RGB format, values [0, 255]
    /// * `frame1` - Second frame as ndarray [H, W, C] in RGB format, values [0, 255]
    /// * `num_interp` - Number of frames to interpolate between frame0 and frame1
    ///
    /// # Returns
    /// Vector of interpolated frames as ndarray [H, W, C], values [0, 255]
    fn interpolate_frames(
        &self,
        gimm: &GimmVfi,
        frame0: &Array<f32, Ix3>,
        frame1: &Array<f32, Ix3>,
        num_interp: usize,
    ) -> Result<Vec<Array<f32, Ix3>>> {
        let (orig_height, orig_width, channels) = frame0.dim();
        if channels != 3 {
            bail!("Expected RGB frames with 3 channels, got {}", channels);
        }

        // Validate that both frames have the same dimensions
        if frame1.dim() != (orig_height, orig_width, channels) {
            bail!("Frame dimensions mismatch");
        }

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
        let use_fp16 = gimm.use_fp16;
        let img_xs_fp16 = use_fp16.then(|| img_xs.mapv(|value| f16::from_f32(value)));

        // Generate all interpolated frames
        let mut result_frames = Vec::with_capacity(num_interp);

        for i in 0..num_interp {
            // Calculate time value for this interpolation
            let t_value = (i + 1) as f32 / (num_interp + 1) as f32;

            // ================================================================
            // Generate all inputs based on model precision
            // This avoids unnecessary type conversions between fp16 and fp32
            // ================================================================

            // Note: ds_factor is now fixed at 1.0 inside the ONNX model
            // No need to pass it as an input anymore

            // Prepare inputs and run inference via GimmVfi wrapper
            let padded_frame = if use_fp16 {
                // FP16 path: img_xs and t are fp16, coord is ALWAYS fp32
                let coord_array = self.generate_coord(1, height, width, t_value)?;

                // Create t tensor in fp16
                let t_array = Array::from_shape_vec((1,), vec![f16::from_f32(t_value)])?;

                // Prepare owned arrays and convert to dynamic dims
                let img_xs_arr = img_xs_fp16
                    .as_ref()
                    .expect("fp16 tensor available")
                    .view()
                    .to_owned()
                    .into_dyn();
                let coord_arr = coord_array.into_dyn();
                let t_arr = t_array.into_dyn();

                // Run model and get owned 4D output [1, C, H, W]
                let output_4d = gimm.infer_fp16(img_xs_arr, coord_arr, t_arr)?;

                let output_3d = output_4d.index_axis(Axis(0), 0);
                let hwc_view = output_3d.permuted_axes([1, 2, 0]);
                let hwc = hwc_view.as_standard_layout().into_owned();

                // Convert fp16 to f32 and scale to [0, 255]
                hwc.mapv(|value| (value.to_f32() * 255.0).clamp(0.0, 255.0))
            } else {
                // FP32 path
                let coord_array = self.generate_coord(1, height, width, t_value)?;
                let t_array = Array::from_shape_vec((1,), vec![t_value])?;

                let img_xs_arr = img_xs.view().to_owned().into_dyn();
                let coord_arr = coord_array.into_dyn();
                let t_arr = t_array.into_dyn();

                let output_4d = gimm.infer_fp32(img_xs_arr, coord_arr, t_arr)?;

                let output_3d = output_4d.index_axis(Axis(0), 0);
                let hwc_view = output_3d.permuted_axes([1, 2, 0]);
                let hwc = hwc_view.as_standard_layout().into_owned();

                hwc.mapv(|value| (value * 255.0).clamp(0.0, 255.0))
            };

            // Unpad the output frame back to original dimensions
            let result_frame =
                self.unpad_frame(&padded_frame, pad_top, pad_left, orig_height, orig_width)?;

            result_frames.push(result_frame);
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
                0 => t_value,  // t: raw value in [0, 1], NOT mapped to [-1, 1]
                1 => -1.0 + 2.0 * ((h as f32 + 0.5) / height as f32),  // y (h)
                2 => -1.0 + 2.0 * ((w as f32 + 0.5) / width as f32),   // x (w)
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
        Ok(Array::from_shape_fn((new_height, new_width, channels), |(h, w, c)| {
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
        }))
    }

    /// Remove padding from a frame to restore original dimensions
    fn unpad_frame(
        &self,
        padded_frame: &Array<f32, Ix3>,
        pad_top: usize,
        pad_left: usize,
        orig_height: usize,
        orig_width: usize,
    ) -> Result<Array<f32, Ix3>> {
        Ok(
            padded_frame
                .slice(s![
                    pad_top..pad_top + orig_height,
                    pad_left..pad_left + orig_width,
                    ..
                ])
                .to_owned(),
        )
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
            println!("\nBy downloading this model, you agree to comply with the S-Lab License 1.0.");
            println!("========================================\n");
            
            print!("Do you agree to the license terms? (y/N): ");
            std::io::Write::flush(&mut std::io::stdout())?;
            
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim().to_lowercase();
            
            if input != "y" && input != "yes" {
                bail!("Model download cancelled. You must agree to the license to use GIMM-VFI models.");
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
            println!("License: https://raw.githubusercontent.com/xinntao/Real-ESRGAN/refs/heads/master/LICENSE");
            println!("\nThis is a permissive open-source license that allows:");
            println!("  - Commercial use");
            println!("  - Modification");
            println!("  - Distribution");
            println!("  - Private use");
            println!("\nYou must:");
            println!("  - Include the copyright notice");
            println!("  - Include the license text");
            println!("  - Not use author's name for endorsement");
            println!("\nBy downloading this model, you agree to comply with the BSD 3-Clause License.");
            println!("===========================================\n");
            
            print!("Do you agree to the license terms? (y/N): ");
            std::io::Write::flush(&mut std::io::stdout())?;
            
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim().to_lowercase();
            
            if input != "y" && input != "yes" {
                bail!("Model download cancelled. You must agree to the license to use Real-ESRGAN models.");
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
            .into_raw_vec();
        let image = ImageBuffer::<Rgb<u8>, _>::from_raw(
            current_w as u32,
            current_h as u32,
            frame_u8,
        )
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
    pub old_fps: u64,
    pub fps: u64,
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

fn new_builder(env: &Arc<Environment>, silent: bool) -> Result<SessionBuilder> {
    // Optimize ONNX Runtime for maximum CPU utilization
    let num_threads_intra = thread::available_parallelism()
        .map(|n| n.get() as i16)
        .unwrap_or(4);

    let num_threads_inter = if num_threads_intra > 8 {
        4
    } else {
        2
    };

    let builder = SessionBuilder::new(env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_threads_intra)?
        .with_inter_threads(num_threads_inter)?;
    let mut providers = Vec::new();

    #[cfg(target_os = "macos")]
    {
        let coreml = ExecutionProvider::CoreML(
            ort::execution_providers::CoreMLExecutionProviderOptions::default(),
        );
        if coreml.is_available() {
            if !silent {
                println!("CoreML execution provider is available");
            }
            providers.push(coreml);
        }
    }

    #[cfg(target_os = "windows")]
    {
        let tensorrt = ExecutionProvider::TensorRT(
            ort::execution_providers::TensorRTExecutionProviderOptions::default(),
        );
        if tensorrt.is_available() {
            if !silent {
                println!("TensorRT execution provider is available");
            }
            providers.push(tensorrt);
        }

        let cuda = ExecutionProvider::CUDA(
            ort::execution_providers::CUDAExecutionProviderOptions::default(),
        );
        if cuda.is_available() {
            if !silent {
                println!("CUDA execution provider is available");
            }
            providers.push(cuda);
        }
    }

    #[cfg(target_os = "linux")]
    {
        let tensorrt = ExecutionProvider::TensorRT(
            ort::execution_providers::TensorRTExecutionProviderOptions::default(),
        );
        if tensorrt.is_available() {
            if !silent {
                println!("TensorRT execution provider is available");
            }
            providers.push(tensorrt);
        }

        let cuda = ExecutionProvider::CUDA(
            ort::execution_providers::CUDAExecutionProviderOptions::default(),
        );
        if cuda.is_available() {
            if !silent {
                println!("CUDA execution provider is available");
            }
            providers.push(cuda);
        }
    }

    // Test each provider individually and collect successful ones
    let mut successful_providers = Vec::new();
    for provider in providers {
        // Create a temporary builder to test this provider
        let test_builder = SessionBuilder::new(env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads_intra)?
            .with_inter_threads(num_threads_inter)?;
        
        match test_builder.with_execution_providers(vec![provider.clone()]) {
            Ok(_) => {
                if !silent {
                    let provider_name = match &provider {
                        ExecutionProvider::CPU(_) => "CPU",
                        ExecutionProvider::CUDA(_) => "CUDA", 
                        ExecutionProvider::TensorRT(_) => "TensorRT",
                        #[cfg(target_os = "macos")]
                        ExecutionProvider::CoreML(_) => "CoreML",
                        #[cfg(target_os = "windows")]
                        ExecutionProvider::DirectML(_) => "DirectML",
                        _ => "Unknown",
                    };
                    println!("✓ {} execution provider test passed", provider_name);
                }
                successful_providers.push(provider);
            }
            Err(e) => {
                let provider_name = match &provider {
                    ExecutionProvider::CPU(_) => "CPU",
                    ExecutionProvider::CUDA(_) => "CUDA",
                    ExecutionProvider::TensorRT(_) => "TensorRT", 
                    #[cfg(target_os = "macos")]
                    ExecutionProvider::CoreML(_) => "CoreML",
                    #[cfg(target_os = "windows")]
                    ExecutionProvider::DirectML(_) => "DirectML",
                    _ => "Unknown",
                };
                if !silent {
                    eprintln!("✗ {} execution provider failed: {}", provider_name, e);
                    match &provider {
                        ExecutionProvider::CUDA(_) => {
                            eprintln!("  → Check if NVIDIA GPU drivers and CUDA are properly installed");
                            eprintln!("  → Verify CUDA version compatibility with ONNX Runtime");
                        }
                        ExecutionProvider::TensorRT(_) => {
                            eprintln!("  → Ensure TensorRT is installed and compatible");
                            eprintln!("  → Check NVIDIA driver version");
                        }
                        #[cfg(target_os = "windows")]
                        ExecutionProvider::DirectML(_) => {
                            eprintln!("  → Verify Windows version supports DirectML");
                            eprintln!("  → Check if compatible GPU drivers are installed");
                        }
                        #[cfg(target_os = "macos")]
                        ExecutionProvider::CoreML(_) => {
                            eprintln!("  → Ensure running on macOS with CoreML support");
                            eprintln!("  → Check macOS version compatibility");
                        }
                        _ => {}
                    }
                    eprintln!("  → Skipping this provider and continuing...");
                }
            }
        }
    }
    
    // Register all successful providers at once
    let final_builder = if !successful_providers.is_empty() {
        match builder.with_execution_providers(successful_providers.clone()) {
            Ok(new_builder) => {
                if !silent {
                    println!("✓ Successfully registered {} execution provider(s)", successful_providers.len());
                }
                new_builder
            }
            Err(e) => {
                if !silent {
                    eprintln!("✗ Failed to register execution providers collectively: {}", e);
                    eprintln!("  → Falling back to CPU-only execution");
                }
                // If registration fails, create a new builder without providers
                SessionBuilder::new(env)?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_threads_intra)?
                    .with_inter_threads(num_threads_inter)?
            }
        }
    } else {
        if !silent {
            println!("ℹ No hardware acceleration providers available, using CPU execution");
        }
        builder
    };
    
    Ok(final_builder)
}

/// Load a single frame from disk into memory as ndarray [H, W, C] with f32 values [0, 255]
fn load_frame(path: &PathBuf) -> Result<Array<f32, Ix3>> {
    let img = image::open(path)?.to_rgb8();
    let (width, height) = img.dimensions();

    let data: Vec<f32> = img.into_raw().into_iter().map(|value| value as f32).collect();
    Ok(Array::from_shape_vec(
        (height as usize, width as usize, 3),
        data,
    )?)
}

/// Save a frame to disk as PNG
fn save_frame(
    frame: &Array<f32, Ix3>,
    output_dir: &PathBuf,
    frame_idx: usize,
) -> Result<()> {
    let (height, width, _channels) = frame.dim();

    // CRITICAL: Ensure array is in standard (contiguous) layout before converting to raw buffer
    // After permuted_axes(), the array may not be contiguous, causing incorrect memory layout
    let frame_owned = frame.to_owned();

    // Convert f32 [0, 255] to contiguous u8 buffer in HWC (row-major) order
    let frame_u8 = frame_owned.mapv(|x| x.clamp(0.0, 255.0) as u8);
    let rgb_buffer = frame_u8.into_raw_vec();

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
