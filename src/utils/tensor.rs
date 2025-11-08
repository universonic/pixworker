use crate::utils::ffmpeg::{ArchiveOptions, ExtractOptions, FFProbe};
use crate::utils::gimm::GimmVfi;
use crate::utils::realesrgan::{RRDBNet, SRVGGNetCompact, blend_balanced_frame};
use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, Error, Tensor};
use half::f16;
use image::{DynamicImage, ImageBuffer, ImageReader};
use scopeguard::defer;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

pub fn enhance(
    input: &PathBuf,
    output: &Option<PathBuf>,
    upscale: &Option<String>,
    upscale_model: &Option<String>,
    denoise: &Option<f32>,
    vfi: &Option<String>,
    vfi_model: &Option<String>,
    silent: &Option<bool>,
) -> Result<()> {
    let options = EnhanceOptions::try_new(
        input,
        output,
        upscale,
        upscale_model,
        denoise,
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
    denoise: f32,
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
        denoise: f32,
        vfi: VFI,
        vfi_model: VFIModel,
        silent: bool,
    ) -> Self {
        Self {
            input,
            output,
            upscale,
            upscale_model,
            denoise,
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
        denoise: &Option<f32>,
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
                        anyhow!("Invalid width in resolution format: {}", values[0])
                    })?;
                    let height = values[1].parse::<u64>().map_err(|_| {
                        anyhow!("Invalid height in resolution format: {}", values[1])
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
                            * upscale_str
                                .parse::<f64>()
                                .map_err(|_| anyhow!("Invalid upscale factor: {}", upscale_str))?)
                            as u64,
                        height: (info.height.unwrap() as f64
                            * upscale_str
                                .parse::<f64>()
                                .map_err(|_| anyhow!("Invalid upscale factor: {}", upscale_str))?)
                            as u64,
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

        let denoise = denoise.unwrap_or(0.0);
        if denoise < 0.0 || denoise > 1.0 {
            bail!("Denoise strength must be between 0.0 and 1.0");
        }

        let vfi = match vfi {
            Some(vfi_str) => {
                // Check if it's in "XXfps" format.
                let vfi_str = vfi_str.to_lowercase();
                if vfi_str.ends_with("fps") {
                    let fps_value = &vfi_str[..vfi_str.len() - 3];
                    let fps = fps_value
                        .parse::<u64>()
                        .map_err(|_| anyhow!("Invalid fps format: {}", vfi_str))?;

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
                        .map_err(|_| anyhow!("Invalid VFI factor: {}", vfi_str))?;

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
            denoise,
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

        // Use Candle-based model loading (no ONNX Runtime)
        self.process_vfi(&tempdir_orig_frames, &tempdir_vfi)?;
        self.process_upscale(&tempdir_vfi, &tempdir_frames)?;

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

    fn process_vfi(&self, input_dir: &Path, output_dir: &Path) -> Result<()> {
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

        let use_fp16 = matches!(
            self.vfi_model,
            VFIModel::GimmVfiFPHf | VFIModel::GimmVfiRPHf
        );
        let device = auto_device()?;

        let gimm = GimmVfi::from_model(&model_path, device.clone(), use_fp16)?;

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
            // Load frames as Tensor [C, H, W] with f32 values [0, 1]
            let frame_start = load_image(&frame_files[i], &device)
                .map_err(|e| anyhow!("Error loading frame: {}", e))?;
            let frame_end = load_image(&frame_files[i + 1], &device)
                .map_err(|e| anyhow!("Error loading frame: {}", e))?;

            if !self.silent && i % 10 == 0 {
                println!("Processing frame pair {}/{}", i + 1, frame_files.len() - 1);
            }

            // Save original frame (denormalize f16 [0,1] to u8 [0,255])
            save_image(
                &frame_start,
                output_path.join(format!("{}.png", output_frame_idx)),
            )?;
            output_frame_idx += 1;

            let num_interp = frame_multiplier as usize - 1;
            // Generate interpolated frames using GIMM wrapper (Tensor-based)
            let interpolated_frames = gimm.inference(&frame_start, &frame_end, num_interp)?;

            // Save interpolated frames and immediately release memory
            for interp_tensor in interpolated_frames {
                save_image(
                    &interp_tensor,
                    output_path.join(format!("{}.png", output_frame_idx)),
                )?;
                output_frame_idx += 1;
            }
        }

        // Save last frame
        let last_frame = load_image(frame_files.last().unwrap(), &device)
            .map_err(|e| anyhow!("Error loading last frame: {}", e))?;
        save_image(
            &last_frame,
            output_path.join(format!("{}.png", output_frame_idx)),
        )?;

        if !self.silent {
            println!(
                "Interpolation complete! Generated {} frames total.",
                output_frame_idx + 1
            );
        }
        Ok(())
    }

    fn process_upscale(&self, input_dir: &Path, output_dir: &Path) -> Result<()> {
        if !input_dir.exists() {
            bail!("Upscale input directory does not exist");
        }

        fs::create_dir_all(output_dir)?;

        // Calculate upscale factor needed
        let target_scale = self.upscale.width as f64 / self.upscale.old_width as f64;

        // Determine model filename and whether it uses FP16
        let (model_filename, dtype) = match self.upscale_model {
            UpscaleModel::RealESRAnimeVideoV3 => {
                ("realesr-animevideov3_fp32.safetensors", DType::F32)
            }
            UpscaleModel::RealESRAnimeVideoV3Hf => {
                ("realesr-animevideov3_fp16.safetensors", DType::F16)
            }
            UpscaleModel::RealESRGeneralx4v3 => {
                ("realesr-general-x4v3_fp32.safetensors", DType::F32)
            }
            UpscaleModel::RealESRGeneralx4v3Hf => {
                ("realesr-general-x4v3_fp16.safetensors", DType::F16)
            }
            UpscaleModel::RealESRGANx4Plus => ("RealESRGAN_x4plus_fp32.safetensors", DType::F32),
            UpscaleModel::RealESRGANx4PlusHf => ("RealESRGAN_x4plus_fp16.safetensors", DType::F16),
            UpscaleModel::RealESRGANx4PlusAnime => {
                ("RealESRGAN_x4plus_anime_6B_fp32.safetensors", DType::F32)
            }
            UpscaleModel::RealESRGANx4PlusAnimeHf => {
                ("RealESRGAN_x4plus_anime_6B_fp16.safetensors", DType::F16)
            }
        };

        let model_path = self.find_upscale_model(model_filename)?;

        if !self.silent {
            println!("Loading upscaler: {}", model_path.display());
        }

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

        // Load model using the new Candle-based API
        let device = auto_device()?;

        // Auto-detect architecture and load model
        let model = match self.upscale_model {
            UpscaleModel::RealESRGANx4Plus
            | UpscaleModel::RealESRGANx4PlusHf
            | UpscaleModel::RealESRGANx4PlusAnime
            | UpscaleModel::RealESRGANx4PlusAnimeHf => {
                // RRDBNet model
                RRDBNet::from_model(&model_path, device.clone(), dtype)?
            }
            UpscaleModel::RealESRAnimeVideoV3
            | UpscaleModel::RealESRAnimeVideoV3Hf
            | UpscaleModel::RealESRGeneralx4v3
            | UpscaleModel::RealESRGeneralx4v3Hf => {
                // SRVGGNetCompact model
                SRVGGNetCompact::from_model(&model_path, device.clone(), dtype)?
            }
        };

        // Load denoise model if needed
        let denoise_model = if self.denoise > 0.0
            && (self.upscale_model == UpscaleModel::RealESRGeneralx4v3
                || self.upscale_model == UpscaleModel::RealESRGeneralx4v3Hf)
        {
            let denoise_model_filename = if dtype == DType::F16 {
                "realesr-general-wdn-x4v3_fp16.safetensors"
            } else {
                "realesr-general-wdn-x4v3_fp32.safetensors"
            };
            let denoise_model_path = self.find_upscale_model(denoise_model_filename)?;
            Some(SRVGGNetCompact::from_model(
                &denoise_model_path,
                device.clone(),
                dtype,
            )?)
        } else {
            None
        };

        for (idx, path) in frame_files.iter().enumerate() {
            if !self.silent && idx % 10 == 0 {
                println!("Upscaling frame {}/{}", idx + 1, frame_files.len());
            }

            // Load frame as Tensor [C, H, W] with f16 values [0.0, 1.0]
            let tensor =
                load_image(&path, &device).map_err(|e| anyhow!("Error loading image: {}", e))?;

            // Run inference: Tensor [C, H, W] → [C, 4H, 4W]
            let mut upscaled_tensor = model.inference(&tensor, &target_scale)?;

            // Apply denoise if model is available
            if let Some(ref denoise_model) = denoise_model {
                let denoised_tensor = denoise_model.inference(&tensor, &target_scale)?;
                upscaled_tensor =
                    blend_balanced_frame(&upscaled_tensor, &denoised_tensor, self.denoise)?;
            }

            // Save resized tensor directly
            let output_path = output_dir.join(format!("{}.png", idx));
            resize_and_save_image(
                &upscaled_tensor,
                &output_path,
                self.upscale.height,
                self.upscale.width,
            )
            .map_err(|e| anyhow!("Error saving image: {}", e))?;
        }

        if !self.silent {
            println!(
                "Upscaling complete! Processed {} frames.",
                frame_files.len()
            );
        }

        Ok(())
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
}

pub struct Upscale {
    pub old_width: u64,
    pub old_height: u64,
    pub width: u64,
    pub height: u64,
}

#[derive(PartialEq)]
pub enum UpscaleModel {
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-animevideov3_fp32.safetensors
    RealESRAnimeVideoV3,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-animevideov3_fp16.safetensors
    RealESRAnimeVideoV3Hf,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-x4v3_fp32.safetensors
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-wdn-x4v3_fp32.safetensors
    RealESRGeneralx4v3,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-x4v3_fp16.safetensors
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/realesr-general-wdn-x4v3_fp16.safetensors
    RealESRGeneralx4v3Hf,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_fp32.safetensors
    RealESRGANx4Plus,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_fp16.safetensors
    RealESRGANx4PlusHf,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp32.safetensors
    RealESRGANx4PlusAnime,
    // https://huggingface.co/universonic/RealESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B_fp16.safetensors
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

/// Convert a DynamicImage to a Candle Tensor [C, H, W] with f16 values [0.0, 1.0] (normalized)
pub fn image_to_tensor(img: &DynamicImage, device: &Device) -> Result<Tensor> {
    let img = img.to_rgb8();
    let (width, height) = img.dimensions();
    // Convert u8 to f16 and normalize to [0.0, 1.0]
    let data: Vec<f16> = img
        .into_raw()
        .into_iter()
        .map(|v| f16::from_f32(v as f32 / 255.0))
        .collect();
    let data =
        Tensor::from_vec(data, (height as usize, width as usize, 3), device)?.permute((2, 0, 1))?;
    Ok(data)
}

/// Convert a Candle Tensor [C, H, W] with f16 values [0.0, 1.0] (normalized) to a DynamicImage
///
/// # Arguments
/// * `img` - A tensor of shape [3, H, W] with f16 values in range [0.0, 1.0]
///
/// # Returns
/// A DynamicImage with RGB8 format (values [0, 255])
pub fn tensor_to_image(img: &Tensor) -> Result<DynamicImage> {
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        bail!(
            "tensor_to_image expects an input of shape (3, height, width), got ({}, {}, {})",
            channel,
            height,
            width
        )
    }

    // Permute from CHW to HWC and flatten to 1D
    let img = img.permute((1, 2, 0))?.flatten_all()?;

    // Convert f16 to u8: denormalize from [0.0, 1.0] to [0, 255]
    let pixels_f16: Vec<f16> = img.to_vec1()?;
    let pixels: Vec<u8> = pixels_f16
        .into_iter()
        .map(|v| (v.to_f32() * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    let image: ImageBuffer<image::Rgb<u8>, Vec<u8>> = match ImageBuffer::from_raw(
        width as u32,
        height as u32,
        pixels,
    ) {
        Some(image) => image,
        None => bail!(
            "error converting tensor to image: failed to create ImageBuffer with dimensions {}x{}",
            width,
            height
        ),
    };
    Ok(DynamicImage::ImageRgb8(image))
}

/// Load an image from disk into a Candle Tensor [C, H, W] with f16 values [0.0, 1.0] (normalized)
pub fn load_image<P: AsRef<Path>>(p: P, device: &Device) -> Result<Tensor> {
    let img = ImageReader::open(p)?.decode().map_err(Error::wrap)?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = img
        .into_raw()
        .into_iter()
        .map(|value| f16::from_f32(value as f32 / 255.0))
        .collect();
    let data = Tensor::from_vec(data, (height, width, 3), device)?.permute((2, 0, 1))?;
    Ok(data)
}

/// Resizes an image tensor and saves it to disk using the image crate
/// Input expects shape [C, H, W] with f16 values in range [0.0, 1.0] (normalized)
pub fn resize_and_save_image<P: AsRef<Path>>(
    img: &Tensor,
    to: P,
    new_height: u64,
    new_width: u64,
) -> Result<()> {
    let img = if img.dtype() != DType::F16 {
        img.to_dtype(DType::F16)?
    } else {
        img.to_owned()
    };
    tensor_to_image(&img)?
        .resize(
            new_height as u32,
            new_width as u32,
            image::imageops::Lanczos3,
        )
        .save(to)
        .map_err(Error::wrap)?;
    Ok(())
}

/// Saves an image to disk using the image crate
/// Input expects shape [C, H, W] with f16 values in range [0.0, 1.0] (normalized)
pub fn save_image<P: AsRef<Path>>(img: &Tensor, to: P) -> Result<()> {
    let img = if img.dtype() != DType::F16 {
        img.to_dtype(DType::F16)?
    } else {
        img.to_owned()
    };
    tensor_to_image(&img)?.save(to).map_err(Error::wrap)?;
    Ok(())
}

fn auto_device() -> Result<Device> {
    let mut device = Device::Cpu;

    #[cfg(target_os = "macos")]
    {
        if candle_core::utils::metal_is_available() {
            println!("Using Metal device for acceleration");
            device = Device::new_metal(0)?;
        }
    }
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        if candle_core::utils::cuda_is_available() {
            println!("Using CUDA device for acceleration");
            device = Device::new_cuda(0)?;
        }
    }
    Ok(device)
}
