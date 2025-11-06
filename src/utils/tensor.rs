use crate::utils::ffmpeg::ArchiveOptions;
use crate::utils::ffmpeg::{ExtractOptions, FFProbe};
use crate::utils::gimm::GimmVfi;
use crate::utils::realesrgan::RealESRGAN;
use anyhow::{Result, anyhow, bail};
use candle_core::Device;
use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array, Ix3};
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

        let gimm = GimmVfi::from_model(&model_path, device, use_fp16)?;

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
            let frame_start = load_frame(&frame_files[i])?;
            let frame_end = load_frame(&frame_files[i + 1])?;

            if !self.silent && i % 10 == 0 {
                println!("Processing frame pair {}/{}", i + 1, frame_files.len() - 1);
            }

            // Save original frame
            save_frame(&frame_start, &output_path, output_frame_idx)?;
            output_frame_idx += 1;

            let num_interp = frame_multiplier as usize - 1;
            // Generate interpolated frames using GIMM wrapper
            let interpolated_frames = gimm.run(&frame_start, &frame_end, &num_interp)?;

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

    fn process_upscale(&self, input_dir: &Path, output_dir: &Path) -> Result<()> {
        if !input_dir.exists() {
            bail!("Upscale input directory does not exist");
        }

        fs::create_dir_all(output_dir)?;

        // Calculate upscale factor needed
        let target_scale = self.upscale.width as f64 / self.upscale.old_width as f64;

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

        // Wrap session in RealESRGAN helper
        let supports_denoise = matches!(
            self.upscale_model,
            UpscaleModel::RealESRGeneralx4v3 | UpscaleModel::RealESRGeneralx4v3Hf
        );
        let device = auto_device()?;

        // Denoise strength: 1.0 favors detail, 0.0 favors denoise
        let denoise_strength = 0.1f32; // Balanced default

        let model = RealESRGAN::from_model(&model_path, device, use_fp16, supports_denoise)?;

        for (idx, path) in frame_files.iter().enumerate() {
            if !self.silent && idx % 10 == 0 {
                println!("Upscaling frame {}/{}", idx + 1, frame_files.len());
            }

            // Load frame [H, W, C] with values in [0, 255]
            let mut current_frame = load_frame(&path)?;

            current_frame = model.run(&current_frame, &target_scale, &denoise_strength)?;

            // Final resize to exact target dimensions
            let final_frame = self.resize_to_target(
                &current_frame,
                self.upscale.width as usize,
                self.upscale.height as usize,
            )?;

            save_frame(&final_frame, &output_dir.to_path_buf(), idx)?;
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

        let frame_u8_arr = frame.mapv(|v| v.clamp(0.0, 255.0) as u8);
        // capture shape/strides before consuming the array
        let shape = frame_u8_arr.dim();
        let strides = frame_u8_arr.strides().to_vec();
        let (raw_vec, offset_opt) = if frame_u8_arr.is_standard_layout() {
            frame_u8_arr.into_raw_vec_and_offset()
        } else {
            // Non-contiguous: make a contiguous copy and then take its raw vec
            let contiguous = frame_u8_arr.to_owned();
            contiguous.into_raw_vec_and_offset()
        };

        let rgb_buffer = if offset_opt.unwrap_or(0) == 0 {
            raw_vec
        } else {
            // Reconstruct contiguous HWC ordering using shape and strides
            let (h, w, c) = (shape.0, shape.1, shape.2);
            let offset = offset_opt.unwrap();
            let mut contiguous = Vec::with_capacity(h * w * c);
            for i in 0..h {
                for j in 0..w {
                    for k in 0..c {
                        let raw_idx = (offset as isize
                            + (i as isize) * strides[0]
                            + (j as isize) * strides[1]
                            + (k as isize) * strides[2])
                            as usize;
                        contiguous.push(raw_vec[raw_idx]);
                    }
                }
            }
            contiguous
        };
        let image =
            ImageBuffer::<Rgb<u8>, _>::from_raw(current_w as u32, current_h as u32, rgb_buffer)
                .ok_or_else(|| anyhow!("Failed to create image buffer for resizing"))?;

        let resized = DynamicImage::ImageRgb8(image)
            .resize_exact(width as u32, height as u32, FilterType::Lanczos3)
            .to_rgb8();

        let data: Vec<f32> = resized.into_raw().into_iter().map(|v| v as f32).collect();
        let frame = Array::from_shape_vec((height, width, 3), data)
            .map_err(|e| anyhow!("Failed to create shape array from frame: {}", e))?;
        Ok(frame)
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

/// Load a single frame from disk into memory as ndarray [H, W, C] with f32 values [0, 255]
fn load_frame(path: &PathBuf) -> Result<Array<f32, Ix3>> {
    let img = image::open(path)?.to_rgb8();
    let (width, height) = img.dimensions();

    let data: Vec<f32> = img
        .into_raw()
        .into_iter()
        .map(|value| value as f32)
        .collect();
    let frame = Array::from_shape_vec((height as usize, width as usize, 3), data)
        .map_err(|e| anyhow!("Failed to create shape array from frame: {}", e))?;
    Ok(frame)
}

/// Save a frame to disk as PNG
fn save_frame(frame: &Array<f32, Ix3>, output_dir: &PathBuf, frame_idx: usize) -> Result<()> {
    let (height, width, _channels) = frame.dim();

    // CRITICAL: Ensure array is in standard (contiguous) layout before converting to raw buffer
    // After permuted_axes(), the array may not be contiguous, causing incorrect memory layout
    let frame_owned = frame.to_owned();

    // Convert f32 [0, 255] to contiguous u8 buffer in HWC (row-major) order
    let frame_u8 = frame_owned.mapv(|x| x.clamp(0.0, 255.0) as u8);
    // capture shape/strides before consuming the array
    let shape = frame_u8.dim();
    let strides = frame_u8.strides().to_vec();
    let (raw_vec, offset_opt) = if frame_u8.is_standard_layout() {
        frame_u8.into_raw_vec_and_offset()
    } else {
        let contiguous = frame_u8.to_owned();
        contiguous.into_raw_vec_and_offset()
    };

    let rgb_buffer = if offset_opt.unwrap_or(0) == 0 {
        raw_vec
    } else {
        // Non-zero offset: reconstruct contiguous logical data (H, W, C)
        let (h, w, c) = (shape.0, shape.1, shape.2);
        let offset = offset_opt.unwrap();
        let mut contiguous = Vec::with_capacity(h * w * c);
        for i in 0..h {
            for j in 0..w {
                for k in 0..c {
                    let raw_idx = (offset as isize
                        + (i as isize) * strides[0]
                        + (j as isize) * strides[1]
                        + (k as isize) * strides[2]) as usize;
                    contiguous.push(raw_vec[raw_idx]);
                }
            }
        }
        contiguous
    };

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
