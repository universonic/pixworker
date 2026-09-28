use crate::utils::tensor::enhance;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about,
    long_about = "A simple tool to handle media file operations."
)]
pub struct RootCmd {
    /// Turn debugging information on.
    // #[arg(short, long, action = clap::ArgAction::Count, default_value_t = 0)]
    // debug: u8,

    #[command(subcommand)]
    subcommand: Option<SubCommands>,
}

impl RootCmd {
    pub fn new() -> Self {
        Self::parse()
    }
    pub fn run(&self) -> Result<()> {
        match &self.subcommand {
            Some(SubCommands::Enhance {
                input,
                output,
                upscale,
                upscale_model,
                vfi,
                vfi_model,
                silent,
            }) => enhance(
                input,
                output,
                upscale,
                upscale_model,
                vfi,
                vfi_model,
                silent,
            )?,
            None => {}
        }
        Ok(())
    }
}

#[derive(Subcommand)]
pub enum SubCommands {
    /// Upscale video resolution and/or interpolate frames to increase frame rate.
    Enhance {
        /// Specify a video file as input.
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,

        /// Specify a output path. If not provided, it will be automatically generated.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Upscaling factor for resolution. "2.0" indicates doubling the resolution.
        /// Specific values such as "1920x1080" or "1080p" can also be used to define a target resolution.
        /// Powered by Real-ESRGAN.
        #[arg(short, long, value_name = "FLOAT64|STRING", default_value = "2.0")]
        upscale: Option<String>,

        /// Specify the upscaling model. Supported models: "realesr-animevideov3" (default), "realesr-animevideov3-hf", "realesr-generalx4v3", "realesr-generalx4v3-hf", "realesrgan-x4plus", "realesrgan-x4plus-hf", "realesrgan-x4plus-anime", "realesrgan-x4plus-anime-hf".
        #[arg(long, value_name = "STRING", default_value = "realesr-animevideov3")]
        upscale_model: Option<String>,

        /// Frame interpolation factor. Set to "1.0" to disable interpolation.
        /// You can specify values like "2.5" to convert 24fps to 60fps, or directly enter a target frame rate such as "60fps" for frame interpolation.
        #[arg(short, long, value_name = "FLOAT64|STRING", default_value = "1.0")]
        vfi: Option<String>,

        /// Specify the interpolation model. Supported model: "rife-v4.25" (default).
        #[arg(long, value_name = "STRING", default_value = "rife-v4.25")]
        vfi_model: Option<String>,

        /// Silent mode, no FFmpeg output except errors.
        #[arg(short, long, action = clap::ArgAction::SetTrue)]
        silent: Option<bool>,
    },
}
