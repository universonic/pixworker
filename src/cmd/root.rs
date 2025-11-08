use crate::utils::ffmpeg::{archive, extract};
use crate::utils::onnx as onnx_util;
use crate::utils::tensor::enhance;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about,
    long_about = "A simple tool to handle media file operations.",
    arg_required_else_help(true)
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
            Some(SubCommands::Extract {
                input,
                output,
                group_cap,
                trans_frames,
                acodec,
                silent,
            }) => extract(input, output, group_cap, trans_frames, acodec, silent)?,
            Some(SubCommands::Archive {
                input,
                output,
                frame_rate,
                keyframes,
                acodec,
                silent,
            }) => archive(input, output, frame_rate, keyframes, acodec, silent)?,
            Some(SubCommands::Enhance {
                input,
                output,
                upscale,
                upscale_model,
                denoise,
                vfi,
                vfi_model,
                silent,
            }) => enhance(
                input,
                output,
                upscale,
                upscale_model,
                denoise,
                vfi,
                vfi_model,
                silent,
            )?,
            Some(SubCommands::Devtool {
                dev: Some(DevToolSubCommands::Onnx { path, graphviz, list }),
            }) => {
                onnx_util::inspect(path, *graphviz, *list)?;
            }
            Some(SubCommands::Devtool { dev: None }) => {}
            None => {}
        }
        Ok(())
    }
}

#[derive(Subcommand)]
pub enum SubCommands {
    /// Extract frames and audio from the input video file.
    Extract {
        /// Specify a video file as input.
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,

        /// Specify a output path. If not provided, it will be automatically generated.
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,

        /// Organize frames into several groups with specified group capacity.
        #[arg(short, long, action = clap::ArgAction::Set, value_parser = clap::value_parser!(u64), value_name = "UINT64", default_value = "0")]
        group_cap: Option<u64>,

        /// Transition frames that will be prepended between splitted groups.
        #[arg(short, long, action = clap::ArgAction::Set, value_parser = clap::value_parser!(u64), value_name = "UINT64", default_value = "10")]
        trans_frames: Option<u64>,

        /// Set the audio codec of the output video.
        #[arg(long, value_name = "STRING", default_value = "pcm_s16le")]
        acodec: Option<String>,

        /// Silent mode, no ffmpeg output except errors.
        #[arg(short, long, action = clap::ArgAction::SetTrue)]
        silent: Option<bool>,
    },

    /// Archive frames and audio into a h.265 video file.
    Archive {
        /// Specify the directory containing frames and audio.
        #[arg(short, long, value_name = "DIR")]
        input: PathBuf,

        /// Specify a output path. If not provided, it will be automatically generated.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Set the frame rate of the output video (fps).
        #[arg(short, long, value_name = "UINT64", default_value = "30")]
        frame_rate: Option<u64>,

        /// Set the keyframes interval (seconds).
        #[arg(short, long, value_name = "FLOAT64", default_value = "1.0")]
        keyframes: Option<f64>,

        /// Set the audio codec of the output video.
        #[arg(long, value_name = "STRING", default_value = "pcm_s16le")]
        acodec: Option<String>,

        /// Silent mode, no ffmpeg output except errors.
        #[arg(short, long, action = clap::ArgAction::SetTrue)]
        silent: Option<bool>,
    },

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

        /// Denoising strength for upscaling. Only meaningful while using "realesr-generalx4v3" or "realesr-generalx4v3-hf". Range: 0.0 (less denoise) to 1.0 (more denoise).
        #[arg(short, long, value_name = "FLOAT32", default_value = "0.0")]
        denoise: Option<f32>,

        /// Frame interpolation factor. Set to "1.0" to disable interpolation.
        /// You can specify values like "2.5" to convert 24fps to 60fps, or directly enter a target frame rate such as "60fps" for frame interpolation.
        #[arg(short, long, value_name = "FLOAT64|STRING", default_value = "1.0")]
        vfi: Option<String>,

        /// Specify the interpolation model. Supported models: "gimm-vfi-f-p" (default), "gimm-vfi-f-p-hf", "gimm-vfi-r-p", "gimm-vfi-r-p-hf".
        #[arg(long, value_name = "STRING", default_value = "gimm-vfi-f-p")]
        vfi_model: Option<String>,

        /// Silent mode, no ffmpeg output except errors.
        #[arg(short, long, action = clap::ArgAction::SetTrue)]
        silent: Option<bool>,
    },

    /// Developer utilities
    #[command(hide = true)]
    Devtool {
        #[command(subcommand)]
        dev: Option<DevToolSubCommands>,
    },
}

#[derive(Subcommand)]
#[command(arg_required_else_help(true))]
pub enum DevToolSubCommands {
    /// Inspect an ONNX model file, inspired by huggingface/candle-onnx
    Onnx {
        /// Path to the ONNX model file
        #[arg(value_name = "FILE")]
        path: PathBuf,

        /// Generate a Graphviz DOT output next to the model file
        #[arg(short, long, action = clap::ArgAction::SetTrue)]
        graphviz: bool,

        /// List nodes in the model, printing their names and attributes
        #[arg(short, long, action = clap::ArgAction::SetTrue)]
        list: bool,
    },
}
