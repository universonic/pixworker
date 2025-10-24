use crate::utils::ntsc::NTSC;
use anyhow::{Result, bail};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tempfile::TempDir;

pub fn extract(
    input: &PathBuf,
    output: &Option<PathBuf>,
    group_cap: &Option<u64>,
    trans_frames: &Option<u64>,
    acodec: &Option<String>,
    silent: &Option<bool>,
) -> Result<()> {
    let options = ExtractOptions::try_new(input, output, group_cap, trans_frames, acodec, silent)?;
    options.process()?;
    Ok(())
}

pub struct ExtractOptions {
    input: PathBuf,
    output: PathBuf,
    group_cap: u64,
    trans_frames: u64,
    acodec: String,
    silent: bool,
}

impl ExtractOptions {
    pub fn new(
        input: PathBuf,
        output: PathBuf,
        group_cap: u64,
        trans_frames: u64,
        acodec: String,
        silent: bool,
    ) -> Self {
        Self {
            input,
            output,
            group_cap,
            trans_frames,
            acodec,
            silent,
        }
    }

    pub fn try_new(
        input: &PathBuf,
        output: &Option<PathBuf>,
        group_cap: &Option<u64>,
        trans_frames: &Option<u64>,
        acodec: &Option<String>,
        silent: &Option<bool>,
    ) -> Result<Self> {
        let input = input.as_path();
        if !input.exists() {
            bail!("Specified input video does not exist.");
        }
        println!("Input video: {}", input.display());

        let mut workdir: PathBuf;
        if !output.is_none() {
            workdir = output.as_ref().unwrap().to_path_buf();
        } else {
            let default_dir_name = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap()
                .to_string();

            workdir = input.parent().as_ref().unwrap().to_path_buf();
            workdir.push(Path::new(&default_dir_name));
        }

        let group_cap = group_cap.unwrap_or(0);
        let trans_frames = trans_frames.unwrap_or(10);

        if group_cap > 0 && group_cap < trans_frames {
            bail!(
                "Value of `group_cap` '{}' is less than `trans_frames`: {}",
                group_cap,
                trans_frames
            )
        }

        let acodec = acodec.clone().unwrap_or("pcm_s16le".to_string());
        let silent = silent.unwrap_or(false);

        Ok(Self::new(
            input.to_path_buf(),
            workdir,
            group_cap,
            trans_frames,
            acodec,
            silent,
        ))
    }

    pub fn process(&self) -> Result<()> {
        if self.output.exists() {
            if self.output.is_file() {
                if let Err(e) = fs::remove_file(&self.output) {
                    bail!(e)
                }
            } else if self.output.is_dir() {
                if let Err(e) = fs::remove_dir_all(&self.output) {
                    bail!(e)
                }
            }
            println!("Removed existing files in '{}'...", self.output.display());
        }

        if let Err(e) = fs::create_dir_all(&self.output) {
            bail!(e);
        }

        let mut frame_files_location = self.output.clone();
        frame_files_location.push("frames");
        if let Err(e) = fs::create_dir(frame_files_location.to_str().unwrap()) {
            bail!(e);
        }
        let frame_root = frame_files_location.clone();
        frame_files_location.push("%d.png");

        let mut audio_file_location = self.output.clone();
        audio_file_location.push("audio");
        if let Err(e) = fs::create_dir(audio_file_location.to_str().unwrap()) {
            bail!(e);
        }
        audio_file_location.push("0.wav");

        println!(
            "Extracting to directory '{}' with ffmpeg...",
            self.output.display()
        );

        // NOTE: Currently we do not support VFR videos.
        // We does not use `-vsync 0 -frame_pts 1` arguments because it may cause
        // extra frames being extracted. Stream will be converted to CFR during extraction.
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-i",
            self.input.to_str().unwrap(),
            "-map",
            "0:v:0",
            "-start_number",
            "0",
            frame_files_location.to_str().unwrap(),
            "-map",
            "0:a:0",
            "-acodec",
            self.acodec.as_str(),
            "-ar",
            "44100",
            "-ac",
            "2",
            audio_file_location.to_str().unwrap(),
        ]);
        if !self.silent {
            cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        }
        let mut child = cmd.spawn()?;

        let status = child.wait()?;
        if !status.success() {
            bail!("Error executing ffmpeg: {}", status.code().unwrap())
        }

        let mut frame_files: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(frame_root.as_path())? {
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
                .and_then(|num_str| num_str.parse::<u64>().ok())
                .unwrap_or(0);
            let y = b
                .file_stem()
                .unwrap()
                .to_str()
                .and_then(|num_str| num_str.parse::<u64>().ok())
                .unwrap_or(0);
            x.cmp(&y)
        });

        let frame_count = frame_files.len() as u64;
        println!("Extract {} frames in total.", frame_count);

        if frame_count == 0 {
            return Ok(());
        }

        if self.group_cap == 0 {
            return Ok(());
        }

        let group_count = if self.group_cap == 0 {
            0
        } else {
            (frame_count + self.group_cap - self.trans_frames - 1)
                / (self.group_cap - self.trans_frames)
        };
        println!("Organize into {} groups.", group_count);

        for group_index in 0..group_count {
            let start_index = (group_index * (self.group_cap - self.trans_frames)) as usize;
            let end_index = std::cmp::min(
                (start_index as u64 + self.group_cap) as usize,
                frame_count as usize - 1,
            );

            let group_dir_name = format!("{}-{}", start_index, end_index);
            let group_dir = frame_root.as_path().join(&group_dir_name);

            fs::create_dir_all(&group_dir)?;

            println!("Creating group#{}: {}", group_index, group_dir_name);

            for file_index in start_index..end_index {
                if file_index < frame_files.len() {
                    let source_path = &frame_files[file_index];
                    let file_name = source_path.file_name().unwrap();
                    let dest_path = group_dir.join(file_name);

                    fs::copy(source_path, &dest_path)?;
                }
            }
        }

        println!("Cleaning up...");
        for file in frame_files {
            if let Err(e) = fs::remove_file(file) {
                bail!(e);
            }
        }
        Ok(())
    }
}

pub fn archive(
    input: &PathBuf,
    output: &Option<PathBuf>,
    frame_rate: &Option<u64>,
    keyframes: &Option<f64>,
    acodec: &Option<String>,
    silent: &Option<bool>,
) -> Result<()> {
    let options = ArchiveOptions::try_new(input, output, frame_rate, keyframes, acodec, silent)?;
    options.process()?;
    Ok(())
}

pub struct ArchiveOptions {
    input: PathBuf,
    output: PathBuf,
    frame_rate: u64,
    keyframes: f64,
    acodec: String,
    silent: bool,
}

impl ArchiveOptions {
    pub fn new(
        input: PathBuf,
        output: PathBuf,
        frame_rate: u64,
        keyframes: f64,
        acodec: String,
        silent: bool,
    ) -> Self {
        Self {
            input,
            output,
            frame_rate,
            keyframes,
            acodec,
            silent,
        }
    }

    pub fn try_new(
        input: &PathBuf,
        output: &Option<PathBuf>,
        frame_rate: &Option<u64>,
        keyframes: &Option<f64>,
        acodec: &Option<String>,
        silent: &Option<bool>,
    ) -> Result<Self> {
        let input = input.as_path();
        if !input.exists() || !input.is_dir() {
            bail!("Specified input directory does not exist or is not a directory.");
        }
        println!("Input directory: {}", input.display());

        let mut actual_output: PathBuf;
        if !output.is_none() {
            actual_output = output.as_ref().unwrap().to_path_buf();
        } else {
            let default_file_name = input
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap()
                .to_string()
                + ".mov";

            actual_output = input.parent().as_ref().unwrap().to_path_buf();
            actual_output.push(Path::new(&default_file_name));
        }

        let frame_rate = frame_rate.unwrap_or(30);
        let keyframes = keyframes.unwrap_or(1.0);
        let acodec = acodec.clone().unwrap_or("pcm_s16le".to_string());
        let silent = silent.unwrap_or(false);

        Ok(Self::new(
            input.to_path_buf(),
            actual_output,
            frame_rate,
            keyframes,
            acodec,
            silent,
        ))
    }

    pub fn process(&self) -> Result<()> {
        if self.output.exists() {
            if self.output.is_dir() {
                if let Err(e) = fs::remove_dir_all(&self.output) {
                    bail!(e)
                }
            } else if self.output.is_file() {
                if let Err(e) = fs::remove_file(&self.output) {
                    bail!(e)
                }
            }
            println!("Removed existing files at '{}'...", self.output.display());
        }

        let mut frame_files_location = self.input.clone();
        frame_files_location.push("frames");

        if !frame_files_location.exists() || !frame_files_location.is_dir() {
            bail!("There is no frames directory inside the input directory.");
        }

        // List frame group directories if there are any.
        let mut frame_dirs: Vec<PathBuf> = Vec::new();
        let mut frame_files: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(frame_files_location.as_path())? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Check if the directory name is in the format of "start-end".
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if let Some(idx) = name.find("-") {
                        if idx > 0 && idx < name.len() - 1 {
                            frame_dirs.push(path);
                        }
                    }
                }
                continue;
            }
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "png" {
                        frame_files.push(path);
                    }
                }
            }
        }

        frame_dirs.sort_by(|a, b| {
            let x = get_first_num_from_group_name(a.file_name().unwrap().to_str().unwrap());
            let y = get_first_num_from_group_name(b.file_name().unwrap().to_str().unwrap());
            x.cmp(&y)
        });

        let tmp_dir = TempDir::new_in(frame_files_location.parent().unwrap())?;

        if !frame_dirs.is_empty() {
            // There are frame groups.
            // Gather all frame files from the groups to `frame_files_location`.
            println!("Found {} frame groups.", frame_dirs.len());

            if !frame_files.is_empty() {
                for file in &frame_files {
                    if let Err(e) = fs::remove_file(file) {
                        bail!(e);
                    }
                }
                frame_files.clear();
            }

            let mut seen_files = HashSet::new();
            for dir in frame_dirs {
                for entry in fs::read_dir(dir.as_path())? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.is_file() && path.extension().map_or(false, |ext| ext == "png") {
                        if let Some(file_name) = path.file_name().and_then(|name| name.to_str()) {
                            if seen_files.insert(file_name.to_string()) {
                                frame_files.push(path);
                            }
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

            let tmp_dir_path = tmp_dir.path().to_path_buf();
            for (index, file) in frame_files.iter().enumerate() {
                let new_file_name = format!("{}.png", index);
                let new_file_path = tmp_dir_path.join(new_file_name);
                fs::copy(file, &new_file_path)?;
            }
            frame_files_location = tmp_dir_path;
        }
        println!("Total {} frames.", frame_files.len());

        let mut audio_file_location = self.input.clone();
        audio_file_location.push("audio");
        audio_file_location.push("0.wav");
        let audio_info = FFProbe::new(&audio_file_location).inspect_audio()?;
        if audio_info.duration.is_none() {
            bail!("Failed to inspect audio file.");
        }

        if !audio_file_location.exists() || !audio_file_location.is_file() {
            bail!("Audio file does not exist or is not a file.");
        }

        println!(
            "Archiving to file '{}' with ffmpeg...",
            self.output.display()
        );

        let ntsc = NTSC::from_strict_fps(&self.frame_rate);
        let actual_frame_rate = round_to_decimal(ntsc.to_fps(), 8);
        let keyframe_interval =
            (self.keyframes as f64 * ntsc.num as f64 / ntsc.den as f64).round() as u64;
        let atempo = round_to_decimal(
            audio_info.duration.unwrap() * ntsc.num as f64
                / (frame_files.len() as f64 * ntsc.den as f64),
            8,
        )
        .min(2.0)
        .max(0.5);

        // TODO: currently we only archive to H.265 MOV format in quality (lossless) profile.
        // We may add a performance profile in the future.
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-framerate",
            actual_frame_rate.to_string().as_str(),
            "-i",
            &format!("{}/%d.png", frame_files_location.to_str().unwrap()),
            "-i",
            audio_file_location.to_str().unwrap(),
            "-c:v",
            "libx265",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "lossless=1:aq-mode=3",
            "-profile:v",
            "main444-12",
            "-pix_fmt",
            "yuv444p",
            "-crf",
            "18",
            "-g",
            &keyframe_interval.to_string(),
            "-c:a",
            self.acodec.as_str(),
            "-af",
            &format!("atempo={:.8}", atempo),
            self.output.to_str().unwrap(),
        ]);
        if !self.silent {
            cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        }
        let mut child = cmd.spawn()?;
        let status = child.wait()?;
        if !status.success() {
            bail!("Error executing ffmpeg: {}", status.code().unwrap())
        }
        Ok(())
    }
}

pub struct FFProbe {
    input: PathBuf,
}

impl FFProbe {
    pub fn new(input: &PathBuf) -> Self {
        Self {
            input: input.clone(),
        }
    }

    pub fn inspect_audio(&self) -> Result<FFProbeAudioInfo> {
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "quiet",
                "-select_streams",
                "a:0",
                "-show_entries",
                "format=codec_name,sample_rate,channels,duration",
                "-of",
                "default=noprint_wrappers=1",
                self.input.to_str().unwrap(),
            ])
            .output()?;

        if !output.status.success() {
            bail!("Error executing ffprobe: {}", output.status);
        }

        let raw_str = String::from_utf8_lossy(&output.stdout);
        let lines = raw_str.trim().lines().collect::<Vec<&str>>();
        let mut info = FFProbeAudioInfo::new();

        // parse the output
        for line in lines {
            let kv_pair: Vec<&str> = line.split('=').collect();
            if kv_pair.len() != 2 {
                continue;
            }

            let key = kv_pair[0].trim();
            let value = kv_pair[1].trim();

            match key {
                "codec_name" => {
                    info.codec_name = Some(value.to_string());
                }
                "sample_rate" => {
                    if value.to_lowercase() != "n/a" {
                        info.sample_rate = match value.parse::<u64>() {
                            Ok(v) => Some(v),
                            Err(e) => {
                                println!("Parse sample_rate value '{}': {}", value, e);

                                None
                            }
                        };
                    }
                }
                "channels" => {
                    if value.to_lowercase() != "n/a" {
                        info.channels = match value.parse::<u64>() {
                            Ok(v) => Some(v),
                            Err(e) => {
                                println!("Parse channels value '{}': {}", value, e);

                                None
                            }
                        };
                    }
                }
                "duration" => {
                    if value.to_lowercase() != "n/a" {
                        info.duration = match value.parse::<f64>() {
                            Ok(v) => Some(v),
                            Err(e) => {
                                println!("Parse duration value '{}': {}", value, e);

                                None
                            }
                        };
                    }
                }
                _ => {}
            }
        }
        Ok(info)
    }

    pub fn inspect_video(&self) -> Result<FFProbeVideoInfo> {
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,duration",
                "-of",
                "default=noprint_wrappers=1",
                self.input.to_str().unwrap(),
            ])
            .output()?;

        if !output.status.success() {
            bail!("Error executing ffprobe: {}", output.status);
        }

        let raw_str = String::from_utf8_lossy(&output.stdout);
        let lines = raw_str.trim().lines().collect::<Vec<&str>>();
        let mut info = FFProbeVideoInfo::new();

        // parse the output
        for line in lines {
            let kv_pair: Vec<&str> = line.split('=').collect();
            if kv_pair.len() != 2 {
                continue;
            }

            let key = kv_pair[0].trim();
            let value = kv_pair[1].trim();

            match key {
                "codec_name" => {
                    info.codec_name = Some(value.to_string());
                }
                "width" => {
                    if value.to_lowercase() != "n/a" {
                        info.width = match value.parse::<u64>() {
                            Ok(v) => Some(v),
                            Err(e) => {
                                println!("Parse width value '{}': {}", value, e);

                                None
                            }
                        };
                    }
                }
                "height" => {
                    if value.to_lowercase() != "n/a" {
                        info.height = match value.parse::<u64>() {
                            Ok(v) => Some(v),
                            Err(e) => {
                                println!("Parse height value '{}': {}", value, e);

                                None
                            }
                        };
                    }
                }
                "pix_fmt" => {
                    info.pix_fmt = Some(value.to_string());
                }
                "r_frame_rate" => {
                    info.r_frame_rate = NTSC::from_string(value);
                }
                "avg_frame_rate" => {
                    info.avg_frame_rate = NTSC::from_string(value);
                }
                "duration" => {
                    if value.to_lowercase() != "n/a" {
                        info.duration = match value.parse::<f64>() {
                            Ok(v) => Some(v),
                            Err(e) => {
                                println!("Parse duration value '{}': {}", value, e);

                                None
                            }
                        };
                    }
                }
                _ => {}
            }
        }
        Ok(info)
    }
}

pub struct FFProbeVideoInfo {
    pub codec_name: Option<String>,
    pub width: Option<u64>,
    pub height: Option<u64>,
    pub pix_fmt: Option<String>,
    pub r_frame_rate: Option<NTSC>,
    pub avg_frame_rate: Option<NTSC>,
    pub duration: Option<f64>,
}

impl FFProbeVideoInfo {
    fn new() -> Self {
        Self {
            codec_name: None,
            width: None,
            height: None,
            pix_fmt: None,
            r_frame_rate: None,
            avg_frame_rate: None,
            duration: None,
        }
    }
}

pub struct FFProbeAudioInfo {
    pub codec_name: Option<String>,
    pub sample_rate: Option<u64>,
    pub channels: Option<u64>,
    pub duration: Option<f64>,
}

impl FFProbeAudioInfo {
    fn new() -> Self {
        Self {
            codec_name: None,
            sample_rate: None,
            channels: None,
            duration: None,
        }
    }
}

fn round_to_decimal(value: f64, decimal_places: u32) -> f64 {
    let multiplier = 10_f64.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

fn get_first_num_from_group_name(s: &str) -> u64 {
    s.split('-')
        .next()
        .and_then(|num_str| num_str.parse::<u64>().ok())
        .unwrap_or(0)
}
