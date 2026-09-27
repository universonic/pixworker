use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, ImageBuffer, Rgb, imageops::FilterType};
use native_tch::{OUTPUT_SHAPE, infer, input_from_rgb, load_model};
use pixworker::utils::ffmpeg::{ArchiveOptions, ExtractOptions, FFProbe};
use std::{
    env, fs,
    path::Path,
    time::{Duration, Instant},
};
use tch::Device;
use tempfile::TempDir;

fn ensure_new(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => bail!("refusing to overwrite existing output: {}", path.display()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

fn main() -> Result<()> {
    let command_start = Instant::now();
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 && args.len() != 5 {
        bail!("usage: video <input-short.mkv> <new-output.mov> <model.safetensors> [mps|cuda]");
    }
    let (source, output, weights) = (
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
    );
    ensure!(
        source.is_file() && weights.is_file(),
        "input video and model must be existing files"
    );
    ensure!(
        output.extension().is_some_and(|ext| ext == "mov"),
        "output must be a new .mov file"
    );
    ensure!(
        output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .is_dir(),
        "output parent directory must exist"
    );
    ensure_new(output)?;
    let device = match args.get(4).map(String::as_str).unwrap_or("mps") {
        "mps" => {
            ensure!(
                env::var("PYTORCH_ENABLE_MPS_FALLBACK").as_deref() == Ok("0"),
                "set PYTORCH_ENABLE_MPS_FALLBACK=0 to prevent CPU fallback"
            );
            Device::Mps
        }
        "cuda" => {
            ensure!(tch::Cuda::is_available(), "CUDA unavailable");
            Device::Cuda(0)
        }
        other => bail!("unknown device: {other}"),
    };

    let info = FFProbe::new(&source.to_path_buf()).inspect_video()?;
    ensure!(
        info.width == Some(1280) && info.height == Some(720),
        "expected 1280x720 input"
    );
    let fps = info.r_frame_rate.context("missing source frame rate")?;
    ensure!(
        (fps.num, fps.den) == (24000, 1001),
        "expected 24000/1001 source fps"
    );
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);
    let _no_grad = tch::no_grad_guard();
    println!(
        "backend={device:?} input={} output={} fps={}/{} vfi=1 threads_intra={} threads_inter={} OMP_NUM_THREADS={} MKL_NUM_THREADS={} PYTORCH_ENABLE_MPS_FALLBACK={}",
        source.display(),
        output.display(),
        fps.num,
        fps.den,
        tch::get_num_threads(),
        tch::get_num_interop_threads(),
        env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        env::var("MKL_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        env::var("PYTORCH_ENABLE_MPS_FALLBACK").unwrap_or_else(|_| "unset".into())
    );

    // Keep every extraction, PNG and encoded staging file inside this crate's ignored target/.
    let work = TempDir::new_in(Path::new(env!("CARGO_MANIFEST_DIR")).join("target"))?;
    let extracted = work.path().join("extracted");
    let start = Instant::now();
    ExtractOptions::new(
        source.to_path_buf(),
        extracted.clone(),
        0,
        0,
        "pcm_s16le".into(),
        true,
    )
    .process()?;
    fs::rename(extracted.join("frames"), work.path().join("frames_orig"))?;
    fs::rename(extracted.join("audio"), work.path().join("audio"))?;
    fs::rename(work.path().join("frames_orig"), work.path().join("vfi"))?;
    let extract = start.elapsed();

    let start = Instant::now();
    let model = load_model(weights, device)?;
    let session_load = start.elapsed();
    let mut readback = vec![0.0_f32; OUTPUT_SHAPE.iter().product::<i64>() as usize];
    let mut paths: Vec<_> = fs::read_dir(work.path().join("vfi"))?
        .map(|entry| -> Result<_> {
            let path = entry?.path();
            let index = path
                .file_stem()
                .context("frame without index")?
                .to_string_lossy()
                .parse::<usize>()?;
            ensure!(
                path.is_file() && path.file_name().unwrap() == format!("{index}.png").as_str(),
                "unexpected frame: {}",
                path.display()
            );
            Ok((index, path))
        })
        .collect::<Result<_>>()?;
    paths.sort_unstable_by_key(|(index, _)| *index);
    ensure!(
        !paths.is_empty() && paths.iter().enumerate().all(|(i, (index, _))| i == *index),
        "extracted frames must be consecutive 0.png..N.png"
    );
    let frames_dir = work.path().join("frames");
    fs::create_dir(&frames_dir)?;
    let (mut png_io, mut preprocess, mut inference, mut postprocess) = (
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );
    for (index, path) in &paths {
        let start = Instant::now();
        let rgb = image::open(path)?.to_rgb8();
        png_io += start.elapsed();

        let start = Instant::now();
        let input = input_from_rgb(&rgb, device)?;
        preprocess += start.elapsed();
        drop(rgb);

        let start = Instant::now();
        infer(&model, input, &mut readback)?;
        inference += start.elapsed();

        let start = Instant::now();
        let plane = 2880 * 5120;
        let mut interleaved = Vec::with_capacity(3 * plane);
        for pixel in 0..plane {
            for channel in 0..3 {
                interleaved
                    .push((readback[channel * plane + pixel] * 255.0).clamp(0.0, 255.0) as u8);
            }
        }
        let frame = ImageBuffer::<Rgb<u8>, _>::from_raw(5120, 2880, interleaved)
            .context("invalid RGB buffer")?;
        let resized = DynamicImage::ImageRgb8(frame)
            .resize_exact(2560, 1440, FilterType::Lanczos3)
            .to_rgb8();
        postprocess += start.elapsed();

        let start = Instant::now();
        image::save_buffer(
            frames_dir.join(format!("{index}.png")),
            resized.as_raw(),
            2560,
            1440,
            image::ColorType::Rgb8,
        )?;
        png_io += start.elapsed();
        println!("processed_frame={index}");
    }

    // ArchiveOptions deletes an existing target. Encode only inside our TempDir,
    // then atomically publish with hard_link, which fails if the requested path exists.
    let encoded = work.path().join("encoded.mov");
    let start = Instant::now();
    ArchiveOptions::new(
        work.path().to_path_buf(),
        encoded.clone(),
        fps,
        1.0,
        "pcm_s16le".into(),
        true,
    )
    .process()?;
    ensure_new(output)?;
    fs::hard_link(&encoded, output).with_context(|| {
        format!(
            "publish to {} without overwrite (same filesystem required)",
            output.display()
        )
    })?;
    let archive = start.elapsed();
    let bytes = fs::metadata(output)?.len();
    let command_to_file = command_start.elapsed();
    println!(
        "frames={} output_bytes={} extract_ms={:.3} session_load_ms={:.3} preprocess_ms={:.3} infer_sync_readback_ms={:.3} postprocess_ms={:.3} png_io_ms={:.3} archive_publish_ms={:.3} command_to_file_ms={:.3}",
        paths.len(),
        bytes,
        extract.as_secs_f64() * 1000.0,
        session_load.as_secs_f64() * 1000.0,
        preprocess.as_secs_f64() * 1000.0,
        inference.as_secs_f64() * 1000.0,
        postprocess.as_secs_f64() * 1000.0,
        png_io.as_secs_f64() * 1000.0,
        archive.as_secs_f64() * 1000.0,
        command_to_file.as_secs_f64() * 1000.0
    );
    Ok(())
}
