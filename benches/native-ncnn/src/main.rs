use anyhow::{Result, bail, ensure};
use native_ncnn::{OUTPUT_SHAPE, check_reference, infer, load_model, prepare_frame};
use std::{env, fs, path::Path, time::Instant};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 6 {
        bail!(
            "usage: native-ncnn <model.param> <model.bin> <frame.png|frames-dir> <ort-cpu-frame0.f32> <cpu|vulkan>"
        );
    }
    let vulkan = match args[5].as_str() {
        "cpu" => false,
        "vulkan" => true,
        other => bail!("unknown device: {other}"),
    };
    let model = load_model(Path::new(&args[1]), Path::new(&args[2]), vulkan)?;
    let frames = Path::new(&args[3]);
    let mut readback = vec![0.0_f32; OUTPUT_SHAPE.iter().product::<usize>()];
    if frames.is_file() {
        let image = image::open(frames)?.to_rgb8();
        let input = prepare_frame(&image)?;
        infer(&model, &input, &mut readback)?;
        check_reference(&readback, Path::new(&args[4]))?;
        return Ok(());
    }
    let entries = fs::read_dir(frames)?.collect::<std::io::Result<Vec<_>>>()?;
    let count = entries
        .iter()
        .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
        .count();
    ensure!(
        count >= 121 && (0..count).all(|i| frames.join(format!("{i}.png")).is_file()),
        "expected 121+ consecutive PNG frames"
    );
    let warmup = image::open(frames.join("0.png"))?.to_rgb8();
    let input = prepare_frame(&warmup)?;
    let start = Instant::now();
    infer(&model, &input, &mut readback)?;
    println!(
        "warmup_frame=0 total_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    check_reference(&readback, Path::new(&args[4]))?;
    println!("frame,preprocess_ms,inference_sync_readback_ms,total_ms");
    for i in 1..count {
        let start = Instant::now();
        let image = image::open(frames.join(format!("{i}.png")))?.to_rgb8();
        let input = prepare_frame(&image)?;
        let inference_start = Instant::now();
        infer(&model, &input, &mut readback)?;
        println!(
            "{i},{:.3},{:.3},{:.3}",
            (inference_start - start).as_secs_f64() * 1000.0,
            inference_start.elapsed().as_secs_f64() * 1000.0,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}
