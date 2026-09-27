use anyhow::{Result, bail, ensure};
use native_tch::{OUTPUT_SHAPE, check_reference, infer, load_frame, load_model};
use std::{env, fs, path::Path, time::Instant};
use tch::Device;

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 {
        bail!(
            "usage: native-tch <model.safetensors> <frames-dir> <ort-cpu-frame0.f32> <cpu|mps|cuda>"
        );
    }
    let device = match args[4].as_str() {
        "cpu" => Device::Cpu,
        "mps" => Device::Mps,
        "cuda" => {
            ensure!(tch::Cuda::is_available(), "CUDA unavailable");
            Device::Cuda(0)
        }
        other => bail!("unknown device: {other}"),
    };
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);
    println!(
        "device={device:?} intra_op_threads={} inter_op_threads={} OMP_NUM_THREADS={} MKL_NUM_THREADS={} PYTORCH_ENABLE_MPS_FALLBACK={}",
        tch::get_num_threads(),
        tch::get_num_interop_threads(),
        env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        env::var("MKL_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        env::var("PYTORCH_ENABLE_MPS_FALLBACK").unwrap_or_else(|_| "unset".into())
    );
    let _no_grad = tch::no_grad_guard();
    let layers = load_model(Path::new(&args[1]), device)?;
    let frames = Path::new(&args[2]);
    let mut readback = vec![0.0_f32; OUTPUT_SHAPE.iter().product::<i64>() as usize];
    if frames.is_file() {
        infer(&layers, load_frame(frames, device)?, &mut readback)?;
        check_reference(&readback, Path::new(&args[3]))?;
        return Ok(());
    }
    let entries = fs::read_dir(frames)?.collect::<std::io::Result<Vec<_>>>()?;
    let count = entries
        .iter()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "png"))
        .count();
    if count < 8 || (0..count).any(|index| !frames.join(format!("{index}.png")).is_file()) {
        bail!("expected at least 8 consecutive frames named 0.png..N.png");
    }
    let warmup = frames.join("0.png");
    let input = load_frame(&warmup, device)?;
    let started = Instant::now();
    infer(&layers, input, &mut readback)?;
    println!(
        "warmup_frame=0 inference_sync_readback_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    check_reference(&readback, Path::new(&args[3]))?;
    println!("frame,preprocess_ms,inference_sync_readback_ms,total_ms");
    for index in 1..count {
        let frame_start = Instant::now();
        let input = load_frame(&frames.join(format!("{index}.png")), device)?;
        let inference_start = Instant::now();
        infer(&layers, input, &mut readback)?;
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{index},{:.3},{inference_ms:.3},{:.3}",
            (inference_start - frame_start).as_secs_f64() * 1000.0,
            frame_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}
