use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_onnx::onnx::{AttributeProto, NodeProto, TensorProto};
use std::{
    collections::{BTreeMap, HashMap},
    env,
    path::Path,
    time::Instant,
};

fn expand_depth_to_space(graph: &mut candle_onnx::onnx::GraphProto, h: i64, w: i64) -> Result<()> {
    let mut nodes = Vec::with_capacity(graph.node.len() + 2);
    for node in graph.node.drain(..) {
        if node.op_type != "DepthToSpace" {
            nodes.push(node);
            continue;
        }
        let block = node
            .attribute
            .iter()
            .find(|a| a.name == "blocksize")
            .context("missing blocksize")?
            .i;
        let mode = node
            .attribute
            .iter()
            .find(|a| a.name == "mode")
            .context("missing mode")?;
        if block != 4 || mode.s != b"CRD" || node.input.len() != 1 || node.output.len() != 1 {
            bail!("unsupported DepthToSpace configuration: {node:?}");
        }
        let producer = nodes
            .iter()
            .rev()
            .find(|n: &&NodeProto| n.output.contains(&node.input[0]))
            .context("DepthToSpace producer missing")?;
        if producer.op_type != "Conv" {
            bail!("DepthToSpace producer is not Conv");
        }
        let weight = graph
            .initializer
            .iter()
            .find(|t| t.name == producer.input[1])
            .context("DepthToSpace Conv weight missing")?;
        let channels = *weight.dims.first().context("Conv weight shape missing")?;
        if channels % (block * block) != 0 {
            bail!("DepthToSpace channels {channels} not divisible by blocksize squared");
        }
        let c = channels / (block * block);
        let shape1 = format!("{}:shape1", node.name);
        let shape2 = format!("{}:shape2", node.name);
        for (name, dims) in [
            (&shape1, vec![1, c, block, block, h, w]),
            (&shape2, vec![1, c, h * block, w * block]),
        ] {
            graph.initializer.push(TensorProto {
                name: name.clone(),
                data_type: 7,
                dims: vec![dims.len() as i64],
                int64_data: dims,
                ..Default::default()
            });
        }
        let intermediate = format!("{}:reshaped", node.name);
        let permuted = format!("{}:permuted", node.name);
        nodes.push(NodeProto {
            op_type: "Reshape".into(),
            input: vec![node.input[0].clone(), shape1],
            output: vec![intermediate.clone()],
            ..Default::default()
        });
        nodes.push(NodeProto {
            op_type: "Transpose".into(),
            input: vec![intermediate],
            output: vec![permuted.clone()],
            attribute: vec![AttributeProto {
                name: "perm".into(),
                r#type: 7,
                ints: vec![0, 1, 4, 2, 5, 3],
                ..Default::default()
            }],
            ..Default::default()
        });
        nodes.push(NodeProto {
            op_type: "Reshape".into(),
            input: vec![permuted, shape2],
            output: node.output,
            ..Default::default()
        });
        println!("expanded DepthToSpace CRD blocksize={block} with Reshape/Transpose/Reshape");
    }
    graph.node = nodes;
    Ok(())
}

fn load_input(path: &Path, device: &Device) -> Result<Tensor> {
    let img = image::open(path)?.to_rgb8();
    let (w, h) = img.dimensions();
    if (w, h) != (1280, 720) {
        bail!("expected 1280x720 PNG, got {w}x{h}");
    }
    let rgb = img.into_raw();
    let mut chw = Vec::with_capacity(rgb.len());
    for c in 0..3 {
        for i in 0..h as usize * w as usize {
            chw.push(rgb[3 * i + c] as f32 / 255.0);
        }
    }
    Ok(Tensor::from_vec(
        chw,
        (1, 3, h as usize, w as usize),
        device,
    )?)
}

fn run_frames(model_path: &str, dir: &Path, device: &Device, backend: &str) -> Result<()> {
    let mut frames = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "png") {
            let index = path
                .file_stem()
                .context("PNG has no stem")?
                .to_string_lossy()
                .parse::<usize>()
                .with_context(|| format!("non-numeric PNG frame: {}", path.display()))?;
            if path.file_name().unwrap() != format!("{index}.png").as_str() {
                bail!(
                    "expected canonical frame name {index}.png, got {}",
                    path.display()
                );
            }
            frames.push((index, path));
        }
    }
    if frames.is_empty() {
        bail!("no PNG frames in {}", dir.display());
    }
    frames.sort_unstable_by_key(|(index, _)| *index);
    for (expected, (index, _)) in frames.iter().enumerate() {
        if *index != expected {
            bail!("missing or duplicate frame {expected}.png");
        }
    }
    if frames.len() < 8 {
        bail!(
            "need 0.png for warmup and at least 7 subsequent frames, got {}",
            frames.len()
        );
    }

    let mut model = candle_onnx::read_file(model_path)?;
    let graph = model.graph.as_ref().context("model has no graph")?;
    let input_name = graph
        .input
        .first()
        .context("model has no input")?
        .name
        .clone();
    let output_name = graph
        .output
        .first()
        .context("model has no output")?
        .name
        .clone();
    expand_depth_to_space(model.graph.as_mut().unwrap(), 720, 1280)?;
    let mut weights = HashMap::new();
    for initializer in model.graph.as_mut().unwrap().initializer.drain(..) {
        let tensor = candle_onnx::eval::get_tensor(&initializer, &initializer.name)?;
        let tensor = if backend == "metal" && tensor.dtype() == DType::F32 {
            tensor.to_device(device)?
        } else {
            tensor
        };
        weights.insert(initializer.name, tensor);
    }
    println!(
        "parsed graph once; cached {} initializers; FP32 weights/input on {device:?}, shape tensors on CPU",
        weights.len()
    );
    for (index, path) in frames {
        let frame_start = Instant::now();
        let input = load_input(&path, device)?;
        let mut inputs = weights.clone();
        inputs.insert(input_name.clone(), input);
        let preprocess_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        let inference_start = Instant::now();
        let mut outputs = candle_onnx::simple_eval(&model, inputs)?;
        let output = outputs
            .remove(&output_name)
            .context("model output missing")?;
        let output_device = format!("{:?}", output.device());
        let output = output.to_device(&Device::Cpu)?;
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;
        if output.dims() != [1, 3, 2880, 5120] {
            bail!(
                "unexpected 4x output shape for frame {index}: {:?}",
                output.dims()
            );
        }
        if index == 0 {
            println!("warmup frame=0 output_device={output_device} (excluded from timings)");
        } else {
            println!(
                "frame={index} preprocess_adapter_ms={preprocess_ms:.3} inference_sync_readback_ms={inference_ms:.3} total_ms={:.3} output_device={output_device}",
                frame_start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let model_path = args.next().context("expected local ONNX model path")?;
    let frame_path = args.next().context("expected local 1280x720 PNG path")?;
    let backend = args.next().unwrap_or_else(|| "cpu".into());
    let reference_path = args.next();
    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        _ => bail!("device must be cpu or metal"),
    };
    if Path::new(&frame_path).is_dir() {
        if reference_path.is_some() {
            bail!("directory mode does not accept a reference or profile path");
        }
        return run_frames(&model_path, Path::new(&frame_path), &device, &backend);
    }

    let input = load_input(Path::new(&frame_path), &device)?;
    let started = Instant::now();
    let mut model = candle_onnx::read_file(&model_path)?;
    let graph = model.graph.as_ref().context("model has no graph")?;
    let mut ops = BTreeMap::new();
    for node in &graph.node {
        *ops.entry(node.op_type.as_str()).or_insert(0usize) += 1;
    }
    println!("model import: {:?}; ops: {ops:?}", started.elapsed());
    let input_name = graph
        .input
        .first()
        .context("model has no input")?
        .name
        .clone();
    let output_name = graph
        .output
        .first()
        .context("model has no output")?
        .name
        .clone();
    expand_depth_to_space(model.graph.as_mut().unwrap(), 720, 1280)?;
    let mut inputs = HashMap::from([(input_name, input)]);
    if backend == "metal" {
        for initializer in model.graph.as_mut().unwrap().initializer.drain(..) {
            let tensor = candle_onnx::eval::get_tensor(&initializer, &initializer.name)?;
            let tensor = if tensor.dtype() == DType::F32 {
                tensor.to_device(&device)?
            } else {
                tensor
            };
            inputs.insert(initializer.name, tensor);
        }
        println!(
            "preloaded {} initializers (FP32 weights on Metal)",
            inputs.len() - 1
        );
    }
    let started = Instant::now();
    let mut output = candle_onnx::simple_eval(&model, inputs)?;
    let output = output
        .remove(&output_name)
        .context("model output missing")?;
    println!("computed on: {:?}", output.device());
    let output = output.to_device(&Device::Cpu)?;
    println!(
        "{backend} first run including output readback: {:?}; output shape: {:?}",
        started.elapsed(),
        output.dims()
    );
    if output.dims() != [1, 3, 2880, 5120] {
        bail!("unexpected 4x output shape");
    }
    if let Some(path) = reference_path {
        let reference = std::fs::read(path)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;
        if reference.len() != values.len() * 4 {
            bail!(
                "reference tensor has wrong size: {} bytes for {} values",
                reference.len(),
                values.len()
            );
        }
        let (mut square_error, mut peak_error) = (0.0_f64, 0.0_f64);
        for (bytes, value) in reference.chunks_exact(4).zip(values) {
            let error = (f32::from_le_bytes(bytes.try_into()?) - value).abs() as f64;
            if !error.is_finite() {
                bail!("non-finite tensor difference");
            }
            square_error += error * error;
            peak_error = peak_error.max(error);
        }
        let mse = square_error / output.elem_count() as f64;
        println!(
            "ORT CPU reference: float mse={mse}, peak error={peak_error}, psnr (0..1)={} dB",
            -10.0 * mse.log10()
        );
    } else {
        println!(
            "output first value: {}",
            output.flatten_all()?.narrow(0, 0, 1)?.to_vec1::<f32>()?[0]
        );
    }
    Ok(())
}
