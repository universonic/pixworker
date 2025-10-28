use crate::proto::onnx;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Inspect an ONNX model file and print inputs, outputs, metadata and operator usage.
pub fn inspect<P: AsRef<Path>>(path: P, graphviz: bool, list: bool) -> Result<()> {
    let path = path.as_ref();
    let buf =
        fs::read(path).with_context(|| format!("failed to read ONNX model: {}", path.display()))?;

    // Decode protobuf using the prost-generated types via UFCS (ensure prost::Message trait method is used)
    let model = <onnx::ModelProto as prost::Message>::decode(buf.as_slice())
        .context("failed to decode ONNX ModelProto")?;

    println!("Model: {}", path.display());
    if !model.producer_name.is_empty() {
        println!("Producer: {}", model.producer_name);
    }
    if !model.domain.is_empty() {
        println!("Domain: {}", model.domain);
    }
    if model.model_version != 0 {
        println!("Model version: {}", model.model_version);
    }

    // doc_string is a plain string in the generated proto; print if non-empty
    if !model.doc_string.is_empty() {
        println!("Doc: {}", model.doc_string);
    }

    // Metadata props
    if !model.metadata_props.is_empty() {
        println!("Metadata:");
        for p in model.metadata_props.iter() {
            println!("  {}: {}", p.key, p.value);
        }
    }

    let graph = match model.graph {
        Some(g) => g,
        None => {
            println!("(no graph found)");
            return Ok(());
        }
    };

    // If the user requested a raw listing, print each node's name, op_type and attributes
    if list {
        println!("Graph: {}", graph.name);
        println!("Nodes (name: op_type, attributes):");
        for (idx, node) in graph.node.iter().enumerate() {
            let node_name = if node.name.is_empty() { format!("<unnamed_{}>", idx) } else { node.name.clone() };
            println!("- {}: {}", node_name, node.op_type);

            if node.attribute.is_empty() {
                println!("  (no attributes)");
                continue;
            }

            for attr in node.attribute.iter() {
                // Collect readable representation of common AttributeProto fields
                let mut pieces: Vec<String> = Vec::new();

                if !attr.s.is_empty() {
                    pieces.push(format!("s=\"{}\"", String::from_utf8_lossy(&attr.s)));
                }
                if attr.f != 0.0 {
                    pieces.push(format!("f={}", attr.f));
                }
                // Note: i may legitimately be zero in some models; still useful to print
                pieces.push(format!("i={}", attr.i));
                if !attr.ints.is_empty() {
                    pieces.push(format!("ints={:?}", attr.ints));
                }
                if !attr.floats.is_empty() {
                    pieces.push(format!("floats={:?}", attr.floats));
                }
                if !attr.strings.is_empty() {
                    let strs: Vec<String> = attr
                        .strings
                        .iter()
                        .map(|b| String::from_utf8_lossy(b).to_string())
                        .collect();
                    pieces.push(format!("strings={:?}", strs));
                }
                if let Some(t) = attr.t.as_ref() {
                    pieces.push(format!("tensor(dtype={}, dims={:?})", t.data_type, t.dims));
                }

                if pieces.is_empty() {
                    // Fallback: print attribute name and raw type id
                    println!("  - {} (raw type={})", attr.name, attr.r#type);
                } else {
                    println!("  - {}: {}", attr.name, pieces.join(", "));
                }
            }
        }

        return Ok(());
    }

    println!("Graph: {}", graph.name);

    // Inputs
    println!("Inputs:");
    for input in graph.input.iter() {
        let name = &input.name;
        let ty = type_str_from_value_info(input);
        println!("  - {}: {}", name, ty);
    }

    // Outputs
    println!("Outputs:");
    for output in graph.output.iter() {
        let name = &output.name;
        let ty = type_str_from_value_info(output);
        println!("  - {}: {}", name, ty);
    }

    // Operators used
    let mut op_count: HashMap<String, usize> = HashMap::new();
    for node in graph.node.iter() {
        let op = node.op_type.clone();
        *op_count.entry(op).or_default() += 1;
    }

    println!("Operators (type: count):");
    let mut ops: Vec<_> = op_count.into_iter().collect();
    ops.sort_by(|a, b| b.1.cmp(&a.1));
    for (op, cnt) in ops.iter() {
        println!("  {}: {}", op, cnt);
    }

    if graphviz {
        let dot = build_graphviz(&graph);
        // Warn if DOT is large before writing
        let dot_size = dot.len();
        const WARN_THRESHOLD: usize = 5 * 1024 * 1024; // 5 MiB
        if dot_size > WARN_THRESHOLD {
            eprintln!(
                "Warning: generated Graphviz DOT is too large ({} > {}). Writing and opening it may be slow or consume a lot of memory.",
                human(dot_size), human(WARN_THRESHOLD)
            );
        }

        let out = path.with_extension("dot");
        let mut f = File::create(&out)
            .with_context(|| format!("failed to create dot file: {}", out.display()))?;
        f.write_all(dot.as_bytes())?;
        println!(
            "\nWrote Graphviz DOT to: {} ({} bytes)",
            out.display(),
            dot_size
        );
        println!(
            "You can visualize it using Graphviz tools, e.g.:\n  dot -Tpng {} -o {}",
            out.display(),
            path.with_extension("png").display()
        );
    }

    Ok(())
}

fn type_str_from_value_info(vi: &onnx::ValueInfoProto) -> String {
    // Try to extract tensor element type and shape if present
    if let Some(typ) = vi.r#type.as_ref() {
        // New ONNX proto encodes the concrete kind under the `value` oneof
        if let Some(onnx::type_proto::Value::TensorType(tensor)) = typ.value.as_ref() {
            let elem = tensor.elem_type;
            let shape = tensor.shape.as_ref().map(|s| {
                s.dim
                    .iter()
                    .map(|d| match d.value.as_ref() {
                        Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                            v.to_string()
                        }
                        Some(onnx::tensor_shape_proto::dimension::Value::DimParam(p)) => p.clone(),
                        _ => "?".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            });

            let elem_s = elem_to_str(elem);
            if let Some(s) = shape {
                return format!("{}[{}]", elem_s, s);
            } else {
                return elem_s;
            }
        }
    }
    "<unknown>".to_string()
}

fn elem_to_str(elem: i32) -> String {
    let dt = onnx::tensor_proto::DataType::try_from(elem).ok();
    // ONNX TensorProto DataType enum values
    match dt {
        Some(onnx::tensor_proto::DataType::Undefined) => "UNSPECIFIED".to_string(),
        Some(onnx::tensor_proto::DataType::Float) => "float32".to_string(),
        Some(onnx::tensor_proto::DataType::Uint8) => "uint8".to_string(),
        Some(onnx::tensor_proto::DataType::Int8) => "int8".to_string(),
        Some(onnx::tensor_proto::DataType::Uint16) => "uint16".to_string(),
        Some(onnx::tensor_proto::DataType::Int16) => "int16".to_string(),
        Some(onnx::tensor_proto::DataType::Int32) => "int32".to_string(),
        Some(onnx::tensor_proto::DataType::Int64) => "int64".to_string(),
        Some(onnx::tensor_proto::DataType::String) => "string".to_string(),
        Some(onnx::tensor_proto::DataType::Bool) => "bool".to_string(),
        Some(onnx::tensor_proto::DataType::Float16) => "float16".to_string(),
        Some(onnx::tensor_proto::DataType::Double) => "float64".to_string(),
        Some(onnx::tensor_proto::DataType::Uint32) => "uint32".to_string(),
        Some(onnx::tensor_proto::DataType::Uint64) => "uint64".to_string(),
        Some(onnx::tensor_proto::DataType::Complex64) => "complex64".to_string(),
        Some(onnx::tensor_proto::DataType::Complex128) => "complex128".to_string(),
        Some(onnx::tensor_proto::DataType::Bfloat16) => "bfloat16".to_string(),
        _ => format!("dtype({})", elem),
    }
}

fn build_graphviz(graph: &onnx::GraphProto) -> String {
    let mut dot = String::new();
    dot.push_str(
        "digraph onnx_graph {\n  rankdir=LR;\n  node [shape=box,fontname=\"Helvetica\"];\n",
    );

    // create input nodes
    for input in graph.input.iter() {
        let nid = sanitize_name(&input.name);
        dot.push_str(&format!(
            "  \"{}\" [label=\"IN:{}\",shape=oval];\n",
            nid,
            escape_label(&input.name)
        ));
    }

    // nodes
    for (i, node) in graph.node.iter().enumerate() {
        let nid = format!("n{}", i);
        let label = if node.name.is_empty() {
            format!("{}", node.op_type)
        } else {
            format!("{}\\n{}", node.op_type, node.name)
        };
        dot.push_str(&format!(
            "  {} [label=\"{}\"];\n",
            nid,
            escape_label(&label)
        ));

        // edges from inputs -> node
        for inp in node.input.iter() {
            let src = sanitize_name(inp);
            dot.push_str(&format!("  \"{}\" -> {} ;\n", src, nid));
        }

        // edges from node -> outputs
        for out in node.output.iter() {
            let dst = sanitize_name(out);
            dot.push_str(&format!("  {} -> \"{}\" ;\n", nid, dst));
        }
    }

    // create output nodes
    for output in graph.output.iter() {
        let nid = sanitize_name(&output.name);
        dot.push_str(&format!(
            "  \"{}\" [label=\"OUT:{}\",shape=oval];\n",
            nid,
            escape_label(&output.name)
        ));
    }

    dot.push_str("}\n");
    dot
}

fn sanitize_name(s: &str) -> String {
    // Keep short names for DOT nodes; if empty, use placeholder
    if s.is_empty() {
        "\"<empty>\"".to_string()
    } else {
        s.replace('"', "_")
    }
}

fn escape_label(s: &str) -> String {
    s.replace('"', "\\\"")
}

// Format bytes to a human readable string, e.g. 1024 -> "1.00 KiB"
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