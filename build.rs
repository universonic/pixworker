use std::env;
use std::fs;
use std::io::{copy, Read, Cursor};
use std::path::Path;
use std::process::Command;
use anyhow::Result;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // HTTP GET
    let link = choose_download_url()?;
    println!("cargo:warning=downloading from: {}", &link);
    let mut resp = ureq::get(&link).call()?;
    if resp.status() != 200 {
        return Err(format!("failed to download archive: HTTP {}", resp.status()).into());
    }

    // Prepare destination directory: <manifest_dir>/target/{profile}
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let target = &env::var("TARGET").unwrap_or_default();
    let profile = env::var("PROFILE").unwrap_or_default();
    let out_dir = Path::new(&manifest_dir).join("target").join(target).join(&profile);
    fs::create_dir_all(&out_dir)?;
    unsafe {
        env::set_var("ORT_STRATEGY", "system");
        env::set_var("ORT_LIB_LOCATION", out_dir.display().to_string());
    }

    // Stream the response, gunzip and untar; copy only dynamic libraries to out_dir
    let mut reader = resp.body_mut().as_reader();
    let mut buf: Vec<u8> = Vec::new();
    reader.read_to_end(&mut buf)?;
    let cursor = Cursor::new(buf);

    if link.ends_with(".tgz") {
        let gz = flate2::read::GzDecoder::new(cursor);
        let mut archive = tar::Archive::new(gz);

        for entry in archive.entries()? {
            let mut entry = entry?;

            // Extract an owned file-name String in a short-lived scope so we don't
            // hold any immutable borrow of `entry` when we later need a mutable
            // borrow for copying the entry contents.
            let name_opt = {
                match entry.path() {
                    Ok(p) => p.file_name().and_then(|s| s.to_str()).map(|s| s.to_owned()),
                    Err(_) => None,
                }
            };

            if let Some(name) = name_opt {
                // Accept common dynamic library extensions for macOS/Linux
                if name.ends_with(".dylib") || name.ends_with(".so") {
                    let dest = out_dir.join(&name);

                    // for .dylibs on macOS, only copy dynamically linked shared libraries.
                    if target.contains("apple-darwin") {
                        // Verify the extracted file is actually a dynamic library on macOS
                        // by running the `file` command and looking for the expected string.
                        // If the check fails, remove the file and skip it.
                        let tmp_dest = out_dir.join(format!("{}.tmp", &name));
                        let mut out = fs::File::create(&tmp_dest)?;
                        // Now we can mutably borrow `entry` to read its contents.
                        copy(&mut entry, &mut out)?;

                        let mut keep = true;
                        match Command::new("file").arg(&tmp_dest).output() {
                            Ok(outp) => {
                                let stdout = String::from_utf8_lossy(&outp.stdout).to_lowercase();
                                if !stdout.contains("dynamically linked shared library") || !stdout.contains("mach-o") {
                                    println!("cargo:warning=skipping {}: not a dynamically linked shared library (file output: {})", name, stdout.trim());
                                    keep = false;
                                }
                            }
                            Err(e) => {
                                println!("cargo:warning=file check failed for {}: {} -- assuming OK", name, e);
                            }
                        }

                        if !keep {
                            let _ = std::fs::remove_file(&tmp_dest);
                            continue;
                        }

                        fs::rename(&tmp_dest, &dest)?;
                        println!("cargo:warning=extracted {} -> {}", name, dest.display());
                        continue;
                    }

                    if entry.header().entry_type().is_symlink() {
                        // create symlink at dest
                        if let Ok(target_path) = entry.link_name() {
                            if let Some(target_str) = target_path.unwrap().to_str() {
                                println!("cargo:warning=create symlink {} -> {}", &dest.display(), target_str);
                                #[cfg(unix)] {
                                    use std::os::unix::fs::symlink;
                                    let _ = symlink(target_str, &dest);
                                }
                                #[cfg(windows)] {
                                    use std::os::windows::fs::symlink_file;
                                    let _ = symlink_file(target_str, &dest);
                                }
                                #[cfg(not(any(unix, windows)))] {
                                    println!("cargo:error=symlinks not supported on this platform");
                                }
                            }
                        }
                        continue;
                    }
                    // If file already exists, overwrite it
                    let mut out = fs::File::create(&dest)?;
                    // Now we can mutably borrow `entry` to read its contents.
                    copy(&mut entry, &mut out)?;
                    println!("cargo:warning=extracted {} -> {}", name, dest.display());
                }
            }
        }
    } else {
        // zip::ZipArchive requires a Read + Seek. The HTTP response reader is
        // not seekable, so read the entire body into memory and open the
        // archive from a Cursor.
        let mut zip = zip::ZipArchive::new(cursor)?;
        for i in 0..zip.len() {
            let mut file = zip.by_index(i)?;
            let name = match file.enclosed_name() {
                Some(p) => p.file_name().and_then(|s| s.to_str()).map(|s| s.to_owned()),
                None => None,
            };
            if let Some(name) = name {
                if name.ends_with(".dll") {
                    let dest = out_dir.join(&name);
                    if file.is_symlink() {
                        // create symlink at dest
                        if let Some(target_path) = file.enclosed_name() {
                            #[cfg(windows)]
                            if let Some(target_str) = target_path.file_name().and_then(|s| s.to_str()) {
                                use std::os::windows::fs::symlink_file;
                                let _ = symlink_file(target_str, &dest);
                            }
                        }
                        continue;
                    }
                    if dest.exists() {
                        std::fs::remove_file(&dest)?;
                    }
                    let mut out = fs::File::create(&dest)?;
                    copy(&mut file, &mut out)?;
                    println!("cargo:warning=extracted {} -> {}", name, dest.display());
                }
            }
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    Ok(())
}

fn choose_download_url() -> Result<String> {
    // Allow override from environment (CI / manual test)
    if let Ok(override_url) = env::var("ORT_DOWNLOAD_URL") {
        if !override_url.is_empty() {
            return Ok(override_url);
        }
    }

    // Cargo provides these at build-script time for the *target* (not the host)
    let target = env::var("TARGET").unwrap_or_default(); // full triple e.g. aarch64-apple-darwin
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default(); // e.g. "macos", "linux", "windows"
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default(); // e.g. "aarch64", "x86_64"

    println!(
        "cargo:warning=build.rs: TARGET={} OS={} ARCH={}",
        target, target_os, target_arch
    );

    // Simple mapping: add cases you need
    match (target_os.as_str(), target_arch.as_str()) {
        ("macos", "aarch64") => {
            Ok("https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz".to_string())
        }
        ("macos", "x86_64") => {
            Ok("https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-x86_64-1.16.3.tgz".to_string())
        }
        ("linux", "x86_64") => {
            Ok("https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz".to_string())
        }
        ("linux", "aarch64") => {
            Ok("https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz".to_string())
        }
        ("windows", "x86_64") => {
            Ok("https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-win-x64-gpu-1.16.3.zip".to_string())
        }
        _ => anyhow::bail!("unsupported target platform: {} (set ORT_DOWNLOAD_URL to override)", target)
    }
}
