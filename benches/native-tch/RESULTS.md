# Native tch FP32 frame benchmark (2026-09-26)

This is a 24-frame compatibility and speed probe, not a complete-video migration test. Source: the same `realesr-animevideov3` FP32 weights, loaded by the Rust safetensors API into native tch 0.22.0 tensors (18 Conv2d, 17 PReLU, DepthToSpace 4x, nearest-neighbor 4x residual add). No TorchScript or Python runtime. The first frame is excluded from the 23 timed frames. All timings in `timings.csv` are single-process, single-session, ordered 1..23; `inference_sync_readback_ms` includes CPU readback of the full 4x FP32 output.

- libtorch: `/Users/universonic/Workspace/pytorch/torch` (2.9.0), build cache `USE_MPS:BOOL=1`; libraries read-only. Apple M3 Max, rustc/cargo 1.98.1.
- Model: `$HOME/.cache/pixworker/models/upscale/realesr-animevideov3_fp32.safetensors`, SHA-256 `fc71c784b8fbff34fcce550e7823bfe8e20cf805c5cce5c7c18cde0f4218a257`.
- Frame 0: `tmp/upscale-bench-20260926-1728/frames/0.png`, SHA-256 `90e0fad13e29e3874a2437ba163d07482622b24715e8f642c501c98dfb532634`. ORT CPU FP32 reference: `tmp/upscale-bench-20260926-1728/ort-cpu-frame0.f32`, SHA-256 `cf5859d9bbfcd4c0ccdf748fcd153f2180f34fba4fdf0a273c1b8d84927b3a0a` (little-endian NCHW).
- RGB 1280x720 -> NCHW float `/255` -> FP32 output `[1,3,2880,5120]`. Gate set before timing: PSNR (peak 1) >=100 dB and maximum absolute difference <=0.001; failure exits before printing any frame timings.
- Both runs: `tch::set_num_threads(1)` and `tch::set_num_interop_threads(1)` reported intra-op=1, inter-op=1; `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTORCH_ENABLE_MPS_FALLBACK=0`. The Rust loop runs frames serially. libtorch/MPS driver-internal workers are not claimed to be a single OS thread.

Build from `benches/native-tch/`:

```sh
LIBTORCH=/Users/universonic/Workspace/pytorch/torch cargo build --release --locked --offline
```

Run from the same directory, replacing `cpu` with `mps` for the second, separate run:

```sh
LIBTORCH=/Users/universonic/Workspace/pytorch/torch DYLD_LIBRARY_PATH=/Users/universonic/Workspace/pytorch/torch/lib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 ./target/release/native-tch "$HOME/.cache/pixworker/models/upscale/realesr-animevideov3_fp32.safetensors" ../../tmp/upscale-bench-20260926-1728/frames ../../tmp/upscale-bench-20260926-1728/ort-cpu-frame0.f32 cpu
```

CPU warmup frame 0 inference+readback: `5274.199 ms`; reference MSE `3.148717474025e-15`, max abs `2.861022949219e-6`, PSNR `145.019 dB`. 23-frame inference+readback median `3621.308 ms` (`0.276 fps`), range `2725.027..4741.374 ms`.

MPS warmup frame 0 inference+readback: `1987.423 ms`; reference MSE `2.038756130292e-15`, max abs `1.788139343262e-6`, PSNR `146.906 dB`. 23-frame inference+readback median `227.459 ms` (`4.396 fps`), range `129.487..588.978 ms`.

Both qualify for this preliminary **frame-level** comparison. The measurements in `timings.csv` are the original runs; moving the implementation to `src/lib.rs` did not change the FP32 operators. A later MPS regression run after the extraction still passed frame 0 with the identical `[1,3,2880,5120]`, MSE `2.038756130292e-15`, max abs `1.788139343262e-6` and PSNR `146.906 dB`. Do not replace the earlier frame median with that separate run's timings.

## One complete 5-second movie (MPS, not a migration decision)

`video` uses the same `pixworker::utils::ffmpeg::{ExtractOptions, ArchiveOptions, FFProbe}` and `NTSC` settings as `EnhanceOptions::process`: PNG extraction and 44.1 kHz stereo PCM audio; VFI=1 (no GIMM); same shared tch native FP32 model; output NCHW FP32 -> multiply by 255, clamp, truncate to u8 and interleave RGB HWC -> `DynamicImage::ImageRgb8(...).resize_exact(2560,1440,Lanczos3)` -> numbered PNG; lossless x265 main444-12 MOV + PCM via `ArchiveOptions`. It requires `PYTORCH_ENABLE_MPS_FALLBACK=0`. `tempfile::TempDir` is created inside this crate's ignored `target/`. Archive writes only to that unique temporary directory and publishes via `hard_link` (atomic no-overwrite; output and crate target must be on the same filesystem). Existing paths, including dangling symlinks, are rejected before extraction. The produced movie and raw FFmpeg quality logs remain under ignored `benches/native-tch/target/`.

Build and actual successful command (from `benches/native-tch/`, no Python/PyTorch scripts):

```sh
LIBTORCH=/Users/universonic/Workspace/pytorch/torch cargo build --release --locked --offline --bins
/usr/bin/time -l /usr/bin/env LIBTORCH=/Users/universonic/Workspace/pytorch/torch DYLD_LIBRARY_PATH=/Users/universonic/Workspace/pytorch/torch/lib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 ./target/release/video ../../tmp/upscale-bench-20260926-1728/short.mkv target/short-native-mps.mov /Users/universonic/.cache/pixworker/models/upscale/realesr-animevideov3_fp32.safetensors
```

The first attempted launch placed `DYLD_LIBRARY_PATH` before `/usr/bin/time`; macOS stripped it and `dyld` could not find `libtorch_cpu.dylib`. It exited before running `video` and created no output. The successful invocation above sets it in `env` **after** `time`. A second invocation with the same output path is rejected immediately with `refusing to overwrite existing output`.

- Input `short.mkv` SHA-256 `04eb5ad906a1abd80c9babe82193166c708484956d5f91d1c6350be0edec0287`. Output `target/short-native-mps.mov` SHA-256 `7c9151b8b5ec9f470ff3b763649c3c88d8518c887c653e66484b04472759873d`, `162450763` bytes. ORT CPU comparison file `../../tmp/upscale-bench-20260926-1728/short-ort-cpu.mov` SHA-256 `ffbc88823eb1ee7aced47e10f24239dfd9acb193115d23ab2668dfc81d4f56e4`.
- Raw elapsed: extract `1759.187 ms`; session+weight load `48.890 ms`; preprocess `525.743 ms`; inference including full MPS->CPU readback `15568.076 ms`; postprocess `31103.740 ms`; PNG read/write `3680.561 ms`; archive+atomic publish `26982.909 ms`; first Rust `main` instruction to completed output file `79724.331 ms` (unassigned setup/bookkeeping about 56 ms). External `/usr/bin/time -l`: `79.88 real`, `299.95 user`, `4.67 sys`, peak resident set `1736785920` bytes, reported peak memory footprint `2452522880` bytes. One cold new process, no warmup frame excluded, no concurrent pressure test. `tch` intra/inter-op=1, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, MPS fallback=0; production x265 uses its normal 16-thread pool.
- `ffprobe -v error -count_frames -show_entries stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration,sample_rate,channels -show_entries format=duration -of json target/short-native-mps.mov` (and the same for the ORT CPU movie): both have HEVC `yuv444p`, 2560x1440, `24000/1001` for both rates, video `nb_frames=nb_read_frames=123`, video/format duration `5.130125 s`; audio `pcm_s16le`, 44100 Hz, 2 channels, duration `5.002336 s`. The source MKV has only **122 decoded video frames**; the shared production `ExtractOptions` FFmpeg invocation reported `dup=1` and extracted 123. Both produced movies agree, but the plan's strict original-source frame-count acceptance remains unresolved.

All-frame quality against the ORT CPU movie (both decoded streams explicitly converted to planar RGB `gbrp`; n=123):

```sh
ffmpeg -hide_banner -nostats -i ../../tmp/upscale-bench-20260926-1728/short-ort-cpu.mov -i target/short-native-mps.mov -filter_complex '[0:v]format=gbrp[ref];[1:v]format=gbrp[cand];[ref][cand]psnr=stats_file=target/psnr-rgb.log' -an -f null -
ffmpeg -hide_banner -nostats -i ../../tmp/upscale-bench-20260926-1728/short-ort-cpu.mov -i target/short-native-mps.mov -filter_complex '[0:v]format=gbrp[ref];[1:v]format=gbrp[cand];[ref][cand]ssim=stats_file=target/ssim-rgb.log' -an -f null -
```

PSNR RGB average `92.657757 dB`, minimum frame `90.896482 dB`; SSIM RGB All `1.000000` (FFmpeg prints 6 decimals, dB form `63.627254`). Frame indices 0/61/122 correspond to per-frame log n=1/62/123: PSNR `93.57/93.53/93.34 dB`, SSIM All `1.000000/1.000000/1.000000`. For reproducible spatial checks on those same three decoded RGB frames, the exact filters after the two `-i` arguments were:

```sh
-filter_complex "[0:v]format=gbrp,select='eq(n,0)+eq(n,61)+eq(n,122)',crop=256:256:0:0[ref];[1:v]format=gbrp,select='eq(n,0)+eq(n,61)+eq(n,122)',crop=256:256:0:0[cand];[ref]split=2[refp][refs];[cand]split=2[candp][cands];[refp][candp]psnr=stats_file=target/psnr-edge-rgb.log[p];[refs][cands]ssim=stats_file=target/ssim-edge-rgb.log[s]" -map '[p]' -map '[s]' -an -f null -
-filter_complex "[0:v]format=gbrp,select='eq(n,0)+eq(n,61)+eq(n,122)',crop=512:512:1024:464[ref];[1:v]format=gbrp,select='eq(n,0)+eq(n,61)+eq(n,122)',crop=512:512:1024:464[cand];[ref]split=2[refp][refs];[cand]split=2[candp][cands];[refp][candp]psnr=stats_file=target/psnr-center-rgb.log[p];[refs][cands]ssim=stats_file=target/ssim-center-rgb.log[s]" -map '[p]' -map '[s]' -an -f null -
```

Edge top-left 256x256 PSNR for 0/61/122: `inf/inf/91.07 dB` (first two are byte-identical), SSIM All `1.000000/1.000000/0.999999`. Center 512x512 at `(1024,464)` PSNR `97.09/100.10/101.07 dB`, SSIM All `1.000000/1.000000/1.000000`. The central crop is a fixed region, not a claimed visually verified texture ROI; visual artifact inspection was not performed. Short-clip pixel metrics pass the plan's PSNR >=45 dB and SSIM >=0.995, but one short run is **not** the five paired main-segment runs or full 24-minute reliability check, and does not establish the 1.20x complete-video migration threshold.

## Second 5-second pair, reverse order

The same short input was run as `tch MPS -> default ORT` (new, distinct output paths). The successful command was the same as above with `target/short-native-mps-2.mov` as output. It took `69.96 s` external wall clock (`69818.348 ms` from Rust `main` to completed file, maximum resident set `1738588160` bytes): extract `1534.230 ms`, session+weight load `43.072 ms`, preprocess `542.091 ms`, inference+readback `15299.919 ms`, postprocess `29451.027 ms`, PNG I/O `3659.736 ms`, archive+publish `19238.795 ms`. SHA-256 `7c9151b8b5ec9f470ff3b763649c3c88d8518c887c653e66484b04472759873d`: byte-for-byte identical to the first tch movie. Both movie streams match the first run's 2560x1440, 123-frame 24000/1001 fps video and 44.1 kHz stereo PCM audio.

The default ORT command was `PIXWORKER_TIMING=1 /usr/bin/time -l target/release/pixworker enhance --input tmp/upscale-bench-20260926-1728/short.mkv --output tmp/upscale-bench-20260926-1728/short-ort-default-2.mov --upscale 2.0 --upscale-model realesr-animevideov3 --vfi 1.0 --silent`. It took `146.21 s` external wall clock, maximum resident set `10237984768` bytes; extract `1.484032167 s`, upscale session `264.915584 ms`, upscale stage `126.031222709 s` (preprocess `169.329060 ms`, `Session::run` `91.508072467 s`, postprocess `29.701198382 s`, PNG I/O `4.380613049 s`), archive `18.547968666 s`. Its video SHA-256 is `3e9784f2c8b51486caca5b5ffae22e4e67a0f4b44ef69b70e41f8122abdf8184`. The two source/output frame-count mismatch persists. Raw external pair totals are also in `../upscale-inference/short-pairs.csv`; the pairs are only 5 seconds and cannot be substituted for five interleaved long-segment measurements.

## Isolated serial frame recheck

The first `timings.csv` runs were launched while the native Candle prototype could also be measuring frames. They are retained as raw observations but are **not** used for the current native frame ranking. With no other inference process active, the same 24 frames were remeasured sequentially as MPS, native Candle Metal, tch CPU and native Candle CPU. The complete tch recheck is in `timings-serial.csv`; its first-frame MPS float PSNR remains `146.906 dB`. After warmup, tch MPS inference+readback median is `120.622 ms` (min `120.065`, max `123.639`), and tch CPU median is `2314.745 ms` (min `2270.427`, max `2547.569`). Native Candle serial results live in `../native-candle/metal-serial.csv` and `../native-candle/cpu-serial.csv`. The MPS median from the possibly contended first run (`227.459 ms`) must not be used as a stable throughput estimate; neither run substitutes for multi-minute interleaved testing.
