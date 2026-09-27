# Apple M3 Max upscale inference: preliminary run (2026-09-26)

The scope was subsequently expanded by the user's 2026-09-26 directive to include native non-ONNX Rust implementations and Rust `tch` (not Python), with a static HTML deliverable. See `report.html` and the independent `../native-candle/` and `../native-tch/` crates for the added evidence. This original log covers only the ONNX/imported candidates and the ORT baseline.

This is a partial experiment, **not** a backend migration recommendation or a completed 1.20x acceptance test. Scope and criteria are defined only by `.kilo/plans/1790410518571-rust-upscale-inference-benchmark.md`. Original frame inference observations (23 measured frames per configuration) are in `results.json`; there are no invented results for missing full-video runs.

## Frozen inputs and environment

- Repository: `5e15965a91f40f333c317f76b10f9c4dc58c7d42`; root `Cargo.lock` SHA-256 `1a46f5311910fb24024b7e6b7602e732c079b5b614ea6ad3441aef343a01d381` before running. Root `ort` remains `2.0.0-rc.13`; independent candidate crates have separate lockfiles (`tract 0.23.8`, `candle-onnx 0.11.0`, `burn-onnx 0.21.0`).
- Apple M3 Max, 128 GiB memory, macOS 26.7 (25G229), rustc 1.98.1, FFmpeg 9.0.2 with libx265 4.3. AC power attached, battery 79% at setup; temperature was not logged. Available disk at setup: 995 GiB. Build: `cargo build --release --locked`. Real-ESRGAN source license: BSD-3-Clause (the URL printed by `find_upscale_model`); no model or footage uploaded or committed.
- Model: `$HOME/.cache/pixworker/models/upscale/realesr-animevideov3_fp32.onnx`, SHA-256 `0a25cdd3cdfded5566850fe9d9dcb5e08335fbf2e7bb0da73adbb12b020c7ba2`. Local cache was present; no interactive download. Original `tmp/test.mkv` SHA-256 `07c52a2e7ccf3f4b6c6ddc29583453c540df5c0d48db72c12a4b5ede2086dbb3` (49.257 s, 1280x720, 24000/1001 fps, AAC audio). The available 1439.914 s long video was not sampled for this preliminary run; its SHA-256 was `715fe16d58e67f98e0a91fc31029e9db3376787d158afc9da0e3c45895ffc9b6`.
- `tmp/upscale-bench-20260926-1728/short.mkv` is a stream-copy of the first approximately 5 seconds of `tmp/test.mkv`: SHA-256 `04eb5ad906a1abd80c9babe82193166c708484956d5f91d1c6350be0edec0287`, 5.130 s, 122 decoded source video frames. The 24-frame continuous PNG sequence was extracted from it, with `0.png` reserved for warmup and frames 1..23 measured. `frames/0.png` SHA-256 `90e0fad13e29e3874a2437ba163d07482622b24715e8f642c501c98dfb532634`; ORT CPU `[1,3,2880,5120]` FP32 reference tensor SHA-256 `cf5859d9bbfcd4c0ccdf748fcd153f2180f34fba4fdf0a273c1b8d84927b3a0a`. All intermediate assets are under ignored `tmp/`.

## Short complete-video baseline

Command, once per configuration (no ORT profiler):

```sh
PIXWORKER_TIMING=1 /usr/bin/time -l target/release/pixworker enhance --input tmp/upscale-bench-20260926-1728/short.mkv --output tmp/upscale-bench-20260926-1728/short-ort-default.mov --upscale 2.0 --upscale-model realesr-animevideov3 --vfi 1.0 --silent
PIXWORKER_TIMING=1 PIXWORKER_CPU_ONLY=1 /usr/bin/time -l target/release/pixworker enhance --input tmp/upscale-bench-20260926-1728/short.mkv --output tmp/upscale-bench-20260926-1728/short-ort-cpu.mov --upscale 2.0 --upscale-model realesr-animevideov3 --vfi 1.0 --silent
```

Raw elapsed seconds, default / CPU-only: total `145.55 / 222.36`; extract `1.396906 / 1.392480`; upscale session `0.249766 / 0.011905`; upscale entire stage `125.243700 / 202.595899`; within upscale, preprocess `0.165376 / 0.166170`, `Session::run` `90.960950 / 168.969300`, postprocess including Lanczos3 `29.531619 / 29.174608`, PNG I/O `4.326124 / 4.272713`; archive `18.767601 / 18.301821`. Maximum resident set: `10,425,090,048 / 2,213,593,088` bytes. The default run's `(session + upscale)/total` is about `0.862`, above the plan's `1/6` early-exit threshold. These are one run each, **not** the best tuned ORT baseline or repeated paired measurements.

The outputs are both 2560x1440, 24000/1001 fps, 5.130125 s, with PCM 44.1 kHz stereo audio, but both contain **123 frames versus 122 source frames**: existing FFmpeg extraction duplicated one frame (its log says `dup=1`). This fails the strict source-frame-count acceptance criterion until the sample/extraction behavior is resolved without changing the comparison pipeline. Default vs ORT CPU decoded-frame SSIM `All=0.999787`; sampled PSNR in the FFmpeg per-frame log ranged roughly 65.47-67.94 dB, not a candidate quality certificate or regional PSNR/SSIM check.

## Compatibility and 24-frame preliminary ranking

Every candidate imported the same local FP32 ONNX graph or its same-weight conversion and produced `[1,3,2880,5120]` from RGB/NCHW FP32 input. Versus ORT CPU reference on frame 0, full raw-float PSNR in the 0..1 range: Tract CPU/Metal `145.51 dB` (peak abs `2.33e-6`); Candle CPU `143.39 dB`, Metal `144.96 dB` (peaks `2.38e-6`, `3.43e-6`); Burn Metal `143.05 dB` (peak `2.62e-6`). These do not replace final decoded RGB-frame checks on multiple source frames and regions.

Each of the following is a median of **23 serial measured frames after one warmup frame**; the 23 observed per-frame `Session::run`/candidate inference-and-readback milliseconds are in `results.json` (min/max can be computed from each array). Higher fps is faster. Tract uses one `State` for the entire sequence; its initial per-frame `Runnable::run` measurements were discarded because that API spawns a new state each call. This is a short, ordered, non-interleaved sample; it is not a sustained 3-5 minute ranking with uncertainty estimates.

- ORT default CoreML/CPU: `732.838 ms`, `1.365 fps` (mixed placement: CPU `Resize` and `DepthToSpace`, CoreML for two other subgraphs, per ORT profile).
- ORT CoreML with CPU+ANE option: `759.904 ms`, `1.316 fps` (option requested, physical ANE usage not independently proven).
- ORT CoreML with CPU+GPU option: `763.904 ms`, `1.309 fps` (option requested, physical GPU usage not independently proven).
- Burn ONNX + WGPU/Metal: `834.861 ms`, `1.198 fps`; adapter reports Apple M3 Max Metal, but per-operator placement is not separately profiled. Build-time code generation is excluded from inference time and must be assessed as deployment cost.
- ORT CPU-only: `1683.635 ms`, `0.594 fps`.
- Candle ONNX + Metal: `2794.465 ms`, `0.358 fps`; output is Metal, FP32 initializers are explicitly preloaded there, shape tensors remain CPU. Missing `DepthToSpace` CRD/blocksize=4 is expanded to existing `Reshape/Transpose/Reshape` operations for 720p in the isolated prototype.
- Tract ONNX + Metal: `3496.665 ms`, `0.286 fps`; its diagnostic profile contains 18 `MetalConv` nodes, Metal elementwise/axis operations, and one upload and one readback, with no CPU compute node shown.
- Candle ONNX CPU: `4382.674 ms`, `0.228 fps` (same DepthToSpace expansion).
- Tract ONNX CPU: `26795.907 ms`, `0.037 fps` (observed range `23908.673..37472.373 ms`; temperature/background load not controlled, so do not attribute its variation to the state change).

Profiling was **not** enabled for these frame timings. Some frame tools include host readback in inference time; full-frame preprocessing and candidate adapter times differ. This sample supports only a preliminary ranking, not a 20% complete-video result. The only ORT CoreML option experiments were default, CPU+GPU, CPU+ANE, and one cold MLProgram probe; the latter printed unbounded-dimension exceptions and still used two CoreML/CPU subgraphs, so it is not a validated optimized baseline. ORT CPU thread tuning remains undone.

## Reproduction and remaining acceptance work

```sh
RUN_DIR=$(mktemp -d tmp/upscale-bench-XXXXXX)
mkdir "$RUN_DIR/frames"
ffmpeg -v error -nostdin -i tmp/test.mkv -t 5 -map 0:v:0 -map 0:a:0 -c copy -n "$RUN_DIR/short.mkv"
ffmpeg -v error -nostdin -i "$RUN_DIR/short.mkv" -frames:v 24 -an -start_number 0 -n "$RUN_DIR/frames/%d.png"
cargo build --release --locked --example upscale_profile
target/release/examples/upscale_profile "$RUN_DIR/frames" "$RUN_DIR/no-profile"
cargo build --release --locked --manifest-path benches/upscale-inference/Cargo.toml
benches/upscale-inference/target/release/upscale-inference "$HOME/.cache/pixworker/models/upscale/realesr-animevideov3_fp32.onnx" "$RUN_DIR/frames" metal
cargo build --release --locked --manifest-path benches/candle-inference/Cargo.toml
benches/candle-inference/target/release/candle-inference "$HOME/.cache/pixworker/models/upscale/realesr-animevideov3_fp32.onnx" "$RUN_DIR/frames" metal
CARGO_HOME="$PWD/benches/burn-inference/.cargo" CARGO_TARGET_DIR="$PWD/benches/burn-inference/target" cargo build --release --locked --manifest-path benches/burn-inference/Cargo.toml
benches/burn-inference/target/release/burn-inference "$RUN_DIR/frames"
```

Still required by the plan: select and hash ~5 s and continuous 3-5 minute clips **from the long video** covering static/motion/detail; resolve or account for the source/output frame count mismatch; compare first/middle/last decoded RGB frames and edge/texture regions against ORT CPU with predeclared PSNR >=45 dB and SSIM >=0.995; test all plausible low-risk ORT configurations against the same quality/resource limits; store complete raw per-run preprocessing, transfer, PNG, encoding and memory metrics; run at least five interleaved paired complete-video repetitions, increase samples near 1.20, and repeat the winning comparison on the whole 24-minute video. No 1.20x lower-bound result or long-run reliability result exists. **No backend migration decision is warranted from this preliminary run.**
