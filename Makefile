# Native release builds only. Set LIBTORCH to the matching libtorch 2.11.0 directory.
SHELL := /bin/bash

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
HOST_TRIPLE := aarch64-apple-darwin
endif
else ifeq ($(UNAME_S),Linux)
ifeq ($(UNAME_M),x86_64)
HOST_TRIPLE := x86_64-unknown-linux-gnu
endif
else ifneq ($(filter MINGW% MSYS% CYGWIN% Windows_NT,$(UNAME_S)),)
ifeq ($(UNAME_M),x86_64)
HOST_TRIPLE := x86_64-pc-windows-msvc
endif
endif

.PHONY: all help build-targets collect dist macos-arm64 linux-x64 windows-x64

all: build-targets

help:
	@printf '%s\n' 'Native host: $(UNAME_S) $(UNAME_M)' 'Target: $(HOST_TRIPLE)' \
		'make              - build the native release binary' \
		'make dist         - build and package the binary with libtorch libraries' \
		'make collect      - package an existing native release build' \
		'make macos-arm64  - build/package on macOS Apple Silicon' \
		'make linux-x64    - build/package on Linux x86_64' \
		'make windows-x64  - build/package on Windows x86_64 (MSYS2 Bash + MSVC)'

build-targets:
	@set -eu; \
	if [ -z '$(HOST_TRIPLE)' ] || { [ -n '$(EXPECTED_TRIPLE)' ] && [ '$(HOST_TRIPLE)' != '$(EXPECTED_TRIPLE)' ]; }; then \
		printf '%s\n' 'Unsupported host or non-native target: $(UNAME_S) $(UNAME_M)' >&2; exit 1; \
	fi; \
	if [ -z "$${LIBTORCH:-}" ] || [ ! -d "$$LIBTORCH/lib" ]; then \
		printf '%s\n' 'Set LIBTORCH to the matching libtorch 2.11.0 directory (with lib/).' >&2; exit 1; \
	fi; \
	if [ '$(HOST_TRIPLE)' = x86_64-pc-windows-msvc ] && [ ! -d "$${FFMPEG_DIR:-}" ]; then \
		printf '%s\n' 'Set FFMPEG_DIR to the FFmpeg 9 shared development package.' >&2; exit 1; \
	fi; \
	cargo build --release --locked --target '$(HOST_TRIPLE)'

collect:
	@set -eu; \
	if [ -z '$(HOST_TRIPLE)' ] || { [ -n '$(EXPECTED_TRIPLE)' ] && [ '$(HOST_TRIPLE)' != '$(EXPECTED_TRIPLE)' ]; }; then \
		printf '%s\n' 'Unsupported host or non-native target: $(UNAME_S) $(UNAME_M)' >&2; exit 1; \
	fi; \
	if [ -z "$${LIBTORCH:-}" ] || [ ! -d "$$LIBTORCH/lib" ]; then \
		printf '%s\n' 'Set LIBTORCH to the matching libtorch 2.11.0 directory (with lib/).' >&2; exit 1; \
	fi; \
	ext=; if [ '$(HOST_TRIPLE)' = x86_64-pc-windows-msvc ]; then ext=.exe; fi; \
	src='target/$(HOST_TRIPLE)/release/pixworker'$$ext; \
	if [ ! -f "$$src" ]; then printf 'Missing binary: %s\n' "$$src" >&2; exit 1; fi; \
	out='dist/$(HOST_TRIPLE)'; mkdir -p "$$out"; \
	copied=0; \
	for lib in "$$LIBTORCH"/lib/*.dylib "$$LIBTORCH"/lib/*.so* "$$LIBTORCH"/lib/*.dll; do \
		[ -f "$$lib" ] || continue; \
		cp -L "$$lib" "$$out/"; copied=1; \
	done; \
	if [ "$$copied" -ne 1 ]; then printf '%s\n' 'No dynamic libtorch libraries found in LIBTORCH/lib.' >&2; exit 1; fi; \
	if [ '$(HOST_TRIPLE)' = aarch64-apple-darwin ]; then \
		install_name_tool -change /opt/llvm-openmp/lib/libomp.dylib @loader_path/libomp.dylib "$$out/libtorch_cpu.dylib"; \
	fi; \
	cp "$$src" "$$out/"; \
	printf 'Package ready: %s/\n' "$$out"

dist: build-targets collect

macos-arm64:
	@$(MAKE) dist EXPECTED_TRIPLE=aarch64-apple-darwin

linux-x64:
	@$(MAKE) dist EXPECTED_TRIPLE=x86_64-unknown-linux-gnu

windows-x64:
	@$(MAKE) dist EXPECTED_TRIPLE=x86_64-pc-windows-msvc
