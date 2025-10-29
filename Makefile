## Multi-platform cross-build Makefile for pixworker
# Builds release binaries for macOS/Linux/Windows (x86_64 + arm64)
# Usage:
#   make dist           # build all configured targets and collect artifacts in ./dist
#   make add-targets    # add required rustup targets
#   make build-targets  # build all targets
#   make clean          # cargo clean
MANIFEST_DIR := $(abspath $(lastword $(MAKEFILE_LIST)))

# Detect host OS to determine available targets
UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),)
    UNAME_S := $(OS)
endif

# Target triples to build. Windows MSVC targets require Windows host due to ring/reqwest dependencies.
ifeq ($(UNAME_S),Windows_NT)
    # Building on Windows - include all targets
    TARGETS := \
        windows-x64 \
        windows-arm64
else ifeq ($(UNAME_S),Darwin)
    # Building on macOS - exclude Windows targets (ring crate requires native MSVC)
    TARGETS := \
        macos-x64 \
        macos-arm64
else
    # Building on Linux - exclude Windows targets (ring crate requires native MSVC)
    TARGETS := \
        linux-x64 \
        linux-arm64
endif

BIN_NAME := pixworker

# Individual target phony declarations
.PHONY: all help clean
.PHONY: macos-x64 macos-arm64 linux-x64 linux-arm64 windows-x64 windows-arm64

all: $(TARGETS)

help:
	@echo "pixworker cross-build Makefile"
	@echo ""
	@echo "Detected host: $(UNAME_S)"
	@echo "Available targets: $(TARGETS)"
	@echo ""
	@echo "Main targets:"
	@echo "  make / make all     - Build all platform targets (default)"
	@echo "  make dist           - Build all targets and collect binaries into ./dist"
	@echo "  make add-targets    - Add rustup targets (may require network)"
	@echo "  make build-targets  - Build all configured targets (assumes targets installed)"
	@echo "  make clean          - cargo clean"
	@echo ""
	@echo "Individual platform targets:"
	@echo "  make macos-x64      - Build for x86_64-apple-darwin"
	@echo "  make macos-arm64    - Build for aarch64-apple-darwin"
	@echo "  make linux-x64      - Build for x86_64-unknown-linux-gnu"
	@echo "  make linux-arm64    - Build for aarch64-unknown-linux-gnu"
	@echo "  make windows-x64    - Build for x86_64-pc-windows-msvc (Windows host only)"
	@echo "  make windows-arm64  - Build for aarch64-pc-windows-msvc (Windows host only)"
	@echo ""
	@echo "Note: Windows targets require building on Windows due to ring/reqwest dependencies"
	@echo ""

## Build all targets (best-effort). Note: cross-compiling to MSVC from non-Windows hosts usually fails
## unless a cross-compiler / toolchain is available; failures are reported but build continues.
all:
	@echo "==> Building targets (this will take a while)"
	@for t in $(TARGETS); do \
		$(MAKE) $$t || exit 1; \
	done

## Individual target builds
macos-x64:
	@echo "==> Building for macOS x86_64"
	@rustup target add x86_64-apple-darwin >/dev/null 2>&1 || true
	@ORT_STRATEGY=system ORT_LIB_LOCATION=$(MANIFEST_DIR)/target/x86_64-apple-darwin/release cargo build --release --target x86_64-apple-darwin
	@echo "✓ Binary: target/x86_64-apple-darwin/release/$(BIN_NAME)"
	@echo "Copying dynamic libraries..."
	@mkdir -p dist/x86_64-apple-darwin
	@cp target/x86_64-apple-darwin/release/$(BIN_NAME) dist/x86_64-apple-darwin/
	@cp target/x86_64-apple-darwin/release/*.dylib dist/x86_64-apple-darwin/ 2>/dev/null || true
	@echo "✓ Package ready: dist/x86_64-apple-darwin/"

macos-arm64:
	@echo "==> Building for macOS ARM64"
	@rustup target add aarch64-apple-darwin >/dev/null 2>&1 || true
	@ORT_STRATEGY=system ORT_LIB_LOCATION=$(MANIFEST_DIR)/target/aarch64-apple-darwin/release cargo build --release --target aarch64-apple-darwin
	@echo "✓ Binary: target/aarch64-apple-darwin/release/$(BIN_NAME)"
	@echo "Copying dynamic libraries..."
	@mkdir -p dist/aarch64-apple-darwin
	@cp target/aarch64-apple-darwin/release/$(BIN_NAME) dist/aarch64-apple-darwin/
	@cp target/aarch64-apple-darwin/release/*.dylib dist/aarch64-apple-darwin/ 2>/dev/null || true
	@echo "✓ Package ready: dist/aarch64-apple-darwin/"

linux-x64:
	@echo "==> Building for Linux x86_64"
	@rustup target add x86_64-unknown-linux-gnu >/dev/null 2>&1 || true
	@ORT_STRATEGY=system ORT_LIB_LOCATION=$(MANIFEST_DIR)/target/x86_64-unknown-linux-gnu/release cargo build --release --target x86_64-unknown-linux-gnu
	@echo "✓ Binary: target/x86_64-unknown-linux-gnu/release/$(BIN_NAME)"
	@echo "Copying dynamic libraries..."
	@mkdir -p dist/x86_64-unknown-linux-gnu
	@cp target/x86_64-unknown-linux-gnu/release/$(BIN_NAME) dist/x86_64-unknown-linux-gnu/
	@cp target/x86_64-unknown-linux-gnu/release/*.so dist/x86_64-unknown-linux-gnu/ 2>/dev/null || true
	@echo "✓ Package ready: dist/x86_64-unknown-linux-gnu/"

linux-arm64:
	@echo "==> Building for Linux ARM64"
	@rustup target add aarch64-unknown-linux-gnu >/dev/null 2>&1 || true
	@ORT_STRATEGY=system ORT_LIB_LOCATION=$(MANIFEST_DIR)/target/aarch64-unknown-linux-gnu/release cargo build --release --target aarch64-unknown-linux-gnu
	@echo "✓ Binary: target/aarch64-unknown-linux-gnu/release/$(BIN_NAME)"
	@echo "Copying dynamic libraries..."
	@mkdir -p dist/aarch64-unknown-linux-gnu
	@cp target/aarch64-unknown-linux-gnu/release/$(BIN_NAME) dist/aarch64-unknown-linux-gnu/
	@cp target/aarch64-unknown-linux-gnu/release/*.so dist/aarch64-unknown-linux-gnu/ 2>/dev/null || true
	@echo "✓ Package ready: dist/aarch64-unknown-linux-gnu/"

windows-x64:
	@echo "==> Building for Windows x86_64"
	@rustup target add x86_64-pc-windows-msvc >/dev/null 2>&1 || true
	@ORT_STRATEGY=system ORT_LIB_LOCATION=$(MANIFEST_DIR)\target\x86_64-pc-windows-msvc\release cargo build --release --target x86_64-pc-windows-msvc
	@echo "✓ Binary: target\x86_64-pc-windows-msvc\release\$(BIN_NAME).exe"
	@echo "Copying dynamic libraries..."
	@mkdir dist\x86_64-pc-windows-msvc
	@cp target\x86_64-pc-windows-msvc\release\$(BIN_NAME).exe dist\x86_64-pc-windows-msvc
	@cp target\x86_64-pc-windows-msvc\release\*.dll dist\x86_64-pc-windows-msvc\ 2>/dev/null || true
	@echo "✓ Package ready: dist\x86_64-pc-windows-msvc\"

windows-arm64:
	@echo "==> Building for Windows ARM64"
	@rustup target add aarch64-pc-windows-msvc >/dev/null 2>&1 || true
	@ORT_STRATEGY=system ORT_LIB_LOCATION=$(MANIFEST_DIR)\target\aarch64-pc-windows-msvc\release cargo build --release --target aarch64-pc-windows-msvc
	@echo "✓ Binary: target\aarch64-pc-windows-msvc\release\$(BIN_NAME).exe"
	@echo "Copying dynamic libraries..."
	@mkdir -p dist\aarch64-pc-windows-msvc
	@cp target\aarch64-pc-windows-msvc\release\$(BIN_NAME).exe dist\aarch64-pc-windows-msvc
	@cp target\aarch64-pc-windows-msvc\release\*.dll dist\aarch64-pc-windows-msvc\ 2>/dev/null || true
	@echo "✓ Package ready: dist\aarch64-pc-windows-msvc\"

clean:
	@echo "Cleaning cargo artifacts"
	@cargo clean
	@rm -rf dist
	@echo "Done"