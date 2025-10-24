## Multi-platform cross-build Makefile for pixworker
# Builds release binaries for macOS/Linux/Windows (x86_64 + arm64)
# Usage:
#   make dist           # build all configured targets and collect artifacts in ./dist
#   make add-targets    # add required rustup targets
#   make build-targets  # build all targets
#   make clean          # cargo clean

SHELL := /bin/bash

# Detect host OS to determine available targets
UNAME_S := $(shell uname -s)

# Target triples to build. Windows MSVC targets require Windows host due to ring/reqwest dependencies.
ifeq ($(UNAME_S),Windows_NT)
    # Building on Windows - include all targets
    TARGETS := \
        x86_64-apple-darwin \
        aarch64-apple-darwin \
        x86_64-unknown-linux-gnu \
        aarch64-unknown-linux-gnu \
        x86_64-pc-windows-msvc \
        aarch64-pc-windows-msvc
else ifeq ($(UNAME_S),Darwin)
    # Building on macOS - exclude Windows targets (ring crate requires native MSVC)
    TARGETS := \
        x86_64-apple-darwin \
        aarch64-apple-darwin \
        x86_64-unknown-linux-gnu \
        aarch64-unknown-linux-gnu
else
    # Building on Linux - exclude Windows targets (ring crate requires native MSVC)
    TARGETS := \
        x86_64-apple-darwin \
        aarch64-apple-darwin \
        x86_64-unknown-linux-gnu \
        aarch64-unknown-linux-gnu
endif

BIN_NAME := pixworker

# Individual target phony declarations
.PHONY: all dist add-targets build-targets collect clean help
.PHONY: macos-x64 macos-arm64 linux-x64 linux-arm64 windows-x64 windows-arm64

all: build-targets

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
	@echo "  make collect        - Collect built binaries into ./dist"
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

## Add rustup targets (best-effort; some targets may already be installed)
add-targets:
	@echo "==> Ensuring rustup targets are installed"
	@for t in $(TARGETS); do \
		printf "Adding $$t... "; \
		rustup target add $$t >/dev/null 2>&1 || true; \
		echo "done"; \
	done

## Build all targets (best-effort). Note: cross-compiling to MSVC from non-Windows hosts usually fails
## unless a cross-compiler / toolchain is available; failures are reported but build continues.
build-targets: add-targets
	@echo "==> Building targets (this will take a while)"
	@for t in $(TARGETS); do \
		echo "-- Building for $$t"; \
		if cargo build --release --target $$t; then \
			echo "   ok: $$t"; \
		else \
			echo "   FAILED: $$t (see cargo output)"; \
		fi; \
	done

## Collect built binaries into dist/<triple>/
collect:
	@echo "==> Collecting binaries into ./dist"
	@rm -rf dist
	@mkdir -p dist
	@for t in $(TARGETS); do \
		case $$t in \
		*windows*) ext=.exe ;; \
		*) ext= ;; \
		esac; \
		src=target/$$t/release/$(BIN_NAME)$${ext}; \
		if [ -f "$$src" ]; then \
			mkdir -p dist/$$t; \
			cp "$$src" dist/$$t/ || true; \
			echo "  copied: $$src -> dist/$$t/"; \
		else \
			echo "  missing: $$src"; \
		fi; \
	done

## Full pipeline: add targets, build, collect
dist: build-targets collect

## Individual target builds
macos-x64:
	@echo "==> Building for macOS x86_64"
	@rustup target add x86_64-apple-darwin >/dev/null 2>&1 || true
	@cargo build --release --target x86_64-apple-darwin
	@echo "✓ Binary: target/x86_64-apple-darwin/release/$(BIN_NAME)"

macos-arm64:
	@echo "==> Building for macOS ARM64"
	@rustup target add aarch64-apple-darwin >/dev/null 2>&1 || true
	@cargo build --release --target aarch64-apple-darwin
	@echo "✓ Binary: target/aarch64-apple-darwin/release/$(BIN_NAME)"

linux-x64:
	@echo "==> Building for Linux x86_64"
	@rustup target add x86_64-unknown-linux-gnu >/dev/null 2>&1 || true
	@cargo build --release --target x86_64-unknown-linux-gnu
	@echo "✓ Binary: target/x86_64-unknown-linux-gnu/release/$(BIN_NAME)"

linux-arm64:
	@echo "==> Building for Linux ARM64"
	@rustup target add aarch64-unknown-linux-gnu >/dev/null 2>&1 || true
	@cargo build --release --target aarch64-unknown-linux-gnu
	@echo "✓ Binary: target/aarch64-unknown-linux-gnu/release/$(BIN_NAME)"

windows-x64:
	@echo "==> Building for Windows x86_64"
	@rustup target add x86_64-pc-windows-msvc >/dev/null 2>&1 || true
	@cargo build --release --target x86_64-pc-windows-msvc
	@echo "✓ Binary: target/x86_64-pc-windows-msvc/release/$(BIN_NAME).exe"

windows-arm64:
	@echo "==> Building for Windows ARM64"
	@rustup target add aarch64-pc-windows-msvc >/dev/null 2>&1 || true
	@cargo build --release --target aarch64-pc-windows-msvc
	@echo "✓ Binary: target/aarch64-pc-windows-msvc/release/$(BIN_NAME).exe"

clean:
	@echo "Cleaning cargo artifacts"
	@cargo clean
	@rm -rf dist
	@echo "Done"