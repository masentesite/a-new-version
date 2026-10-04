#!/usr/bin/env bash
#
# GroundingDINO environment bootstrapper.
#
# Usage:
#   bash setup_cloud_uv.sh
#   bash setup_cloud_uv.sh --check
#   bash setup_cloud_uv.sh --force --jobs 8
#   bash setup_cloud_uv.sh --cuda-home /usr/local/cuda-12.8
#
# This script is driven by the files in this directory:
#   pyproject.toml
#   uv.lock
#
# The GroundingDINO source tree lives one level above this directory. The
# existing lock file expects an editable package at ./GroundingDINO, so this
# script creates a local symlink instead of rewriting the lock file.

set -Eeuo pipefail

PY_VER="3.10"
TORCH_CUDA_MAJOR="12"
MIN_UV_VERSION="0.12.10"

MODE="deploy"
FORCE=0
AUTO_CUDA=1
MAX_JOBS="${MAX_JOBS:-4}"
CUDA_HOME_ARG=""
MIRROR_TUNA="${MIRROR_TUNA:-https://pypi.tuna.tsinghua.edu.cn/simple}"
MIRROR_PYTHON="${MIRROR_PYTHON:-https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
AUTO_CUDA_DIR="${AUTO_CUDA_DIR:-}"

C_RED=$'\033[1;31m'
C_GREEN=$'\033[1;32m'
C_YELLOW=$'\033[1;33m'
C_CYAN=$'\033[1;36m'
C_RESET=$'\033[0m'

usage() {
  cat <<EOF
GroundingDINO uv environment setup

Options:
  --check             Only inspect the machine; do not install or modify.
  --force             Remove the existing venv before syncing.
  --jobs N            Parallel build jobs for GroundingDINO _C (default: $MAX_JOBS).
  --cuda-home DIR     Use this CUDA toolkit first.
  --no-auto-cuda      Do not download a CUDA 12 toolkit if none is found.
  --mirror-tuna URL   PyPI mirror used to install uv when uv is missing.
  --mirror-python URL uv managed-Python mirror.
  -h, --help          Show this help.

Environment overrides:
  DATA_DISK=/path     Cache and optional CUDA location (default: /root/autodl-tmp).
  AUTO_CUDA_DIR=/path Where to install the redist CUDA toolkit.
  SKIP_CUDA_CHECK=1   Skip CUDA toolkit checks before uv sync.
EOF
}

say() { printf '\n%s==> %s%s\n' "$C_CYAN" "$*" "$C_RESET"; }
ok() { printf '  %sOK%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf '  %s!!%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die() { printf '\n  %sERROR%s %s\n\n' "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check) MODE="check" ;;
    --force) FORCE=1 ;;
    --jobs)
      [ "$#" -ge 2 ] || die "--jobs needs a value"
      MAX_JOBS="$2"
      shift
      ;;
    --cuda-home)
      [ "$#" -ge 2 ] || die "--cuda-home needs a directory"
      CUDA_HOME_ARG="$2"
      shift
      ;;
    --no-auto-cuda) AUTO_CUDA=0 ;;
    --mirror-tuna)
      [ "$#" -ge 2 ] || die "--mirror-tuna needs a URL"
      MIRROR_TUNA="$2"
      shift
      ;;
    --mirror-python)
      [ "$#" -ge 2 ] || die "--mirror-python needs a URL"
      MIRROR_PYTHON="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "Unknown option: $1"
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CREATE_ENV_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$CREATE_ENV_DIR/.." && pwd)"
SOURCE_DIR="$PROJECT_ROOT/GroundingDINO"
LOCKED_SOURCE_LINK="$CREATE_ENV_DIR/GroundingDINO"
LOG="$CREATE_ENV_DIR/setup_uv.log"

cd "$CREATE_ENV_DIR"

run() {
  printf '  $'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

version_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

detect_gpu_arch() {
  local cap=""
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | tr -d ' ' \
    | grep -E '^[0-9]+[.][0-9]+$' \
    | head -n1 || true)"
  if [ -n "$cap" ]; then
    printf '%s' "$cap"
  else
    printf '8.9'
  fi
}

nvcc_major() {
  "$1" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+).*/\1/p' | head -n1
}

nvcc_version() {
  "$1" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n1
}

cuda_headers_missing() {
  local home="${1%/}" h
  for h in \
    cuda.h \
    cuda_runtime.h \
    cuda_runtime_api.h \
    crt/host_config.h \
    crt/host_defines.h \
    cublas_v2.h \
    cublasLt.h \
    cusparse.h \
    cusolverDn.h \
    nv/target
  do
    [ -f "$home/include/$h" ] || printf '%s ' "$h"
  done
}

cuda_headers_ok() {
  [ -z "$(cuda_headers_missing "$1")" ]
}

candidate_cuda_roots() {
  [ -n "$CUDA_HOME_ARG" ] && printf '%s\n' "$CUDA_HOME_ARG"
  [ -n "${CUDA_HOME:-}" ] && printf '%s\n' "$CUDA_HOME"
  printf '%s\n' \
    /usr/local/cuda \
    /usr/local/cuda-* \
    /opt/cuda \
    /opt/cuda-* \
    "$PROJECT_ROOT/.cuda$TORCH_CUDA_MAJOR" \
    "$CREATE_ENV_DIR/.cuda$TORCH_CUDA_MAJOR" \
    "${AUTO_CUDA_DIR:-}"
}

find_cuda_home() {
  local cand real major seen=""
  while IFS= read -r cand; do
    [ -n "$cand" ] || continue
    case "$cand" in *'*'*) continue ;; esac
    [ -x "$cand/bin/nvcc" ] || continue
    real="$(cd "$cand" 2>/dev/null && pwd || true)"
    [ -n "$real" ] || continue
    case "$seen" in
      *"|$real|"*) continue ;;
    esac
    seen="$seen|$real|"
    major="$(nvcc_major "$real/bin/nvcc")"
    if [ "$major" != "$TORCH_CUDA_MAJOR" ]; then
      warn "Skip $real: nvcc $(nvcc_version "$real/bin/nvcc") is not CUDA $TORCH_CUDA_MAJOR.x"
      continue
    fi
    if ! cuda_headers_ok "$real"; then
      warn "Skip $real: missing CUDA headers: $(cuda_headers_missing "$real")"
      continue
    fi
    printf '%s' "$real"
    return 0
  done < <(candidate_cuda_roots)
  return 1
}

install_cuda_redist() {
  local dest="$1"
  local ver="${CUDA_REDIST_VER:-12.9.1}"
  local base="${CUDA_REDIST_BASE:-https://developer.download.nvidia.com/compute/cuda/redist}"
  local components="cuda_nvcc cuda_cudart cuda_cccl libcublas libcusparse libcusolver cuda_cuobjdump"
  local headers_only="libcublas libcusparse libcusolver"
  local cache="$dest/.download"
  local manifest="$cache/redistrib_$ver.json"
  local list="$cache/components.tsv"
  local name rel sha size tgz tmp got

  command -v curl >/dev/null 2>&1 || die "curl is required to download CUDA redist"
  command -v python3 >/dev/null 2>&1 || die "python3 is required to parse CUDA redist manifest"
  command -v tar >/dev/null 2>&1 || die "tar is required to extract CUDA redist"
  mkdir -p "$cache"

  say "Download CUDA $ver redist"
  if [ ! -s "$manifest" ]; then
    run curl -fL --retry 3 --connect-timeout 20 -o "$manifest" "$base/redistrib_$ver.json"
  fi

  python3 - "$manifest" "$components" > "$list" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for name in sys.argv[2].split():
    item = (manifest.get(name) or {}).get("linux-x86_64") or {}
    if not item.get("relative_path"):
        raise SystemExit(f"missing linux-x86_64 payload for {name}")
    print("\t".join([name, item["relative_path"], item.get("sha256", ""), str(item.get("size", 0))]))
PY

  while IFS=$'\t' read -r name rel sha size; do
    tgz="$cache/$(basename "$rel")"
    if [ ! -s "$tgz" ]; then
      printf '  downloading %-16s %s MB\n' "$name" "$((size / 1048576))"
      run curl -fL --retry 3 -C - --connect-timeout 20 -o "$tgz" "$base/$rel"
    else
      ok "reuse $(basename "$tgz")"
    fi

    if [ -n "$sha" ]; then
      got="$(sha256sum "$tgz" | awk '{print $1}')"
      [ "$got" = "$sha" ] || die "sha256 mismatch for $tgz"
    fi

    tmp="$cache/extract"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    case " $headers_only " in
      *" $name "*) tar -xJf "$tgz" -C "$tmp" --wildcards '*/include/*' ;;
      *) tar -xJf "$tgz" -C "$tmp" ;;
    esac
    for d in "$tmp"/*; do
      [ -d "$d" ] && cp -a "$d"/. "$dest"/
    done
    rm -rf "$tmp"
    ok "$name installed"
  done < "$list"

  [ -d "$dest/lib" ] && [ ! -e "$dest/lib64" ] && ln -s lib "$dest/lib64"
  [ -x "$dest/bin/nvcc" ] || die "CUDA redist did not provide nvcc"
  cuda_headers_ok "$dest" || die "CUDA redist is missing headers: $(cuda_headers_missing "$dest")"
}

ensure_locked_source_link() {
  [ -d "$SOURCE_DIR" ] || die "GroundingDINO source not found: $SOURCE_DIR"
  [ -f "$CREATE_ENV_DIR/pyproject.toml" ] || die "pyproject.toml not found in $CREATE_ENV_DIR"
  [ -f "$CREATE_ENV_DIR/uv.lock" ] || die "uv.lock not found in $CREATE_ENV_DIR"

  if [ -L "$LOCKED_SOURCE_LINK" ]; then
    local target
    target="$(readlink "$LOCKED_SOURCE_LINK")"
    if [ "$target" = "../GroundingDINO" ]; then
      ok "editable source link exists: GroundingDINO -> $target"
      return 0
    fi
    [ "$MODE" = "check" ] && die "GroundingDINO link points somewhere else: $target"
    rm "$LOCKED_SOURCE_LINK"
  elif [ -e "$LOCKED_SOURCE_LINK" ]; then
    [ "$LOCKED_SOURCE_LINK" -ef "$SOURCE_DIR" ] && return 0
    die "$LOCKED_SOURCE_LINK exists but is not the expected source tree"
  fi

  if [ "$MODE" = "check" ]; then
    warn "Missing editable source link: $LOCKED_SOURCE_LINK -> ../GroundingDINO"
  else
    ln -s ../GroundingDINO "$LOCKED_SOURCE_LINK"
    ok "created editable source link: GroundingDINO -> ../GroundingDINO"
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    local uv_ver
    uv_ver="$(uv --version | awk '{print $2}')"
    version_ge "$uv_ver" "$MIN_UV_VERSION" || die "uv $uv_ver is too old; need >= $MIN_UV_VERSION"
    ok "uv $uv_ver"
    return 0
  fi

  [ "$MODE" = "check" ] && die "uv is not installed"
  command -v python3 >/dev/null 2>&1 || die "python3 is required to install uv"
  run python3 -m pip install --upgrade uv -i "$MIRROR_TUNA"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but is not on PATH"
  ok "$(uv --version)"
}

show_system_info() {
  say "System check"
  printf '  create_env : %s\n' "$CREATE_ENV_DIR"
  printf '  project    : %s\n' "$PROJECT_ROOT"
  printf '  source     : %s\n' "$SOURCE_DIR"
  printf '  log        : %s\n' "$LOG"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/  GPU        : /' || true
  else
    warn "nvidia-smi is not available or the NVIDIA driver is not running"
  fi
  df -h "$CREATE_ENV_DIR" "$DATA_DISK" 2>/dev/null | sed 's/^/  /' || true
}

prepare_cache_dirs() {
  say "Cache directories"
  if [ -d "$DATA_DISK" ] && [ -w "$DATA_DISK" ]; then
    export UV_CACHE_DIR="$DATA_DISK/uv-cache"
    export UV_PYTHON_INSTALL_DIR="$DATA_DISK/uv-python"
    AUTO_CUDA_DIR="${AUTO_CUDA_DIR:-$DATA_DISK/cuda$TORCH_CUDA_MAJOR}"
  else
    export UV_CACHE_DIR="$HOME/.cache/uv"
    export UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"
    AUTO_CUDA_DIR="${AUTO_CUDA_DIR:-$PROJECT_ROOT/.cuda$TORCH_CUDA_MAJOR}"
    warn "$DATA_DISK is not writable; using home/project directories"
  fi

  if [ "$MODE" != "check" ]; then
    mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
  fi
  ok "UV_CACHE_DIR=$UV_CACHE_DIR"
  ok "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
  ok "AUTO_CUDA_DIR=$AUTO_CUDA_DIR"
}

prepare_python() {
  say "Python $PY_VER"
  export UV_PYTHON_INSTALL_MIRROR="$MIRROR_PYTHON"
  ok "UV_PYTHON_INSTALL_MIRROR=$UV_PYTHON_INSTALL_MIRROR"
  if [ "$MODE" = "check" ]; then
    uv python find "$PY_VER" >/dev/null 2>&1 && ok "uv can find Python $PY_VER" || warn "Python $PY_VER is not installed yet"
  else
    run uv python install "$PY_VER"
  fi
}

prepare_cuda() {
  say "CUDA toolkit"
  if [ "${SKIP_CUDA_CHECK:-0}" = "1" ]; then
    warn "SKIP_CUDA_CHECK=1; skipping CUDA toolkit validation"
    return 0
  fi

  CUDA_ROOT="$(find_cuda_home || true)"
  if [ -z "$CUDA_ROOT" ]; then
    if [ "$MODE" = "check" ]; then
      warn "No complete CUDA $TORCH_CUDA_MAJOR.x toolkit found"
      return 0
    fi
    [ "$AUTO_CUDA" = "1" ] || die "No complete CUDA $TORCH_CUDA_MAJOR.x toolkit found; rerun with --cuda-home or without --no-auto-cuda"
    mkdir -p "$AUTO_CUDA_DIR"
    install_cuda_redist "$AUTO_CUDA_DIR"
    CUDA_ROOT="$AUTO_CUDA_DIR"
  fi

  export CUDA_HOME="$CUDA_ROOT"
  export PATH="$CUDA_HOME/bin:$PATH"
  ok "CUDA_HOME=$CUDA_HOME"
  ok "nvcc $(nvcc_version "$CUDA_HOME/bin/nvcc")"
}

sync_environment() {
  say "uv sync"
  if [ "$FORCE" = "1" ] && [ "$MODE" != "check" ]; then
    rm -rf "$CREATE_ENV_DIR/.venv"
    ok "removed $CREATE_ENV_DIR/.venv"
  fi

  export MAX_JOBS
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(detect_gpu_arch)}"
  ok "MAX_JOBS=$MAX_JOBS"
  ok "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

  if [ "$MODE" = "check" ]; then
    warn "check mode: not running uv sync"
    return 0
  fi

  run uv sync --frozen --no-build-isolation-package groundingdino
}

verify_environment() {
  [ "$MODE" = "check" ] && return 0

  say "Verify"
  local py="$CREATE_ENV_DIR/.venv/bin/python"
  [ -x "$py" ] || die "venv python not found: $py"

  "$py" - <<'PY'
import sys
import torch

print("  python:", sys.version.split()[0])
print("  torch :", torch.__version__)
print("  cuda  :", torch.version.cuda)
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
x = torch.randn(256, 256, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("  gpu   :", tuple(y.shape), "OK")
PY

  "$py" - <<'PY'
import torch
from groundingdino import _C

print("  _C    :", _C.__file__)
PY

  ok "environment ready"
  printf '\nActivate with:\n  source %s/.venv/bin/activate\n' "$CREATE_ENV_DIR"
}

main() {
  : > "$LOG" 2>/dev/null || true
  exec > >(tee -a "$LOG") 2>&1
  show_system_info
  ensure_locked_source_link
  prepare_cache_dirs
  ensure_uv
  prepare_python
  prepare_cuda
  sync_environment
  verify_environment
}

main "$@"
