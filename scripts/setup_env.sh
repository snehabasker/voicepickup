#!/usr/bin/env bash
# Reproducible training environment for WSL2 / Linux with an NVIDIA GPU.
# Blackwell laptop GPUs (sm_120, e.g. RTX 50xx) need a PyTorch wheel built
# against CUDA >= 12.8; older cu121/cu124 wheels import fine but fail at the
# first kernel launch with "no kernel image is available".
set -euo pipefail
VENV="${VENV:-$HOME/venvs/voicepickup}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# pesq and pyroomacoustics compile C/C++ extensions. Use $CC if set, else a
# conda-forge GCC under ~/gcc14 if present, else the system compiler.
if [ -z "${CC:-}" ] && [ -x "$HOME/gcc14/bin/gcc" ]; then
  export PATH="$HOME/gcc14/bin:$PATH" CC="$HOME/gcc14/bin/gcc" CXX="$HOME/gcc14/bin/g++"
fi

command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
[ -d "$VENV" ] || uv venv -p 3.12 "$VENV"
export VIRTUAL_ENV="$VENV"
uv pip install "torch==2.11.0" --index-url "$TORCH_INDEX"
uv pip install -e "$REPO[metrics,sim,train,dev]"

"$VENV/bin/python" - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0) if ok else None)
if ok:
    x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item()  # forces a real kernel launch
    print("kernel launch OK, arch list:", torch.cuda.get_arch_list())
PY
