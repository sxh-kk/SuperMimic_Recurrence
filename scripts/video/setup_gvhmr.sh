#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
REVISION=ee960bb6e2ea2d381aa97f08e9b71ef320b624b1
if [ ! -d references/GVHMR/.git ]; then
  git clone https://github.com/zju3dv/GVHMR.git references/GVHMR
  git -C references/GVHMR checkout --detach "$REVISION"
fi
if [ "$(git -C references/GVHMR rev-parse HEAD)" != "$REVISION" ]; then
  printf 'GVHMR版本与记录不一致，请在独立reference clone检查后切换至%s。\n' "$REVISION" >&2
  exit 2
fi
cd references/GVHMR
uv venv .venv --python 3.10 --allow-existing
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.0+cu121 torchvision==0.18.0+cu121 numpy==1.23.5 \
  setuptools==69.5.1 wheel scipy==1.13.1 cython 'pip<26'
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --no-build-isolation -r requirements.txt \
  -c "$PROJECT_ROOT/scripts/video/gvhmr_constraints.txt"
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
cd "$PROJECT_ROOT"
python3 scripts/video/download_gvhmr_assets.py "$@"
