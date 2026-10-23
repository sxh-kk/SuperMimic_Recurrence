#!/usr/bin/env bash
# Local presets; additional sm-train arguments follow the profile name.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
unset PYTHONPATH PYTHONHOME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROFILE="${1:-lafan}"
if [ "$#" -gt 0 ]; then shift; fi
case "$PROFILE" in
  lafan)
    OPTIONS=(--run-name lafan_paper_s0 --data.npz-dir data/npz/lafan1
      --env.num-envs 4096 --policy-trainer.batch-size 4096)
    ;;
  lafan-lowmem)
    OPTIONS=(--run-name lafan_lowmem_s0 --data.npz-dir data/npz/lafan1
      --env.num-envs 2048 --policy-trainer.batch-size 2048 --wm-trainer.batch-size 1024)
    ;;
  bones)
    OPTIONS=(--run-name bones_paper_s0 --data.library data/library/bones_seed_compact.pt
      --env.num-envs 4096 --policy-trainer.batch-size 2048)
    ;;
  *)
    printf '用法：bash scripts/local/train_4090.sh {lafan|lafan-lowmem|bones} [sm-train 参数...]\n' >&2
    exit 2
    ;;
esac
exec uv run --locked sm-train "${OPTIONS[@]}" \
  --policy-trainer.horizon 32 --env.replay-capacity 1000 \
  --schedule.max-hours 6 "$@"
