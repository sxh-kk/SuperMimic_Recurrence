"""Download pinned GVHMR inference assets without using account credentials.

Assets retain their upstream licences and stay inside the ignored reference clone.
The body-model mirror is explicitly optional; it is not the official SMPL-X site.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "d50e513e101f7465d15508dc4c797c328200c78e"
WEIGHTS_ROOT = f"https://huggingface.co/camenduru/GVHMR/resolve/{REVISION}"
ASSETS = [
    (
        "gvhmr/gvhmr_siga24_release.ckpt",
        f"{WEIGHTS_ROOT}/gvhmr/gvhmr_siga24_release.ckpt",
        "4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91",
    ),
    (
        "hmr2/epoch=10-step=25000.ckpt",
        f"{WEIGHTS_ROOT}/hmr2/epoch%3D10-step%3D25000.ckpt",
        "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e",
    ),
    (
        "vitpose/vitpose-h-multi-coco.pth",
        f"{WEIGHTS_ROOT}/vitpose/vitpose-h-multi-coco.pth",
        "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc",
    ),
    (
        "yolo/yolov8x.pt",
        f"{WEIGHTS_ROOT}/yolo/yolov8x.pt",
        "c4d5a3f000d771762f03fc8b57ebd0aae324aeaefdd6e68492a9c4470f2d1e8b",
    ),
]
BODY_ASSET = (
    "body_models/smplx/SMPLX_NEUTRAL.npz",
    (
        "https://huggingface.co/spaces/feifeifeiliu/TalkSHOW/resolve/"
        "4e03fbc95a7c31046b358dfbb5ec7e1566af7ff6/smplx/SMPLX_NEUTRAL.npz"
    ),
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(root: Path, asset: tuple[str, str, str]) -> dict:
    relative, url, expected = asset
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == expected:
        print(f"Verified existing {relative}", flush=True)
    else:
        temporary = target.with_name(target.name + ".part")
        for attempt in range(3):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                request = urllib.request.Request(
                    url, headers={"Range": f"bytes={offset}-"} if offset else {}
                )
                with urllib.request.urlopen(request, timeout=300) as response:
                    append = offset > 0 and response.status == 206
                    with temporary.open("ab" if append else "wb") as handle:
                        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            handle.write(chunk)
                actual = sha256(temporary)
                if actual != expected:
                    temporary.unlink()
                    raise RuntimeError(f"SHA256 mismatch for {relative}: {actual}")
                temporary.replace(target)
                break
            except Exception:
                if attempt == 2:
                    raise
        print(f"Downloaded and verified {relative}", flush=True)
    return {
        "path": str(target),
        "source": url,
        "bytes": target.stat().st_size,
        "sha256": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("references/GVHMR"))
    parser.add_argument("--include-body-model", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--manifest", type=Path, default=Path("logs/gvhmr_setup/assets.json"))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    assets = ASSETS + ([BODY_ASSET] if args.include_body_model else [])
    root = args.repo.resolve() / "inputs/checkpoints"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda asset: download_asset(root, asset), assets))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "assets": results,
                "weight_source": "Mirror used by the upstream-linked LittleFrog/GVHMR demo",
                "body_source": "Third-party TalkSHOW mirror, not an official SMPL-X download",
                "licence_note": "Files retain their respective upstream licences; do not republish them.",
                "gvhmr_licence": "https://github.com/zju3dv/GVHMR/blob/main/LICENSE",
                "smplx_licence": "https://smpl-x.is.tue.mpg.de/modellicense.html",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
