"""Start a detached local training process with persistent console and exit status."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def worker(launch_path: Path) -> int:
    launch = json.loads(launch_path.read_text())
    status_path = launch_path.parent / "status.json"
    status = {"worker_pid": os.getpid(), "started_utc": utc_now(), "state": "starting"}
    # Keep one launcher-managed training job active on this single-GPU workstation.
    with (ROOT / "logs" / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child = subprocess.Popen(
                launch["command"],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                pass_fds=(lock.fileno(),),
            )

            def forward_signal(signum, _frame):
                if child.poll() is None:
                    try:
                        child.send_signal(signum)
                    except ProcessLookupError:
                        pass

            signal.signal(signal.SIGTERM, forward_signal)
            signal.signal(signal.SIGINT, forward_signal)
            status.update(state="running", training_pid=child.pid)
            write_json(status_path, status)
            returncode = child.wait()
            status.update(state="completed" if returncode == 0 else "failed", returncode=returncode)
        except (OSError, subprocess.SubprocessError) as error:
            status.update(state="failed", returncode=1, error=str(error))
        status["finished_utc"] = utc_now()
        write_json(status_path, status)
    return status["returncode"]


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(worker(Path(sys.argv[2])))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("lafan", "lafan-lowmem", "bones"), default="lafan")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_name):
        parser.error("run-name must be a safe directory name (1–96 characters)")
    extra = args.training_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if any(x.split("=", 1)[0].replace("_", "-") in ("--run-name", "--log-root") for x in extra):
        parser.error("run-name and log-root are managed by this launcher")

    run_dir = ROOT / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        "bash",
        str(ROOT / "scripts/local/train_4090.sh"),
        args.profile,
        "--run-name",
        args.run_name,
        "--log-root",
        str(ROOT / "logs"),
        *extra,
    ]
    launch_path = run_dir / "launch.json"
    launch = {
        "created_utc": utc_now(),
        "command": command,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    write_json(launch_path, launch)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONUNBUFFERED"] = "1"
    with (run_dir / "console.log").open("x") as console:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(launch_path)],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch["worker_pid"] = process.pid
    write_json(launch_path, launch)
    print(json.dumps({"run_dir": str(run_dir), "worker_pid": process.pid}, indent=2))
    print("Process dispatched; check status.json and console.log for actual training progress.")


if __name__ == "__main__":
    main()
