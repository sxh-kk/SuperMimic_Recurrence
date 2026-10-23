"""Archive a checkpoint after a cumulative run deadline, then stop its managed worker.

Default fallback: after 300 seconds without a complete post-deadline numbered
checkpoint, archive the newest complete numbered checkpoint available, then stop.
No SIGKILL is sent. --dry-run only reads metadata and verifies a Linux pidfd.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import signal
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUMBERED = re.compile(r"ckpt_(\d+)\.pt$")
TERMINAL = {"completed", "failed"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity(pid: int) -> dict | None:
    """Linux stat field 22 distinguishes PID reuse; zombies count as exited."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return {
            "pid": pid, "starttime_ticks": int(fields[19]), "state": fields[0],
            "parent_pid": int(fields[1]),
            "command": [part.decode(errors="replace") for part in command if part],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def still_alive(identity: dict) -> bool:
    current = process_identity(identity["pid"])
    return bool(
        current and current["starttime_ticks"] == identity["starttime_ticks"]
        and current["state"] not in ("Z", "X")
    )


def pidfd_call(number: int, *arguments) -> int:
    """Python builds may omit pidfd APIs even when the Linux kernel supports them."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("pidfd syscall fallback requires Linux x86_64")
    library = ctypes.CDLL(None, use_errno=True)
    result = library.syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(result)


def open_pidfd(pid: int) -> int:
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    return pidfd_call(434, ctypes.c_int(pid), ctypes.c_uint(0))


def send_term(pidfd: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)
    else:
        pidfd_call(
            424, ctypes.c_int(pidfd), ctypes.c_int(signal.SIGTERM),
            ctypes.c_void_p(), ctypes.c_uint(0),
        )


def bind_worker(pid: int, launch_path: Path) -> tuple[dict, int]:
    identity = process_identity(pid)
    if identity is None or identity["state"] in ("Z", "X"):
        raise RuntimeError("managed worker is not alive")
    command = identity["command"]
    launcher = str(ROOT / "scripts/local/start_train_4090.py")
    if launcher not in command or "--worker" not in command:
        raise RuntimeError("worker PID is not the expected managed launcher")
    index = command.index("--worker")
    if index + 1 >= len(command) or Path(command[index + 1]).resolve() != launch_path:
        raise RuntimeError("worker belongs to a different run")
    descriptor = open_pidfd(pid)
    if not still_alive(identity):
        os.close(descriptor)
        raise RuntimeError("worker identity changed while opening pidfd")
    return identity, descriptor


def capture_children(identity: dict, tracked: dict) -> None:
    if not still_alive(identity):
        return
    # Linux lists children per spawning thread; uv may spawn from a worker thread.
    pids: set[int] = set()
    for children in Path(f"/proc/{identity['pid']}/task").glob("*/children"):
        try:
            pids.update(int(text) for text in children.read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            continue  # A thread may exit while its proc entry is being inspected.
    if not still_alive(identity):
        return
    for pid in sorted(pids):
        child = process_identity(pid)
        if child and child["parent_pid"] == identity["pid"]:
            tracked[(child["pid"], child["starttime_ticks"])] = child


def latest_iteration(console: Path) -> int:
    try:
        with console.open("rb") as stream:
            stream.seek(max(0, console.stat().st_size - 512 * 1024))
            text = stream.read().decode(errors="replace")
    except FileNotFoundError:
        return -1
    return max((int(n) for n in re.findall(r"(?m)^it (\d+) \|", text)), default=-1)


def numbered_files(run_dir: Path, iteration: int) -> list[tuple[Path, int, os.stat_result]]:
    files = []
    for path in run_dir.glob("ckpt_*.pt"):
        match = NUMBERED.fullmatch(path.name)
        if match and int(match[1]) < iteration:
            try:
                files.append((path, int(match[1]), path.stat()))
            except FileNotFoundError:
                pass
    return files


def fingerprint(stat: os.stat_result) -> tuple[int, ...]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def validate_checkpoint(path: Path, iteration: int, run_name: str) -> dict:
    # Import only after the deadline. All tensors are explicitly loaded on CPU.
    import torch

    torch.set_num_threads(2)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("iteration") != iteration:
        raise ValueError("checkpoint iteration differs from its numbered filename")
    if checkpoint.get("config", {}).get("run_name") != run_name:
        raise ValueError("checkpoint run_name does not match this run directory")
    invalid, tensor_count = [], 0
    for name in ("wm", "policy", "critic"):
        values = checkpoint.get(name)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"missing/non-dictionary model state: {name}")
        for key, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"non-tensor model state: {name}.{key}")
            tensor_count += 1
            if not torch.isfinite(value).all().item():
                invalid.append(f"{name}.{key}")
    return {
        "iteration": iteration, "run_name": run_name,
        "num_updates": checkpoint.get("num_updates"), "model_tensors": tensor_count,
        "models_finite": not invalid, "nonfinite_tensors": invalid,
    }


def archive(source: Path, target: Path, expected: os.stat_result) -> dict:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as original, temporary.open("xb") as output:
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if fingerprint(source.stat()) != fingerprint(expected):
            raise RuntimeError("numbered checkpoint changed during validation/copy")
        if temporary.stat().st_size != expected.st_size:
            raise RuntimeError("checkpoint archive size differs from source")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "bytes": expected.st_size, "sha256": digest.hexdigest()}


def run(args, deadline: datetime, target: Path, worker: dict, pidfd: int) -> int:
    run_dir = args.run_dir
    output = run_dir / "stop_status.json"
    status = {
        "controller_pid": os.getpid(), "started_utc": now(), "state": "waiting_deadline",
        "run_dir": str(run_dir), "hours": args.hours, "deadline_utc": deadline.isoformat(),
        "worker_identity": worker, "signal_sent": False, "checkpoint": None,
        "checkpoint_wait_seconds": args.checkpoint_wait_seconds,
        "fallback_policy": "newest complete numbered checkpoint, or stop without archive",
        "checkpoint_errors": [],
    }
    tracked = {(worker["pid"], worker["starttime_ticks"]): worker}

    def observe() -> dict:
        lifecycle = read_json(run_dir / "status.json")
        if lifecycle.get("worker_pid") != worker["pid"]:
            raise RuntimeError("lifecycle status belongs to a different worker")
        # Capture the managed uv child and its immediate trainer children. Never signal them.
        capture_children(worker, tracked)
        for identity in list(tracked.values()):
            if identity["pid"] == lifecycle.get("training_pid"):
                capture_children(identity, tracked)
        status["tracked_processes"] = list(tracked.values())
        return lifecycle

    def finish(state: str, code: int, **details) -> int:
        status.update(state=state, finished_utc=now(), **details)
        write_json(output, status)
        print(json.dumps(status, indent=2), flush=True)
        return code

    try:
        write_json(output, status)
        while datetime.now(UTC) < deadline:
            lifecycle = observe()
            if lifecycle.get("state") in TERMINAL or not still_alive(worker):
                return finish("training_ended_before_deadline", 1, training_status=lifecycle)
            time.sleep(args.poll_seconds)

        status.update(state="waiting_checkpoint", deadline_reached_utc=now())
        write_json(output, status)
        seen = set()
        selected = None
        fallback_reason = None
        wait_until = deadline + timedelta(seconds=args.checkpoint_wait_seconds)

        def try_candidates(candidates) -> dict | None:
            for path, iteration, stat in candidates:
                tag = (path.name, fingerprint(stat))
                if tag in seen:
                    continue
                seen.add(tag)
                try:
                    checked = validate_checkpoint(path, iteration, run_dir.name)
                    copied = archive(path, target, stat)
                    return {
                        "source": str(path), "source_mtime_utc": datetime.fromtimestamp(
                            stat.st_mtime, UTC
                        ).isoformat(), "post_deadline": stat.st_mtime >= deadline.timestamp(),
                        **checked, "archive": copied,
                    }
                except Exception as error:  # noqa: BLE001 - corrupt files need a bounded fallback.
                    status["checkpoint_errors"].append({"path": str(path), "error": repr(error)})
                    write_json(output, status)
            return None

        while selected is None:
            lifecycle = observe()
            iteration = latest_iteration(run_dir / "console.log")
            status["latest_console_iteration"] = iteration
            complete = numbered_files(run_dir, iteration)
            eligible = [item for item in complete if item[2].st_mtime >= deadline.timestamp()]
            selected = try_candidates(sorted(eligible, key=lambda item: (item[2].st_mtime_ns, item[1])))
            if selected:
                break
            if lifecycle.get("state") in TERMINAL or not still_alive(worker):
                fallback_reason = "training ended before a post-deadline checkpoint was confirmed"
                break
            if datetime.now(UTC) >= wait_until:
                fallback_reason = "post-deadline checkpoint wait exceeded its bounded grace period"
                break
            time.sleep(args.poll_seconds)

        if selected is None:
            # Bound recovery work to the three newest already-complete numbered files.
            complete = numbered_files(run_dir, latest_iteration(run_dir / "console.log"))
            selected = try_candidates(sorted(complete, key=lambda item: item[1], reverse=True)[:3])
        status.update(checkpoint=selected, fallback_reason=fallback_reason, state="requesting_stop")
        write_json(output, status)
        lifecycle = observe()
        if still_alive(worker) and lifecycle.get("state") not in TERMINAL:
            # Recheck starttime immediately before signaling the bound pidfd: no PID reuse race.
            try:
                send_term(pidfd)
            except ProcessLookupError:
                status["signal_race"] = "worker exited before SIGTERM delivery"
            else:
                status.update(signal_sent=True, signal="SIGTERM", signal_sent_utc=now())
            write_json(output, status)
        status["stop_reason"] = "user requested a cumulative training-duration limit"
        end_wait = time.monotonic() + args.exit_wait_seconds
        while True:
            lifecycle = observe()
            alive = [identity for identity in tracked.values() if still_alive(identity)]
            if lifecycle.get("state") in TERMINAL and not alive:
                valid = bool(selected and selected["models_finite"])
                return finish(
                    "stopped" if valid else "stopped_checkpoint_invalid_or_missing",
                    0 if valid else 1, training_status=lifecycle,
                    original_training_returncode=lifecycle.get("returncode"), live_processes=[],
                    returncode_explanation=(
                        "A nonzero training returncode after our SIGTERM is an intentional user stop; "
                        "checkpoint archival success or failure is recorded separately above."
                        if status["signal_sent"] else "Training exited without our sending a signal."
                    ),
                )
            if time.monotonic() >= end_wait:
                return finish("stop_unconfirmed", 1, training_status=lifecycle, live_processes=alive)
            time.sleep(args.poll_seconds)
    except Exception as error:  # noqa: BLE001 - persist unexpected controller failures for review.
        return finish("controller_failed", 1, error=repr(error))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--checkpoint-wait-seconds", type=float, default=300.0)
    parser.add_argument("--exit-wait-seconds", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("hours", "poll_seconds", "checkpoint_wait_seconds", "exit_wait_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive and finite")
    args.run_dir = args.run_dir.resolve(strict=True)
    launch_path = args.run_dir / "launch.json"
    launch = read_json(launch_path)
    created = datetime.fromisoformat(launch["created_utc"])
    if created.tzinfo is None:
        parser.error("launch created_utc must include its timezone")
    deadline = created.astimezone(UTC) + timedelta(hours=args.hours)
    target = args.run_dir / f"ckpt_{args.hours:g}h.pt"
    if args.dry_run:
        worker, descriptor = bind_worker(int(launch["worker_pid"]), launch_path)
        os.close(descriptor)
        print(json.dumps({
            "dry_run": True, "run_dir": str(args.run_dir), "deadline_utc": deadline.isoformat(),
            "archive_target": str(target), "worker_identity": worker, "pidfd_verified": True,
            "actions": "wait, validate/archive on CPU, SIGTERM worker, confirm process exit",
            "checkpoint_grace_seconds": args.checkpoint_wait_seconds,
            "no_files_written_or_signals_sent": True,
        }, indent=2))
        return 0

    with (args.run_dir / ".stop_after_checkpoint.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another stop controller already holds this run's lock")
        existing = args.run_dir / "stop_status.json"
        if existing.exists() and read_json(existing).get("signal_sent"):
            parser.error("a stop controller already requested termination for this run")
        worker, descriptor = bind_worker(int(launch["worker_pid"]), launch_path)
        try:
            return run(args, deadline, target, worker, descriptor)
        finally:
            os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
