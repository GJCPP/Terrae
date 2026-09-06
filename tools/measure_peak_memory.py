#!/usr/bin/env python3

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


def _read_status_kb(pid: int) -> tuple[int, int, int]:
    rss_kb = 0
    hwm_kb = 0
    swap_kb = 0
    try:
        with open(f"/proc/{pid}/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm_kb = int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    swap_kb = int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return 0, 0, 0
    return rss_kb, hwm_kb, swap_kb


def _task_children(pid: int) -> list[int]:
    children: list[int] = []
    task_dir = Path(f"/proc/{pid}/task")
    try:
        tids = [entry.name for entry in task_dir.iterdir()]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return children
    for tid in tids:
        try:
            text = Path(f"/proc/{pid}/task/{tid}/children").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for part in text.split():
            try:
                children.append(int(part))
            except ValueError:
                continue
    return children


def _process_tree(root_pid: int) -> set[int]:
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        if not Path(f"/proc/{pid}").exists():
            continue
        seen.add(pid)
        stack.extend(_task_children(pid))
    return seen


def _sample_tree_memory(root_pid: int) -> tuple[int, int, int, int]:
    rss_total_kb = 0
    hwm_total_kb = 0
    swap_total_kb = 0
    pids = _process_tree(root_pid)
    for pid in pids:
        rss_kb, hwm_kb, swap_kb = _read_status_kb(pid)
        rss_total_kb += rss_kb
        hwm_total_kb += hwm_kb
        swap_total_kb += swap_kb
    return rss_total_kb, hwm_total_kb, swap_total_kb, len(pids)


def _read_proc_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        text = path.read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return values
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def _read_mem_available_kb() -> int | None:
    value = _read_proc_key_values(Path("/proc/meminfo")).get("MemAvailable:")
    return value if value is not None else None


def _read_system_oom_kill_count() -> int | None:
    value = _read_proc_key_values(Path("/proc/vmstat")).get("oom_kill")
    return value if value is not None else None


def _cgroup_memory_events_path(pid: int) -> Path | None:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii").splitlines()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3 or parts[0] != "0":
            continue
        relative = parts[2].lstrip("/")
        path = Path("/sys/fs/cgroup") / relative / "memory.events"
        if path.exists():
            return path
    return None


def _counter_delta(current: int | None, initial: int | None) -> int:
    if current is None or initial is None:
        return 0
    return max(0, current - initial)


def _open_optional(path: Path | None):
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("wb")


def _is_suppressed_preprocessing_line(line: bytes) -> bool:
    text = line.decode("utf-8", errors="ignore").strip().lower()
    if not text:
        return False
    if "timing" not in text:
        return False
    suppressed_terms = (
        "assemble_exported_inference_plan",
        "assemble_plan_",
        "build_committed_forest_table",
        "build_inference_public_table",
        "commit_committed_forest_public_inputs",
        "commit_forest",
        "export_leaf_table",
        "prepare_inference_witness",
        "preprocess",
        "preprocessing",
        "preprocessed",
    )
    return any(term in text for term in suppressed_terms) or (
        "prepare_" in text and "verifier_bundle" in text
    )


def _filter_suppressed_lines(buffer: bytes, chunk: bytes) -> tuple[bytes, bytes]:
    combined = buffer + chunk
    lines = combined.split(b"\n")
    pending = lines.pop() if lines else b""
    kept = [line for line in lines if not _is_suppressed_preprocessing_line(line)]
    if not kept:
        return b"", pending
    return b"".join(line + b"\n" for line in kept), pending


def _flush_suppressed_buffer(buffer: bytes) -> bytes:
    if not buffer or _is_suppressed_preprocessing_line(buffer):
        return b""
    return buffer


def _terminate_process_tree(root_pid: int) -> None:
    for pid in sorted(_process_tree(root_pid), reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _kill_process_tree(root_pid: int) -> None:
    for pid in sorted(_process_tree(root_pid), reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _write_json_atomic(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command and sample peak RSS for its Linux /proc process tree."
    )
    parser.add_argument("--output-json", type=Path, help="Path for the measurement JSON.")
    parser.add_argument("--stdout-log", type=Path, help="Optional stdout log path.")
    parser.add_argument("--stderr-log", type=Path, help="Optional stderr log path.")
    parser.add_argument(
        "--suppress-preprocessing-timings",
        action="store_true",
        help="Suppress child output lines that report preprocessing or verifier-preparation timings.",
    )
    parser.add_argument("--cwd", type=Path, help="Working directory for the child command.")
    parser.add_argument("--poll-interval", type=float, default=0.05, help="Memory polling interval in seconds.")
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=1.0,
        help="Persist an atomic in-progress peak-memory snapshot at this interval in seconds.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional wall-clock timeout for the child command.",
    )
    parser.add_argument(
        "--rss-limit-bytes",
        type=int,
        default=None,
        help="Terminate when the process-tree RSS exceeds this limit.",
    )
    parser.add_argument(
        "--abort-on-swap",
        action="store_true",
        help="Terminate the command as soon as its process tree reports non-zero VmSwap.",
    )
    parser.add_argument(
        "--swap-limit-bytes",
        type=int,
        default=None,
        help="Terminate after process-tree VmSwap stays at or above this limit.",
    )
    parser.add_argument(
        "--swap-limit-duration-seconds",
        type=float,
        default=0.0,
        help="Required continuous time above --swap-limit-bytes before termination.",
    )
    parser.add_argument(
        "--min-mem-available-bytes",
        type=int,
        default=None,
        help="Terminate after host MemAvailable stays at or below this limit.",
    )
    parser.add_argument(
        "--min-mem-available-duration-seconds",
        type=float,
        default=0.0,
        help="Required continuous time below --min-mem-available-bytes before termination.",
    )
    parser.add_argument(
        "--abort-on-oom",
        action="store_true",
        help=(
            "Terminate the command when the containing cgroup reports a new OOM event. "
            "If cgroup v2 events are unavailable, use the system oom_kill counter."
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, usually after --.")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command to run")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be positive")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.rss_limit_bytes is not None and args.rss_limit_bytes <= 0:
        parser.error("--rss-limit-bytes must be positive")
    if args.swap_limit_bytes is not None and args.swap_limit_bytes <= 0:
        parser.error("--swap-limit-bytes must be positive")
    if args.swap_limit_duration_seconds < 0:
        parser.error("--swap-limit-duration-seconds must be non-negative")
    if args.min_mem_available_bytes is not None and args.min_mem_available_bytes <= 0:
        parser.error("--min-mem-available-bytes must be positive")
    if args.min_mem_available_duration_seconds < 0:
        parser.error("--min-mem-available-duration-seconds must be non-negative")

    stdout_log = _open_optional(args.stdout_log)
    stderr_log = _open_optional(args.stderr_log)
    selector = selectors.DefaultSelector()
    start = time.perf_counter()
    samples = 0
    peak_rss_kb = 0
    peak_hwm_sum_kb = 0
    current_swap_kb = 0
    peak_swap_kb = 0
    mem_available_kb: int | None = _read_mem_available_kb()
    minimum_mem_available_kb: int | None = mem_available_kb
    peak_process_count = 0
    last_sample = 0.0
    last_checkpoint = 0.0
    swap_limit_since: float | None = None
    low_mem_available_since: float | None = None
    timed_out = False
    interrupted_signal: int | None = None
    wrapper_error: str | None = None
    memory_guard_trigger: dict[str, object] | None = None
    return_code: int | None = None
    child_return_code: int | None = None
    process: subprocess.Popen[bytes] | None = None
    stream_buffers = {"stdout": b"", "stderr": b""}
    cgroup_memory_events_path = _cgroup_memory_events_path(os.getpid())
    cgroup_events_initial = (
        _read_proc_key_values(cgroup_memory_events_path)
        if cgroup_memory_events_path is not None
        else {}
    )
    cgroup_events_current = dict(cgroup_events_initial)
    system_oom_kill_initial = _read_system_oom_kill_count()
    system_oom_kill_current = system_oom_kill_initial

    def oom_deltas() -> tuple[int, int, int]:
        cgroup_oom_delta = _counter_delta(
            cgroup_events_current.get("oom"),
            cgroup_events_initial.get("oom"),
        )
        cgroup_oom_kill_delta = _counter_delta(
            cgroup_events_current.get("oom_kill"),
            cgroup_events_initial.get("oom_kill"),
        )
        system_oom_kill_delta = _counter_delta(
            system_oom_kill_current,
            system_oom_kill_initial,
        )
        return cgroup_oom_delta, cgroup_oom_kill_delta, system_oom_kill_delta

    def refresh_host_memory_evidence() -> None:
        nonlocal cgroup_events_current
        nonlocal system_oom_kill_current
        nonlocal mem_available_kb
        nonlocal minimum_mem_available_kb
        mem_available_kb = _read_mem_available_kb()
        if mem_available_kb is not None:
            if minimum_mem_available_kb is None:
                minimum_mem_available_kb = mem_available_kb
            else:
                minimum_mem_available_kb = min(minimum_mem_available_kb, mem_available_kb)
        if cgroup_memory_events_path is not None:
            cgroup_events_current = _read_proc_key_values(cgroup_memory_events_path)
        system_oom_kill_current = _read_system_oom_kill_count()

    def measurement(status: str, *, final: bool) -> dict[str, object]:
        elapsed = time.perf_counter() - start
        cgroup_oom_delta, cgroup_oom_kill_delta, system_oom_kill_delta = oom_deltas()
        return {
            "command": command,
            "cwd": str(args.cwd) if args.cwd is not None else None,
            "status": status,
            "final": final,
            "returncode": return_code,
            "child_returncode": child_return_code,
            "root_pid": process.pid if process is not None else None,
            "elapsed_seconds": elapsed,
            "poll_interval_seconds": args.poll_interval,
            "checkpoint_interval_seconds": args.checkpoint_interval,
            "timeout_seconds": args.timeout_seconds,
            "rss_limit_bytes": args.rss_limit_bytes,
            "timed_out": timed_out,
            "abort_on_swap": args.abort_on_swap,
            "abort_on_oom": args.abort_on_oom,
            "swap_limit_bytes": args.swap_limit_bytes,
            "swap_limit_duration_seconds": args.swap_limit_duration_seconds,
            "min_mem_available_bytes": args.min_mem_available_bytes,
            "min_mem_available_duration_seconds": args.min_mem_available_duration_seconds,
            "interrupted_signal": interrupted_signal,
            "wrapper_error": wrapper_error,
            "memory_guard_trigger": memory_guard_trigger,
            "samples": samples,
            "peak_rss_kb": peak_rss_kb,
            "peak_rss_bytes": peak_rss_kb * 1024,
            "peak_hwm_sum_kb": peak_hwm_sum_kb,
            "peak_hwm_sum_bytes": peak_hwm_sum_kb * 1024,
            "current_swap_kb": current_swap_kb,
            "current_swap_bytes": current_swap_kb * 1024,
            "peak_swap_kb": peak_swap_kb,
            "peak_swap_bytes": peak_swap_kb * 1024,
            "mem_available_kb": mem_available_kb,
            "mem_available_bytes": (
                mem_available_kb * 1024 if mem_available_kb is not None else None
            ),
            "minimum_mem_available_kb": minimum_mem_available_kb,
            "minimum_mem_available_bytes": (
                minimum_mem_available_kb * 1024
                if minimum_mem_available_kb is not None
                else None
            ),
            "cgroup_memory_events_path": (
                str(cgroup_memory_events_path)
                if cgroup_memory_events_path is not None
                else None
            ),
            "cgroup_memory_events_initial": cgroup_events_initial,
            "cgroup_memory_events_current": cgroup_events_current,
            "cgroup_oom_delta": cgroup_oom_delta,
            "cgroup_oom_kill_delta": cgroup_oom_kill_delta,
            "system_oom_kill_initial": system_oom_kill_initial,
            "system_oom_kill_current": system_oom_kill_current,
            "system_oom_kill_delta": system_oom_kill_delta,
            "peak_process_count": peak_process_count,
            "stdout_log": str(args.stdout_log) if args.stdout_log is not None else None,
            "stderr_log": str(args.stderr_log) if args.stderr_log is not None else None,
        }

    def persist(status: str, *, final: bool) -> None:
        result = measurement(status, final=final)
        if args.output_json is not None:
            _write_json_atomic(args.output_json, result)
        elif final:
            print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted_signal
        if interrupted_signal is None:
            interrupted_signal = signum

    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, handle_signal)

    try:
        process = subprocess.Popen(
            command,
            cwd=str(args.cwd) if args.cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, ("stdout", stdout_log, sys.stdout.buffer))
        selector.register(process.stderr, selectors.EVENT_READ, ("stderr", stderr_log, sys.stderr.buffer))
        persist("running", final=False)
        last_checkpoint = time.perf_counter()

        while selector.get_map() or process.poll() is None:
            now = time.perf_counter()
            should_stop = False
            if interrupted_signal is not None and process.poll() is None:
                should_stop = True
            elif args.timeout_seconds is not None and now - start >= args.timeout_seconds and process.poll() is None:
                timed_out = True
                should_stop = True

            if should_stop:
                _terminate_process_tree(process.pid)
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    _kill_process_tree(process.pid)
                break

            if now - last_sample >= args.poll_interval and process.poll() is None:
                rss_kb, hwm_sum_kb, swap_kb, process_count = _sample_tree_memory(process.pid)
                peak_rss_kb = max(peak_rss_kb, rss_kb)
                peak_hwm_sum_kb = max(peak_hwm_sum_kb, hwm_sum_kb)
                current_swap_kb = swap_kb
                peak_swap_kb = max(peak_swap_kb, swap_kb)
                peak_process_count = max(peak_process_count, process_count)
                samples += 1
                last_sample = now
                refresh_host_memory_evidence()
                cgroup_oom_delta, cgroup_oom_kill_delta, system_oom_kill_delta = oom_deltas()
                cgroup_oom_detected = cgroup_oom_delta > 0 or cgroup_oom_kill_delta > 0
                fallback_oom_detected = (
                    cgroup_memory_events_path is None and system_oom_kill_delta > 0
                )
                effective_swap_limit_bytes = (
                    1 if args.abort_on_swap else args.swap_limit_bytes
                )
                effective_swap_duration_seconds = (
                    0.0 if args.abort_on_swap else args.swap_limit_duration_seconds
                )
                if (
                    effective_swap_limit_bytes is not None
                    and swap_kb * 1024 >= effective_swap_limit_bytes
                ):
                    if swap_limit_since is None:
                        swap_limit_since = now
                else:
                    swap_limit_since = None
                if (
                    args.min_mem_available_bytes is not None
                    and mem_available_kb is not None
                    and mem_available_kb * 1024 <= args.min_mem_available_bytes
                ):
                    if low_mem_available_since is None:
                        low_mem_available_since = now
                else:
                    low_mem_available_since = None

                swap_limit_reached = (
                    swap_limit_since is not None
                    and now - swap_limit_since >= effective_swap_duration_seconds
                )
                low_mem_available_reached = (
                    low_mem_available_since is not None
                    and now - low_mem_available_since
                    >= args.min_mem_available_duration_seconds
                )
                if args.abort_on_oom and (cgroup_oom_detected or fallback_oom_detected):
                    memory_guard_trigger = {
                        "kind": "oom",
                        "elapsed_seconds": now - start,
                        "rss_bytes": rss_kb * 1024,
                        "swap_bytes": swap_kb * 1024,
                        "mem_available_bytes": (
                            mem_available_kb * 1024
                            if mem_available_kb is not None
                            else None
                        ),
                        "cgroup_oom_delta": cgroup_oom_delta,
                        "cgroup_oom_kill_delta": cgroup_oom_kill_delta,
                        "system_oom_kill_delta": system_oom_kill_delta,
                    }
                elif (
                    args.rss_limit_bytes is not None
                    and rss_kb * 1024 > args.rss_limit_bytes
                ):
                    memory_guard_trigger = {
                        "kind": "rss",
                        "elapsed_seconds": now - start,
                        "rss_bytes": rss_kb * 1024,
                        "rss_limit_bytes": args.rss_limit_bytes,
                        "swap_bytes": swap_kb * 1024,
                        "mem_available_bytes": (
                            mem_available_kb * 1024
                            if mem_available_kb is not None
                            else None
                        ),
                    }
                elif low_mem_available_reached:
                    memory_guard_trigger = {
                        "kind": "memory_pressure",
                        "elapsed_seconds": now - start,
                        "rss_bytes": rss_kb * 1024,
                        "swap_bytes": swap_kb * 1024,
                        "mem_available_bytes": mem_available_kb * 1024,
                        "min_mem_available_bytes": args.min_mem_available_bytes,
                        "required_duration_seconds": args.min_mem_available_duration_seconds,
                        "observed_duration_seconds": now - low_mem_available_since,
                    }
                elif swap_limit_reached:
                    memory_guard_trigger = {
                        "kind": "swap",
                        "elapsed_seconds": now - start,
                        "rss_bytes": rss_kb * 1024,
                        "swap_bytes": swap_kb * 1024,
                        "mem_available_bytes": (
                            mem_available_kb * 1024
                            if mem_available_kb is not None
                            else None
                        ),
                        "swap_limit_bytes": effective_swap_limit_bytes,
                        "required_duration_seconds": effective_swap_duration_seconds,
                        "observed_duration_seconds": now - swap_limit_since,
                    }
                if memory_guard_trigger is not None:
                    trigger_status = {
                        "oom": "oom_detected",
                        "rss": "rss_limit_detected",
                        "swap": "swap_detected",
                        "memory_pressure": "memory_pressure_detected",
                    }[str(memory_guard_trigger["kind"])]
                    persist(trigger_status, final=False)
                    _terminate_process_tree(process.pid)
                    sigkill_sent = False
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        _kill_process_tree(process.pid)
                        sigkill_sent = True
                        process.wait()
                    child_return_code = process.returncode
                    memory_guard_trigger["termination"] = {
                        "sigterm_sent": True,
                        "sigkill_sent": sigkill_sent,
                        "child_returncode": child_return_code,
                    }
                    break

            if now - last_checkpoint >= args.checkpoint_interval:
                persist("running", final=False)
                last_checkpoint = now

            for key, _ in selector.select(timeout=args.poll_interval):
                stream_name, log_handle, mirror = key.data
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    if args.suppress_preprocessing_timings:
                        pending = _flush_suppressed_buffer(stream_buffers[stream_name])
                        if pending:
                            mirror.write(pending)
                            mirror.flush()
                            if log_handle is not None:
                                log_handle.write(pending)
                                log_handle.flush()
                        stream_buffers[stream_name] = b""
                    selector.unregister(key.fileobj)
                    continue
                if args.suppress_preprocessing_timings:
                    chunk, stream_buffers[stream_name] = _filter_suppressed_lines(
                        stream_buffers[stream_name], chunk)
                    if not chunk:
                        continue
                mirror.write(chunk)
                mirror.flush()
                if log_handle is not None:
                    log_handle.write(chunk)
                    log_handle.flush()

        return_code = process.wait()
        child_return_code = return_code
        refresh_host_memory_evidence()
    except BaseException as error:
        wrapper_error = f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            _terminate_process_tree(process.pid)
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _kill_process_tree(process.pid)
                process.wait()
        if process is not None:
            return_code = process.returncode
            child_return_code = process.returncode
        if return_code is None:
            return_code = 1
    finally:
        selector.close()
        if stdout_log is not None:
            stdout_log.close()
        if stderr_log is not None:
            stderr_log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    cgroup_oom_delta, cgroup_oom_kill_delta, system_oom_kill_delta = oom_deltas()
    oom_detected = (
        cgroup_oom_delta > 0
        or cgroup_oom_kill_delta > 0
        or (cgroup_memory_events_path is None and system_oom_kill_delta > 0)
    )
    if memory_guard_trigger is not None:
        return_code = 125
        status = {
            "oom": "oom_detected",
            "rss": "rss_limit_detected",
            "swap": "swap_detected",
            "memory_pressure": "memory_pressure_detected",
        }[str(memory_guard_trigger["kind"])]
    elif args.abort_on_oom and oom_detected and return_code != 0:
        memory_guard_trigger = {
            "kind": "oom",
            "detected_after_child_exit": True,
            "cgroup_oom_delta": cgroup_oom_delta,
            "cgroup_oom_kill_delta": cgroup_oom_kill_delta,
            "system_oom_kill_delta": system_oom_kill_delta,
        }
        return_code = 125
        status = "oom_detected"
    elif timed_out:
        return_code = 124
        status = "timed_out"
    elif interrupted_signal is not None:
        return_code = 128 + interrupted_signal
        status = "interrupted"
    elif wrapper_error is not None:
        status = "wrapper_error"
    elif return_code == 0:
        status = "completed"
    else:
        status = "failed"
    persist(status, final=True)

    if memory_guard_trigger is not None:
        print(
            "measure_peak_memory: memory guard stopped command"
            f"; kind={memory_guard_trigger['kind']}"
            f"; peak_rss_bytes={peak_rss_kb * 1024}"
            f"; peak_swap_bytes={peak_swap_kb * 1024}",
            file=sys.stderr,
            flush=True,
        )
    elif timed_out:
        print(
            f"measure_peak_memory: command timed out after {args.timeout_seconds:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    elif interrupted_signal is not None:
        print(
            f"measure_peak_memory: interrupted by signal {interrupted_signal}; final peak snapshot saved",
            file=sys.stderr,
            flush=True,
        )
    elif wrapper_error is not None:
        print(f"measure_peak_memory: {wrapper_error}", file=sys.stderr, flush=True)
    return int(return_code if return_code is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
