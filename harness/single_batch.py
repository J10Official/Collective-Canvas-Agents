"""Run a configured single-team collective-canvas batch without the web UI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import agent_runner as agent_runner_module
from .agent_runner import ROOT, resume_saved_run, run_team_drawing
from .batch_video import build_webm
from .openrouter_client import load_env
from .references import DEFAULT_OBJECTS

PROJECT_ROOT = ROOT.parent


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


class Tee:
    def __init__(self, terminal: Any, log_path: Path) -> None:
        self.terminal = terminal
        self.handle = log_path.open("a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def write(self, value: str) -> int:
        with self.lock:
            self.terminal.write(value)
            self.handle.write(value)
        return len(value)

    def flush(self) -> None:
        with self.lock:
            self.terminal.flush()
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class BudgetReached(RuntimeError):
    pass


class BudgetTracker:
    def __init__(self, run_ids: list[str], limit_usd: float) -> None:
        self.run_ids = run_ids
        self.limit_usd = limit_usd
        self.lock = threading.Lock()
        self.cost_usd = 0.0
        self.seen_paths: set[Path] = set()
        self.seen_response_ids: set[str] = set()
        self.refresh()

    def refresh(self) -> float:
        with self.lock:
            for run_id in self.run_ids:
                response_root = ROOT / "runs" / run_id / "responses"
                for path in response_root.glob("*.json") if response_root.is_dir() else ():
                    resolved = path.resolve()
                    if resolved in self.seen_paths:
                        continue
                    response = _read_json(path, {})
                    if not isinstance(response, dict):
                        continue
                    response_id = response.get("id")
                    identity = str(response_id) if response_id else str(resolved)
                    self.seen_paths.add(resolved)
                    if identity in self.seen_response_ids:
                        continue
                    self.seen_response_ids.add(identity)
                    usage = response.get("usage", {})
                    cost = usage.get("cost") if isinstance(usage, dict) else None
                    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                        self.cost_usd += float(cost)
            return self.cost_usd

    def reached(self) -> bool:
        return self.refresh() >= self.limit_usd


class ProviderGate:
    def __init__(self, budget: BudgetTracker, per_provider: int) -> None:
        self.budget = budget
        self.per_provider = per_provider
        self.lock = threading.Lock()
        self.semaphores: dict[str, threading.BoundedSemaphore] = {}
        self.active: Counter[str] = Counter()
        self.maximum_active: Counter[str] = Counter()

    @contextmanager
    def __call__(self, request: dict[str, Any]) -> Iterator[None]:
        provider_data = request.get("provider", {})
        order = provider_data.get("order", []) if isinstance(provider_data, dict) else []
        provider = str(order[0]) if order else "unknown"
        with self.lock:
            semaphore = self.semaphores.setdefault(provider, threading.BoundedSemaphore(self.per_provider))
        if self.budget.reached():
            raise BudgetReached(f"batch cost limit of ${self.budget.limit_usd:.2f} reached")
        with semaphore:
            if self.budget.reached():
                raise BudgetReached(f"batch cost limit of ${self.budget.limit_usd:.2f} reached")
            with self.lock:
                self.active[provider] += 1
                self.maximum_active[provider] = max(self.maximum_active[provider], self.active[provider])
            try:
                yield
            finally:
                with self.lock:
                    self.active[provider] -= 1


class DummyOpenRouter:
    """A fast randomized valid-response source for end-to-end batch validation."""

    def __init__(self, seed: int) -> None:
        self.random = random.Random(seed)
        self.lock = threading.Lock()
        self.calls = 0

    def __call__(self, payload: dict[str, Any], _api_key: str, **_kwargs: Any) -> dict[str, Any]:
        with self.lock:
            self.calls += 1
            call = self.calls
            delay = self.random.uniform(0.001, 0.012)
            action_roll = self.random.random()
            count = 0 if action_roll < 0.18 else self.random.randint(1, 8)
            coordinates = self.random.sample(range(32 * 32), count)
            colors = ["red", "green", "black", "yellow", "orange", "blue", "purple", "pink"]
            pixels = [
                {"x": coordinate % 32, "y": coordinate // 32, "color": self.random.choice(colors)}
                for coordinate in coordinates
            ]
            forum_post = count > 0 and action_roll > 0.94
        time.sleep(delay)
        if forum_post:
            body = {"action": "write_team_forum", "pixels": [], "message": f"Dummy coordination post {call}."}
        else:
            body = {"action": "color", "pixels": pixels, "message": ""}
        provider_data = payload.get("provider", {})
        order = provider_data.get("order", []) if isinstance(provider_data, dict) else []
        return {
            "id": f"dummy-{call:06d}",
            "model": payload.get("model", "dummy"),
            "provider": order[0] if order else "dummy",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cost": 0.0,
                "completion_tokens_details": {"reasoning_tokens": call % 11},
            },
            "choices": [{"message": {"content": json.dumps(body, separators=(",", ":"))}}],
        }


def _validate_config(config: dict[str, Any]) -> None:
    scope = config.get("batch_scope", {})
    repetitions = scope.get("repetitions_per_configuration")
    conditions = config.get("conditions")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions_per_configuration must be a positive integer")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("conditions must be a non-empty list")
    if scope.get("unit") != f"{repetitions * len(conditions)} complete runs":
        raise ValueError("batch_scope.unit does not match conditions × repetitions")

    names = [condition.get("name") for condition in conditions if isinstance(condition, dict)]
    if len(names) != len(conditions) or len(set(names)) != len(names):
        raise ValueError("every condition needs a unique name")
    models = config.get("models", {})
    if models.get("allow_provider_fallbacks") is not False:
        raise ValueError("provider fallbacks must be disabled")
    model_names = {name for name, route in models.items() if name != "allow_provider_fallbacks" and isinstance(route, dict)}
    for name in model_names:
        route = models[name]
        if not isinstance(route.get("model"), str) or not isinstance(route.get("provider"), str):
            raise ValueError(f"model and provider are required for {name}")
    for condition in conditions:
        members = condition.get("members")
        if not isinstance(members, list) or not members or any(member not in model_names for member in members):
            raise ValueError(f"condition {condition.get('name')} contains an unknown or empty member list")

    task = config.get("task", {})
    expected_task = {
        "object": "apple", "canvas_size": 32, "maximum_artwork_size": 16,
        "attach_reference_image": True, "initial_canvas": "blank", "placement": "unrestricted",
    }
    if any(task.get(key) != value for key, value in expected_task.items()):
        raise ValueError("task settings differ from the registered apple design")

    protocol = config.get("protocol", {})
    if protocol.get("prompt_id") not in {"forum_optional_v1", "forum_suggestions_v2a", "forum_protocol_v3a"}:
        raise ValueError("protocol.prompt_id must name one of the three registered prompt templates")
    if not isinstance(protocol.get("configured_rounds"), int) or protocol["configured_rounds"] < 1:
        raise ValueError("configured_rounds must be a positive integer")
    expected_protocol = {
        "maximum_pixels_per_turn": 8, "round_0_member_1_turn": True,
        "team_forum": True, "public_forum": False, "personal_edit_history": True,
        "stop_when_every_member_returns_empty": True, "stop_when_stalled": True,
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("protocol settings differ from the registered single-team design")

    generation = config.get("generation", {})
    expected_generation = {"reasoning": "low", "reasoning_budget": 1000, "temperature": 0.2, "maximum_output_tokens": 8192}
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        raise ValueError("generation settings differ from the registered batch")
    execution = config.get("execution", {})
    if not isinstance(execution.get("runs_at_once"), int) or execution["runs_at_once"] < 1:
        raise ValueError("runs_at_once must be positive")
    if not isinstance(execution.get("maximum_requests_per_provider"), int) or execution["maximum_requests_per_provider"] < 1:
        raise ValueError("maximum_requests_per_provider must be positive")
    expected_execution = {
        "agents_within_each_round": "parallel", "apply_responses": "completion_order",
        "configuration_order": "randomized", "randomization_seed": 20260912,
        "resume_interrupted_runs": True, "continue_batch_after_failed_run": True,
        "retry_policy": "existing_harness_policy",
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise ValueError("execution settings differ from the registered batch")

def _jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    repetitions = int(config["batch_scope"]["repetitions_per_configuration"])
    jobs: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        base_members = list(condition["members"])
        for repetition in range(1, repetitions + 1):
            members = list(base_members)
            if condition.get("rotate_member_1"):
                offset = (repetition - 1) % len(members)
                members = members[offset:] + members[:offset]
            condition_name = str(condition["name"])
            run_id_prefix = str(config.get("run_id_prefix", "drawing_single_apple"))
            jobs.append({
                "condition": condition_name,
                "repetition": repetition,
                "members": members,
                "leader": members[0],
                "run_id": f"{run_id_prefix}_{condition_name}_{repetition:02d}",
            })
    random.Random(int(config["execution"]["randomization_seed"])).shuffle(jobs)
    for order, job in enumerate(jobs, start=1):
        job["randomized_order"] = order
    return jobs


def _usage_cost(run: Path) -> float:
    usage = _read_json(run / "usage.json", {})
    cost = usage.get("cost") if isinstance(usage, dict) else None
    return float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0


def _analyze_run(run: Path) -> dict[str, Any]:
    history = _read_json(run / "history.json", [])
    history = history if isinstance(history, list) else []
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_agent: dict[int, Counter[str]] = defaultdict(Counter)
    for item in history:
        if not isinstance(item, dict):
            continue
        display_round = item.get("display_round", item.get("round"))
        if isinstance(display_round, int):
            by_round[display_round].append(item)
    round_rows: list[dict[str, Any]] = []
    for round_number in sorted(by_round):
        items = by_round[round_number]
        targets: list[tuple[int, int]] = []
        colors_by_coordinate: dict[tuple[int, int], set[str]] = defaultdict(set)
        changed = 0
        for item in items:
            agent = int(item.get("agent", 0))
            action_path = run / str(item.get("actions", ""))
            action_record = _read_json(action_path, {})
            actions = action_record.get("actions", []) if isinstance(action_record, dict) else []
            valid_actions = [action for action in actions if isinstance(action, dict)]
            for action in valid_actions:
                coordinate = (int(action["x"]), int(action["y"]))
                color = str(action["color"])
                targets.append(coordinate)
                colors_by_coordinate[coordinate].add(color)
            pixels_changed = int(item.get("pixels_changed", 0))
            changed += pixels_changed
            per_agent[agent]["turns"] += 1
            per_agent[agent]["pixels_returned"] += len(valid_actions)
            per_agent[agent]["pixels_changed"] += pixels_changed
            if str(action_record.get("action_kind", "")) != "color":
                per_agent[agent]["forum_actions"] += 1
        counts = Counter(targets)
        unique_targets = len(counts)
        shared_unique = sum(1 for count in counts.values() if count > 1)
        repeated_actions = sum(count - 1 for count in counts.values() if count > 1)
        round_rows.append({
            "round": round_number,
            "agents_responding": len(items),
            "pixel_actions": len(targets),
            "unique_target_coordinates": unique_targets,
            "shared_target_coordinates": shared_unique,
            "overlapping_actions": repeated_actions,
            "shared_coordinate_rate": shared_unique / unique_targets if unique_targets else 0.0,
            "overlapping_action_rate": repeated_actions / len(targets) if targets else 0.0,
            "different_color_conflicts": sum(1 for colors in colors_by_coordinate.values() if len(colors) > 1),
            "pixels_changed_after_application": changed,
        })
    with (run / "round_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(round_rows[0]) if round_rows else ["round"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(round_rows)
    result = _read_json(run / "result.json", {})
    metrics = {
        "run_id": run.name,
        "rounds": round_rows,
        "per_agent": {str(agent): dict(values) for agent, values in sorted(per_agent.items())},
        "overall": {
            "calls": len(history),
            "pixel_actions": sum(row["pixel_actions"] for row in round_rows),
            "shared_target_coordinates": sum(row["shared_target_coordinates"] for row in round_rows),
            "overlapping_actions": sum(row["overlapping_actions"] for row in round_rows),
            "different_color_conflicts": sum(row["different_color_conflicts"] for row in round_rows),
            "final_filled_pixels": result.get("filled") if isinstance(result, dict) else None,
            "reported_cost_usd": _usage_cost(run),
        },
        "definitions": {
            "shared_coordinate_rate": "unique coordinates targeted by at least two members divided by all unique coordinates targeted in that round",
            "overlapping_action_rate": "actions beyond the first action at a coordinate divided by all pixel actions in that round",
            "different_color_conflicts": "coordinates assigned more than one color by members responding from the same round-start canvas",
        },
    }
    _write_json(run / "run_metrics.json", metrics)
    return metrics


def _copy_run(run: Path, batch_root: Path, job: dict[str, Any]) -> Path:
    destination = batch_root / "runs" / job["condition"] / f"rep_{job['repetition']:02d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run, destination, dirs_exist_ok=True)
    return destination


def _mark_orphaned_run_interrupted(run: Path) -> None:
    state_path = run / "run_state.json"
    state = _read_json(state_path, {})
    if isinstance(state, dict) and state.get("status") == "running" and not (run / "result.json").is_file():
        state["status"] = "interrupted"
        state["error"] = "previous batch process ended before this run completed"
        state["updated_utc"] = _utc()
        _write_json(state_path, state)


def _write_summary(batch_root: Path, manifest: dict[str, Any], budget: BudgetTracker) -> None:
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run = ROOT / "runs" / job["run_id"]
        result = _read_json(run / "result.json", {})
        metrics = _read_json(run / "run_metrics.json", {})
        overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
        rows.append({
            "randomized_order": job["randomized_order"],
            "condition": job["condition"],
            "repetition": job["repetition"],
            "leader": job["leader"],
            "members": ",".join(job["members"]),
            "status": job.get("status", "pending"),
            "stop_reason": result.get("stop_reason") if isinstance(result, dict) else "",
            "calls": result.get("calls") if isinstance(result, dict) else "",
            "filled_pixels": result.get("filled") if isinstance(result, dict) else "",
            "team_forum_posts": result.get("forums", {}).get("team_posts", "") if isinstance(result, dict) else "",
            "pixel_actions": overall.get("pixel_actions", ""),
            "shared_target_coordinates": overall.get("shared_target_coordinates", ""),
            "overlapping_actions": overall.get("overlapping_actions", ""),
            "different_color_conflicts": overall.get("different_color_conflicts", ""),
            "reported_cost_usd": _usage_cost(run),
            "error": job.get("error", ""),
        })
    fields = list(rows[0])
    with (batch_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    totals = Counter(row["status"] for row in rows)
    _write_json(batch_root / "summary.json", {
        "updated_utc": _utc(),
        "statuses": dict(totals),
        "reported_batch_cost_usd": budget.refresh(),
        "runs": rows,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "single_v1_core.json"))
    parser.add_argument("--validate-only", action="store_true", help="validate files and settings without calling an API")
    parser.add_argument("--dummy", action="store_true", help="run every configured job against a local randomized fake API")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise RuntimeError(f"could not read batch configuration: {config_path}")
    _validate_config(config)
    jobs = _jobs(config)
    total_jobs = len(jobs)
    output_setting = Path(config["outputs"]["directory"])
    batch_root = output_setting if output_setting.is_absolute() else PROJECT_ROOT / output_setting
    dummy_api = None
    if args.dummy:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        batch_root = batch_root.with_name(f"{batch_root.name}_dummy_{stamp}")
        for job in jobs:
            job["run_id"] = f"drawing_dummy_{stamp}_{job['condition']}_{job['repetition']:02d}"
        dummy_api = DummyOpenRouter(int(config["execution"]["randomization_seed"]))
        agent_runner_module.chat_completion = dummy_api
    batch_root.mkdir(parents=True, exist_ok=True)
    terminal_log = batch_root / "terminal.log"
    tee_out, tee_err = Tee(sys.stdout, terminal_log), Tee(sys.stderr, terminal_log)
    sys.stdout, sys.stderr = tee_out, tee_err

    manifest_path = batch_root / "manifest.json"
    old_manifest = _read_json(manifest_path, {})
    old_by_id = {
        item["run_id"]: item
        for item in old_manifest.get("jobs", [])
        if isinstance(item, dict) and isinstance(item.get("run_id"), str)
    } if isinstance(old_manifest, dict) else {}
    for job in jobs:
        old = old_by_id.get(job["run_id"], {})
        if old.get("status") in {"complete", "complete_video_failed"}:
            job.update({key: value for key, value in old.items() if key in {"status", "started_utc", "finished_utc", "error", "video_error"}})
        else:
            job["status"] = "pending"
    manifest = {
        "schema_version": 1,
        "name": str(config.get("name", "Single-team apple batch")),
        "created_utc": old_manifest.get("created_utc", _utc()) if isinstance(old_manifest, dict) else _utc(),
        "updated_utc": _utc(),
        "config_path": str(config_path),
        "computed_complete_runs": len(jobs),
        "jobs": jobs,
    }
    _write_json(batch_root / "batch_config.json", config)
    _write_json(manifest_path, manifest)
    load_env(ROOT / ".env")
    budget = BudgetTracker([job["run_id"] for job in jobs], float(config["batch_scope"]["stop_batch_after_cost_usd"]))
    gate = ProviderGate(budget, int(config["execution"].get("maximum_requests_per_provider", 2)))
    if args.validate_only:
        print(f"Validated {total_jobs} runs; existing recorded cost ${budget.cost_usd:.4f}; no API calls made.", flush=True)
        return 0
    if not args.dummy and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is empty; add it to harness/.env")

    manifest_lock = threading.Lock()
    video_lock = threading.Lock()
    completed_counter = Counter()
    concurrency_lock = threading.Lock()
    active_runs = 0
    maximum_active_runs = 0

    def save_manifest() -> None:
        with manifest_lock:
            manifest["updated_utc"] = _utc()
            manifest["reported_batch_cost_usd"] = budget.refresh()
            _write_json(manifest_path, manifest)
            _write_summary(batch_root, manifest, budget)

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        nonlocal active_runs, maximum_active_runs
        position = job["randomized_order"]
        if job.get("status") in {"complete", "complete_video_failed"}:
            print(f"[{position:02d}/{total_jobs}] SKIP {job['condition']} rep {job['repetition']:02d}: already complete", flush=True)
            return job
        if budget.reached():
            job["status"] = "budget_not_started"
            save_manifest()
            return job
        job["status"] = "running"
        job["started_utc"] = _utc()
        with concurrency_lock:
            active_runs += 1
            maximum_active_runs = max(maximum_active_runs, active_runs)
        save_manifest()
        print(
            f"[{position:02d}/{total_jobs}] START {job['condition']} rep {job['repetition']:02d} "
            f"leader={job['leader']} members={','.join(job['members'])}",
            flush=True,
        )
        run = ROOT / "runs" / job["run_id"]
        models = config["models"]
        members = [
            {
                "model": models[name]["model"],
                "provider": models[name]["provider"],
                "reasoning": config["generation"]["reasoning"],
                "reasoning_max_tokens": config["generation"]["reasoning_budget"],
            }
            for name in job["members"]
        ]

        def on_step(event: dict[str, Any]) -> None:
            current_cost = budget.refresh()
            action = event.get("action_kind", "color")
            print(
                f"[{position:02d}/{total_jobs}] R{event.get('display_round')} M{event.get('agent')} "
                f"{action} returned={len(event.get('actions', []))} changed={event.get('changed')} "
                f"cost=${current_cost:.4f}",
                flush=True,
            )

        def on_retry(event: dict[str, Any]) -> None:
            delay = float(event.get("delay_seconds", 0))
            print(
                f"[{position:02d}/{total_jobs}] RETRY R{event.get('round')} M{event.get('member')} "
                f"{event.get('kind')} {event.get('retry')}/{event.get('max_retries')} delay={delay:g}s",
                flush=True,
            )

        try:
            if (run / "result.json").is_file():
                result = _read_json(run / "result.json", {})
            elif run.is_dir():
                _mark_orphaned_run_interrupted(run)
                result = resume_saved_run(
                    job["run_id"],
                    on_step=on_step,
                    on_retry=on_retry,
                    should_cancel=budget.reached,
                    request_guard=gate,
                )
            else:
                result = run_team_drawing(
                    subject=config["task"]["object"],
                    description=DEFAULT_OBJECTS["apple"]["description"],
                    reference_tag="apple" if config["task"]["attach_reference_image"] else None,
                    step_size=int(config["protocol"]["maximum_pixels_per_turn"]),
                    max_steps=int(config["protocol"]["configured_rounds"]),
                    size=int(config["task"]["canvas_size"]),
                    artwork_size=int(config["task"]["maximum_artwork_size"]),
                    center=None,
                    members=members,
                    temperature=float(config["generation"]["temperature"]),
                    max_tokens=int(config["generation"]["maximum_output_tokens"]),
                    personal_history_enabled=bool(config["protocol"]["personal_edit_history"]),
                    forum_enabled=bool(config["protocol"]["team_forum"]),
                    public_forum_enabled=bool(config["protocol"]["public_forum"]),
                    prompt_template_id=str(config["protocol"]["prompt_id"]),
                    opening_member_turns=1,
                    on_step=on_step,
                    on_retry=on_retry,
                    should_cancel=budget.reached,
                    run_id_override=job["run_id"],
                    request_guard=gate,
                )
            _analyze_run(run)
            snapshots = [run / item["snapshot"] for item in _read_json(run / "history.json", [])]
            video_error = None
            if config["outputs"].get("save_webm_video"):
                try:
                    with video_lock:
                        build_webm(
                            snapshots,
                            run / "video.webm",
                            frame_duration_seconds=float(config["outputs"]["video_frame_duration_seconds"]),
                        )
                except Exception as exc:
                    video_error = str(exc)
                    print(f"[{position:02d}/{total_jobs}] VIDEO WARNING: {video_error}", flush=True)
            destination = _copy_run(run, batch_root, job)
            job["status"] = "complete_video_failed" if video_error else "complete"
            job["finished_utc"] = _utc()
            job["stop_reason"] = result.get("stop_reason") if isinstance(result, dict) else None
            job["cost_usd"] = _usage_cost(run)
            job["batch_copy"] = str(destination)
            if video_error:
                job["video_error"] = video_error
            completed_counter["runs"] += 1
            print(
                f"[{position:02d}/{total_jobs}] DONE {job['condition']} rep {job['repetition']:02d} "
                f"calls={result.get('calls')} stop={result.get('stop_reason')} cost=${job['cost_usd']:.4f}",
                flush=True,
            )
        except BudgetReached as exc:
            job["status"] = "budget_stopped"
            job["error"] = str(exc)
            if run.is_dir():
                _copy_run(run, batch_root, job)
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["traceback"] = traceback.format_exc()
            print(f"[{position:02d}/{total_jobs}] FAILED: {exc}", flush=True)
            if run.is_dir():
                _copy_run(run, batch_root, job)
            if not config["execution"]["continue_batch_after_failed_run"]:
                raise
        finally:
            with concurrency_lock:
                active_runs -= 1
            save_manifest()
        return job

    print(
        f"Single-team batch: {total_jobs} complete runs, up to {config['execution']['runs_at_once']} active runs, "
        f"at most {gate.per_provider} requests/provider, budget ${budget.limit_usd:.2f}.",
        flush=True,
    )
    pending_jobs = [job for job in jobs if job.get("status") not in {"complete", "complete_video_failed"}]
    try:
        with ThreadPoolExecutor(max_workers=int(config["execution"]["runs_at_once"]), thread_name_prefix="single-run") as executor:
            futures = [executor.submit(run_job, job) for job in pending_jobs]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        print("Batch interrupted by user. Completed and partial runs are saved; rerun the command to resume.", flush=True)
    manifest["concurrency_observed"] = {
        "maximum_active_runs": maximum_active_runs,
        "maximum_requests_by_provider": dict(gate.maximum_active),
    }
    if dummy_api is not None:
        manifest["dummy_api_calls"] = dummy_api.calls
    save_manifest()
    print(
        f"Batch finished: {sum(job.get('status') == 'complete' for job in jobs)}/{total_jobs} complete; "
        f"cost=${budget.refresh():.4f}; max active runs={maximum_active_runs}; "
        f"provider peaks={dict(gate.maximum_active)}",
        flush=True,
    )
    return 0 if all(job.get("status") in {"complete", "complete_video_failed"} for job in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
