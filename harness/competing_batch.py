"""Run a registered three-condition, four-team collective-canvas batch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import threading
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import agent_runner as agent_runner_module
from .agent_runner import ROOT, resume_saved_run, run_competing_teams
from .openrouter_client import load_env
from .references import DEFAULT_OBJECTS, reference_rows
from .single_batch import (
    BudgetReached,
    BudgetTracker,
    DummyOpenRouter,
    ProviderGate,
    Tee,
    _mark_orphaned_run_interrupted,
    _read_json,
    _usage_cost,
    _utc,
    _write_json,
)

PROJECT_ROOT = ROOT.parent
PROMPTS = {
    "V1": ("V1 public forum availability", "forum_optional_v1"),
    "V2": ("V2a public forum recommendations", "forum_suggestions_v2a"),
    "V3": ("V3a explicit public protocol", "forum_protocol_v3a"),
}
MODEL_ROUTES = {
    "GLM": ("z-ai/glm-5.3-flash", "streamlake/fp8"),
    "Gemini": ("google/gemini-3.8-flash", "google-ai-studio/flex"),
    "Qwen": ("qwen/qwen3.8-27b", "reka/fp8"),
    "Luna": ("openai/gpt-5.6-luna", "openai/flex"),
}
OBJECTS = ["apple", "orange", "lemon", "eggplant"]
BASE_MEMBERS = ["GLM", "Gemini", "Qwen", "Luna"]
CONDITIONS = ("four_heterogeneous_teams", "all_gemini", "homogeneous_family_per_team")


def _validate_config(config: dict[str, Any]) -> None:
    scope = config.get("batch_scope", {})
    if scope.get("unit") != "15 competing matches" or scope.get("repetitions_per_condition") != 5:
        raise ValueError("the batch must contain exactly five repetitions of three competing conditions")
    condition_names = [item.get("name") for item in config.get("conditions", []) if isinstance(item, dict)]
    if condition_names != list(CONDITIONS):
        raise ValueError("conditions differ from the registered competing design")
    version = str(config.get("prompt_version", ""))
    if version not in PROMPTS:
        raise ValueError("prompt_version must be V1, V2, or V3")
    models = config.get("models", {})
    if models.get("allow_provider_fallbacks") is not False:
        raise ValueError("provider fallbacks must be disabled")
    for name, route in MODEL_ROUTES.items():
        if (models.get(name, {}).get("model"), models.get(name, {}).get("provider")) != route:
            raise ValueError(f"the {name} model/provider route differs from the registered design")
    task = config.get("task", {})
    expected_task = {
        "team_objects": OBJECTS,
        "canvas_size": 32,
        "maximum_artwork_size": 16,
        "attach_reference_images": True,
        "initial_canvas": "blank",
        "placement": "unrestricted",
        "teams": 4,
        "members_per_team": 4,
        "base_members": BASE_MEMBERS,
        "rotate_member_1": True,
    }
    if any(task.get(key) != value for key, value in expected_task.items()):
        raise ValueError("task or team composition differs from the registered competing design")
    protocol = config.get("protocol", {})
    expected_protocol = {
        "prompt_id": PROMPTS[version][1],
        "team_forum": True,
        "public_forum": True,
        "personal_edit_history": True,
        "round_0_member_1_turn": True,
        "configured_rounds": 20,
        "maximum_pixels_per_turn": 8,
        "stop_when_every_member_returns_empty": True,
        "stop_when_stalled": True,
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("protocol differs from the registered competing design")
    generation = config.get("generation", {})
    expected_generation = {
        "reasoning": "low",
        "reasoning_budget": 1000,
        "temperature": 0.2,
        "maximum_output_tokens": 8192,
    }
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        raise ValueError("generation settings differ from the registered design")
    execution = config.get("execution", {})
    expected_execution = {
        "runs_at_once": 5,
        "maximum_requests_per_provider": 100,
        "agents_within_each_round": "parallel",
        "apply_responses": "completion_order",
        "match_order": "randomized",
        "randomization_seed": 20260912,
        "resume_interrupted_runs": True,
        "continue_batch_after_failed_run": True,
        "retry_policy": "existing_harness_policy",
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise ValueError("execution settings differ from the registered competing design")
    for tag in OBJECTS:
        if tag not in DEFAULT_OBJECTS:
            raise ValueError(f"missing object definition: {tag}")
        reference_rows(tag, 16)


def _jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    version = str(config["prompt_version"])
    jobs: list[dict[str, Any]] = []
    for repetition in range(1, 6):
        heterogeneous_teams = []
        for team_index, subject in enumerate(OBJECTS, start=1):
            offset = (repetition + team_index - 2) % len(BASE_MEMBERS)
            members = BASE_MEMBERS[offset:] + BASE_MEMBERS[:offset]
            heterogeneous_teams.append({"subject": subject, "members": members, "leader": members[0]})
        jobs.append({
            "condition": "four_heterogeneous_teams",
            "repetition": repetition,
            "teams": heterogeneous_teams,
            "leaders": {team["subject"]: team["leader"] for team in heterogeneous_teams},
            "run_id": f"drawing_competing_{version.lower()}_{repetition:02d}",
        })

        gemini_teams = [
            {"subject": subject, "members": ["Gemini"] * 4, "leader": "Gemini"}
            for subject in OBJECTS
        ]
        jobs.append({
            "condition": "all_gemini",
            "repetition": repetition,
            "teams": gemini_teams,
            "leaders": {team["subject"]: team["leader"] for team in gemini_teams},
            "run_id": f"drawing_competing_{version.lower()}_all_gemini_{repetition:02d}",
        })

        homogeneous_teams = []
        for team_index, subject in enumerate(OBJECTS, start=1):
            family = BASE_MEMBERS[(repetition + team_index - 2) % len(BASE_MEMBERS)]
            homogeneous_teams.append({"subject": subject, "members": [family] * 4, "leader": family})
        jobs.append({
            "condition": "homogeneous_family_per_team",
            "repetition": repetition,
            "teams": homogeneous_teams,
            "leaders": {team["subject"]: team["leader"] for team in homogeneous_teams},
            "run_id": f"drawing_competing_{version.lower()}_homofamily_{repetition:02d}",
        })
    random.Random(int(config["execution"]["randomization_seed"])).shuffle(jobs)
    for order, job in enumerate(jobs, start=1):
        job["randomized_order"] = order
    return jobs


def _analyze_run(run: Path) -> dict[str, Any]:
    history = _read_json(run / "history.json", [])
    history = history if isinstance(history, list) else []
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_team: dict[int, Counter[str]] = defaultdict(Counter)
    for item in history:
        if isinstance(item, dict) and isinstance(item.get("display_round"), int):
            by_round[item["display_round"]].append(item)
    rounds = []
    for round_number in sorted(by_round):
        targets: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
        forum_actions = Counter()
        for item in by_round[round_number]:
            team = int(item.get("team", 0))
            per_team[team]["turns"] += 1
            per_team[team]["pixels_changed"] += int(item.get("pixels_changed", 0))
            action_kind = str(item.get("action_kind", "color"))
            if action_kind.startswith("write_"):
                forum_actions[action_kind] += 1
                per_team[team][action_kind] += 1
            for action in item.get("actions", []):
                if isinstance(action, dict) and {"x", "y", "color"} <= set(action):
                    targets[(int(action["x"]), int(action["y"]))].append((team, str(action["color"])))
                    per_team[team]["pixel_actions"] += 1
        cross_team = sum(1 for entries in targets.values() if len({team for team, _ in entries}) > 1)
        conflicting = sum(1 for entries in targets.values() if len({color for _, color in entries}) > 1)
        rounds.append({
            "round": round_number,
            "turns": len(by_round[round_number]),
            "pixel_actions": sum(len(entries) for entries in targets.values()),
            "unique_target_coordinates": len(targets),
            "cross_team_shared_coordinates": cross_team,
            "different_color_conflicts": conflicting,
            "team_forum_writes": forum_actions["write_team_forum"],
            "public_forum_writes": forum_actions["write_public_forum"],
        })
    result = _read_json(run / "result.json", {})
    metrics = {
        "run_id": run.name,
        "rounds": rounds,
        "per_team": {str(team): dict(values) for team, values in sorted(per_team.items())},
        "overall": {
            "calls": len(history),
            "pixel_actions": sum(row["pixel_actions"] for row in rounds),
            "cross_team_shared_coordinates": sum(row["cross_team_shared_coordinates"] for row in rounds),
            "different_color_conflicts": sum(row["different_color_conflicts"] for row in rounds),
            "team_forum_writes": sum(row["team_forum_writes"] for row in rounds),
            "public_forum_writes": sum(row["public_forum_writes"] for row in rounds),
            "final_filled_pixels": result.get("filled") if isinstance(result, dict) else None,
            "reported_cost_usd": _usage_cost(run),
        },
    }
    _write_json(run / "run_metrics.json", metrics)
    fields = list(rounds[0]) if rounds else ["round"]
    with (run / "round_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rounds)
    return metrics


def _copy_run(run: Path, batch_root: Path, condition: str, repetition: int) -> Path:
    destination = batch_root / "runs" / condition / f"rep_{repetition:02d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run, destination, dirs_exist_ok=True)
    return destination


def _update_saved_run_routes(run: Path, models: dict[str, Any]) -> None:
    """Apply an explicitly changed endpoint to an incomplete saved run before resume."""
    config_path = run / "config.json"
    saved = _read_json(config_path, {})
    if not isinstance(saved, dict):
        return
    provider_by_model = {
        str(record["model"]): str(record["provider"])
        for record in models.values()
        if isinstance(record, dict) and record.get("model") and record.get("provider")
    }
    changes = []
    for team in saved.get("teams", []):
        for member in team.get("members", []) if isinstance(team, dict) else []:
            model = str(member.get("model", ""))
            new_provider = provider_by_model.get(model)
            old_provider = member.get("provider")
            if new_provider and old_provider != new_provider:
                member["provider"] = new_provider
                changes.append({"model": model, "from": old_provider, "to": new_provider})
    if changes:
        saved.setdefault("provider_route_changes", []).append({"time_utc": _utc(), "changes": changes})
        _write_json(config_path, saved)


def _write_summary(batch_root: Path, manifest: dict[str, Any], budget: BudgetTracker) -> None:
    rows = []
    for job in manifest["jobs"]:
        run = ROOT / "runs" / job["run_id"]
        result = _read_json(run / "result.json", {})
        metrics = _read_json(run / "run_metrics.json", {})
        overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
        rows.append({
            "randomized_order": job["randomized_order"],
            "condition": job["condition"],
            "repetition": job["repetition"],
            "leaders": json.dumps(job["leaders"], separators=(",", ":")),
            "status": job.get("status", "pending"),
            "stop_reason": result.get("stop_reason", "") if isinstance(result, dict) else "",
            "rounds": result.get("configured_rounds_executed", "") if isinstance(result, dict) else "",
            "calls": result.get("calls", "") if isinstance(result, dict) else "",
            "filled_pixels": result.get("filled", "") if isinstance(result, dict) else "",
            "team_forum_posts": json.dumps(result.get("forums", {}).get("team_posts", {}), separators=(",", ":")) if isinstance(result, dict) else "",
            "public_forum_posts": result.get("forums", {}).get("public_posts", "") if isinstance(result, dict) else "",
            "cross_team_shared_coordinates": overall.get("cross_team_shared_coordinates", ""),
            "different_color_conflicts": overall.get("different_color_conflicts", ""),
            "reported_cost_usd": _usage_cost(run),
            "error": job.get("error", ""),
        })
    with (batch_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(batch_root / "summary.json", {
        "updated_utc": _utc(),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "reported_batch_cost_usd": budget.refresh(),
        "runs": rows,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--dummy-rounds", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise RuntimeError(f"could not read batch configuration: {config_path}")
    _validate_config(config)
    if args.dummy_rounds is not None:
        if not args.dummy or not 1 <= args.dummy_rounds <= 20:
            parser.error("--dummy-rounds requires --dummy and a value from 1 to 20")
        config["protocol"]["configured_rounds"] = args.dummy_rounds
    version = str(config["prompt_version"])
    jobs = _jobs(config)
    output = Path(config["outputs"]["directory"])
    batch_root = output if output.is_absolute() else PROJECT_ROOT / output
    dummy_api = None
    if args.dummy:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        batch_root = batch_root.with_name(f"{batch_root.name}_dummy_{stamp}")
        for job in jobs:
            job["run_id"] = (
                f"drawing_competing_dummy_{version.lower()}_{stamp}_"
                f"{job['condition']}_{job['repetition']:02d}"
            )
        dummy_api = DummyOpenRouter(int(config["execution"]["randomization_seed"]))
        agent_runner_module.chat_completion = dummy_api
    batch_root.mkdir(parents=True, exist_ok=True)
    tee_out, tee_err = Tee(sys.stdout, batch_root / "terminal.log"), Tee(sys.stderr, batch_root / "terminal.log")
    sys.stdout, sys.stderr = tee_out, tee_err
    manifest_path = batch_root / "manifest.json"
    old_manifest = _read_json(manifest_path, {})
    old_by_id = {item["run_id"]: item for item in old_manifest.get("jobs", []) if isinstance(item, dict) and item.get("run_id")} if isinstance(old_manifest, dict) else {}
    for job in jobs:
        old = old_by_id.get(job["run_id"], {})
        if old.get("status") == "complete":
            job.update({key: value for key, value in old.items() if key in {"status", "started_utc", "finished_utc", "cost_usd", "batch_copy"}})
        else:
            job["status"] = "pending"
    manifest = {
        "schema_version": 1,
        "name": f"{version} four-team public-forum batch",
        "created_utc": old_manifest.get("created_utc", _utc()) if isinstance(old_manifest, dict) else _utc(),
        "updated_utc": _utc(),
        "config_path": str(config_path),
        "jobs": jobs,
    }
    _write_json(batch_root / "batch_config.json", config)
    _write_json(manifest_path, manifest)
    load_env(ROOT / ".env")
    budget = BudgetTracker([job["run_id"] for job in jobs], float(config["batch_scope"]["stop_batch_after_cost_usd"]))
    gate = ProviderGate(budget, int(config["execution"]["maximum_requests_per_provider"]))
    if args.validate_only:
        print(f"Validated {version}: {len(jobs)} matches across {len(CONDITIONS)} conditions, 20 rounds; no API calls made.", flush=True)
        return 0
    if not args.dummy and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is empty; add it to harness/.env")

    manifest_lock = threading.Lock()
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
        if job.get("status") == "complete":
            print(f"[{position:02d}/{len(jobs)}] SKIP {job['condition']} rep {job['repetition']:02d}: already complete", flush=True)
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
        print(f"[{position:02d}/{len(jobs)}] START {job['condition']} rep {job['repetition']:02d} leaders={job['leaders']}", flush=True)
        run = ROOT / "runs" / job["run_id"]
        models = config["models"]
        teams = []
        for team in job["teams"]:
            members = [{
                "model": models[name]["model"],
                "provider": models[name]["provider"],
                "reasoning": config["generation"]["reasoning"],
                "reasoning_max_tokens": config["generation"]["reasoning_budget"],
            } for name in team["members"]]
            teams.append({
                "subject": team["subject"],
                "description": DEFAULT_OBJECTS[team["subject"]]["description"],
                "reference_tag": team["subject"] if config["task"]["attach_reference_images"] else None,
                "center": None,
                "members": members,
            })

        def on_step(event: dict[str, Any]) -> None:
            print(
                f"[{position:02d}/{len(jobs)}] R{event.get('display_round')} T{event.get('team')} M{event.get('member')} "
                f"{event.get('action_kind', 'color')} returned={len(event.get('actions', []))} "
                f"changed={event.get('changed')} cost=${budget.refresh():.4f}",
                flush=True,
            )

        def on_retry(event: dict[str, Any]) -> None:
            print(
                f"[{position:02d}/{len(jobs)}] RETRY R{event.get('round')} T{event.get('team')} M{event.get('member')} "
                f"{event.get('kind')} {event.get('retry')}/{event.get('max_retries')} "
                f"delay={float(event.get('delay_seconds', 0)):g}s",
                flush=True,
            )

        try:
            if (run / "result.json").is_file():
                result = _read_json(run / "result.json", {})
            elif run.is_dir():
                _update_saved_run_routes(run, models)
                _mark_orphaned_run_interrupted(run)
                result = resume_saved_run(job["run_id"], on_step=on_step, on_retry=on_retry, should_cancel=budget.reached, request_guard=gate)
            else:
                result = run_competing_teams(
                    teams=teams,
                    artwork_size=int(config["task"]["maximum_artwork_size"]),
                    step_size=int(config["protocol"]["maximum_pixels_per_turn"]),
                    max_steps=int(config["protocol"]["configured_rounds"]),
                    size=int(config["task"]["canvas_size"]),
                    temperature=float(config["generation"]["temperature"]),
                    max_tokens=int(config["generation"]["maximum_output_tokens"]),
                    personal_history_enabled=bool(config["protocol"]["personal_edit_history"]),
                    forum_enabled=True,
                    public_forum_enabled=True,
                    prompt_template_id=str(config["protocol"]["prompt_id"]),
                    opening_member_turns=1,
                    on_step=on_step,
                    on_retry=on_retry,
                    should_cancel=budget.reached,
                    run_id_override=job["run_id"],
                    request_guard=gate,
                )
            _analyze_run(run)
            destination = _copy_run(run, batch_root, str(job["condition"]), int(job["repetition"]))
            job.update(status="complete", finished_utc=_utc(), stop_reason=result.get("stop_reason"), cost_usd=_usage_cost(run), batch_copy=str(destination))
            print(f"[{position:02d}/{len(jobs)}] DONE {job['condition']} rep {job['repetition']:02d} calls={result.get('calls')} stop={result.get('stop_reason')} cost=${job['cost_usd']:.4f}", flush=True)
        except BudgetReached as exc:
            job.update(status="budget_stopped", error=str(exc))
            if run.is_dir():
                _copy_run(run, batch_root, str(job["condition"]), int(job["repetition"]))
        except Exception as exc:
            job.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            print(f"[{position:02d}/{len(jobs)}] FAILED {job['condition']}: {exc}", flush=True)
            if run.is_dir():
                _copy_run(run, batch_root, str(job["condition"]), int(job["repetition"]))
            if not config["execution"]["continue_batch_after_failed_run"]:
                raise
        finally:
            with concurrency_lock:
                active_runs -= 1
            save_manifest()
        return job

    print(f"{version} competing batch: {len(jobs)} matches across {len(CONDITIONS)} conditions, {config['execution']['runs_at_once']} active matches, budget ${budget.limit_usd:.2f}.", flush=True)
    pending = [job for job in jobs if job.get("status") != "complete"]
    try:
        with ThreadPoolExecutor(max_workers=int(config["execution"]["runs_at_once"]), thread_name_prefix=f"competing-{version.lower()}") as executor:
            futures = [executor.submit(run_job, job) for job in pending]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        print("Batch interrupted. Completed and partial matches are saved; rerun the command to resume.", flush=True)
    manifest["concurrency_observed"] = {"maximum_active_matches": maximum_active_runs, "maximum_requests_by_provider": dict(gate.maximum_active)}
    if dummy_api is not None:
        manifest["dummy_api_calls"] = dummy_api.calls
    save_manifest()
    complete = sum(job.get("status") == "complete" for job in jobs)
    print(f"Batch finished: {complete}/{len(jobs)} complete; cost=${budget.refresh():.4f}; provider peaks={dict(gate.maximum_active)}", flush=True)
    return 0 if complete == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
