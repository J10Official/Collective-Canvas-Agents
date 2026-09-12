"""Run and persist a configurable single-call or iterative vision drawing."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from .actions import parse_drawing_response, parse_forum_turn
from .grid_image import write_grid_png, write_pixel_art_png
from .openrouter_client import build_vision_request, chat_completion, request_preview, response_text
from .prompts import DEFAULT_PROMPT_TEMPLATE_ID, artwork_bounds, constructed_prompt, forum_response_schema, pixel_response_schema
from .references import write_reference_png
from .store import COLORS, CanvasError

ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sum_usage(responses: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}

    def add_numeric_tree(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                target[key] = target.get(key, 0) + value
            elif isinstance(value, dict):
                nested = target.setdefault(key, {})
                if isinstance(nested, dict):
                    add_numeric_tree(nested, value)

    for response in responses:
        usage = response.get("usage", {})
        if isinstance(usage, dict):
            add_numeric_tree(totals, usage)
    return totals


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _blank_rows(size: int) -> list[list[str | None]]:
    return [[None for _ in range(size)] for _ in range(size)]


def _within_bounds(action: dict[str, Any], bounds: tuple[int, int, int, int] | None) -> bool:
    if bounds is None:
        return True
    x_min, x_max, y_min, y_max = bounds
    return x_min <= action["x"] <= x_max and y_min <= action["y"] <= y_max


def _decode_agent_response(
    response: dict[str, Any],
    *,
    size: int,
    iterative: bool,
    forum_enabled: bool,
    public_forum_enabled: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    if forum_enabled:
        turn = parse_forum_turn(
            response_text(response),
            size,
            iterative=iterative,
            public_forum_enabled=public_forum_enabled,
        )
        return turn, turn["actions"], turn["complete"]
    actions, complete = parse_drawing_response(response_text(response), size, iterative=iterative)
    return {"kind": "color", "forum": None, "message": "", "actions": actions, "complete": complete}, actions, complete


def _archive_invalid_response(path: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path.replace(path.with_name(f"{path.stem}.invalid_{stamp}{path.suffix}"))


def _retry_logger(
    run: Path,
    tag: str,
    context: dict[str, Any],
    on_retry: Callable[[dict[str, Any]], None] | None,
) -> Callable[[dict[str, Any]], None]:
    """Persist every retry and forward a compact event to the live UI."""
    retry_path = run / "retries" / f"{tag}.jsonl"

    def report(event: dict[str, Any]) -> None:
        event = dict(event)
        invalid_response = event.pop("invalid_response", None)
        record = {"time_utc": _utc_timestamp(), **context, **event}
        retry_path.parent.mkdir(parents=True, exist_ok=True)
        with retry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if isinstance(invalid_response, dict):
            retry_number = int(record.get("retry", 1))
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            _write_json(
                run / "responses" / f"{tag}.invalid_output_{retry_number:02d}_{stamp}.json",
                invalid_response,
            )
        if on_retry is not None:
            on_retry(record)

    return report


def _read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _run_path(run_id: str) -> Path:
    if not isinstance(run_id, str) or not run_id.startswith("drawing_") or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id):
        raise CanvasError("invalid run id")
    run = ROOT / "runs" / run_id
    if not run.is_dir():
        raise CanvasError("run does not exist")
    return run


def _completed_round(history: list[dict[str, Any]], expected_agents: int) -> int:
    counts: dict[int, int] = {}
    for item in history:
        round_number = item.get("round")
        if isinstance(round_number, int):
            counts[round_number] = counts.get(round_number, 0) + 1
    completed = 0
    while counts.get(completed + 1) == expected_agents:
        completed += 1
    return completed


def _resume_context(run: Path, size: int, expected_agents: int) -> dict[str, Any]:
    raw_history = _read_json(run / "history.json", [])
    history = raw_history if isinstance(raw_history, list) else []
    last_round = _completed_round(history, expected_agents)
    history = [item for item in history if isinstance(item, dict) and item.get("round", 0) <= last_round]
    initial = _read_json(run / "initial_canvas.json", {})
    initial_rows = initial.get("pixels") if isinstance(initial, dict) else None
    if not isinstance(initial_rows, list) or len(initial_rows) != size or any(not isinstance(row, list) or len(row) != size for row in initial_rows):
        initial_rows = _blank_rows(size)
    rows = [row[:] for row in initial_rows]
    for item in history:
        for change in item.get("changed_actions", []):
            x, y, color = change.get("x"), change.get("y"), change.get("color")
            if isinstance(x, int) and isinstance(y, int) and 0 <= x < size and 0 <= y < size:
                rows[y][x] = None if color == "erase" else color
    return {
        "history": history,
        "rows": rows,
        "last_round": last_round,
        "frame_number": len(history),
        "total_changes": sum(int(item.get("pixels_changed", 0)) for item in history),
        "responses": [{"usage": item.get("usage", {})} for item in history],
    }


def _write_run_state(
    run: Path,
    *,
    mode: str,
    status: str,
    last_completed_round: int,
    rows: list[list[str | None]],
    error: str | None = None,
) -> None:
    state = {
        "version": 1,
        "mode": mode,
        "status": status,
        "last_completed_round": last_completed_round,
        "pixels": rows,
        "updated_utc": _utc_timestamp(),
    }
    if error:
        state["error"] = error
    _write_json(run / "run_state.json", state)


def resumable_run(run_id: str | None = None) -> dict[str, Any] | None:
    """Return the newest interrupted or intentionally paused configured-agent run."""
    runs_root = ROOT / "runs"
    candidates = [_run_path(run_id)] if run_id else sorted(
        (path for path in runs_root.glob("drawing_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run in candidates:
        config = _read_json(run / "config.json")
        if not isinstance(config, dict):
            continue
        if isinstance(config.get("teams"), list):
            expected_agents = sum(len(team.get("members", [])) for team in config["teams"] if isinstance(team, dict))
            mode = "competing"
        elif isinstance(config.get("members"), list):
            expected_agents = len(config["members"])
            mode = "team"
        else:
            continue
        max_steps = config.get("max_steps")
        if expected_agents < 1 or not isinstance(max_steps, int):
            continue
        history = _read_json(run / "history.json", [])
        history = history if isinstance(history, list) else []
        last_round = _completed_round(history, expected_agents)
        state = _read_json(run / "run_state.json", {})
        status = state.get("status") if isinstance(state, dict) else None
        result = _read_json(run / "result.json", {})
        stop_reason = result.get("stop_reason") if isinstance(result, dict) else None
        is_resumable = last_round < max_steps and (
            status in {"interrupted", "paused"}
            or (not (run / "result.json").exists() and status != "running")
            or stop_reason == "cancelled"
        )
        if is_resumable:
            partial_responses = len([
                path
                for path in (run / "responses").glob(f"round_{last_round + 1:03d}_*.json")
                if ".invalid_" not in path.name and isinstance(_read_json(path), dict)
            ])
            return {
                "available": True,
                "run_id": run.name,
                "mode": mode,
                "last_completed_round": last_round,
                "max_steps": max_steps,
                "calls": len([item for item in history if isinstance(item, dict) and item.get("round", 0) <= last_round]),
                "saved_next_round_responses": min(partial_responses, expected_agents),
                "missing_next_round_responses": max(expected_agents - partial_responses, 0),
                "status": status or ("paused" if stop_reason == "cancelled" else "interrupted"),
            }
    return None


def run_drawing(
    *,
    subject: str,
    step_size: int,
    max_steps: int,
    size: int,
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    reasoning: str,
    reasoning_max_tokens: int | None = None,
    description: str | None = None,
    reference_tag: str | None = None,
    initial_rows: list[list[str | None]] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_retry: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the experiment and retain every input, output, action, and snapshot."""
    if not 0 <= step_size <= size * size:
        raise CanvasError(f"step_size must be from 0 to {size * size}")
    if not 1 <= max_steps <= 256:
        raise CanvasError("max_steps must be from 1 to 256")
    rows = [row[:] for row in (initial_rows if initial_rows is not None else _blank_rows(size))]
    if len(rows) != size or any(len(row) != size for row in rows):
        raise CanvasError(f"initial canvas must be exactly {size}x{size}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id = f"drawing_{stamp}"
    run = ROOT / "runs" / run_id
    for folder in ("inputs", "prompts", "requests", "responses", "actions", "snapshots", "retries"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    reference_path = write_reference_png(run / f"reference_{reference_tag}.png", reference_tag, size) if reference_tag else None

    config = {
        "subject": subject,
        "canvas_size": size,
        "step_size": step_size,
        "max_steps": max_steps,
        "model": model,
        "provider": provider,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": reasoning,
        "reasoning_max_tokens": reasoning_max_tokens,
        "description": description,
        "reference_tag": reference_tag,
    }
    _write_json(run / "config.json", config)

    history: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    total_changes = 0
    step_limit = 1 if step_size == 0 else max_steps
    model_complete = False
    stalled = False
    actual_model, actual_provider = model, provider

    for step_number in range(1, step_limit + 1):
        tag = f"step_{step_number:03d}"
        prompt = constructed_prompt(size, subject, step_size, step_number, description=description, reference_tag=reference_tag, reference_name=subject)
        (run / "prompts" / f"{tag}.txt").write_text(prompt + "\n", encoding="utf-8")
        input_path = write_grid_png(run / "inputs" / f"{tag}.png", size, rows=rows, colors=COLORS)
        request = build_vision_request(
            prompt,
            input_path,
            pixel_response_schema(size, step_size),
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            reasoning_max_tokens=reasoning_max_tokens,
            reference_image_path=reference_path,
        )
        _write_json(run / "requests" / f"{tag}.json", request_preview(request, input_path, reference_path))
        request_started_utc = _utc_timestamp()
        request_started = perf_counter()
        response = chat_completion(
            request,
            os.environ.get("OPENROUTER_API_KEY", ""),
            validate_response=lambda candidate: parse_drawing_response(
                response_text(candidate), size, iterative=step_size > 0
            ),
            on_retry=_retry_logger(run, tag, {"step": step_number}, on_retry),
        )
        latency_seconds = perf_counter() - request_started
        response_received_utc = _utc_timestamp()
        responses.append(response)
        _write_json(run / "responses" / f"{tag}.json", response)
        actual_model = str(response.get("model", actual_model))
        actual_provider = str(response.get("provider", actual_provider))

        actions, model_complete = parse_drawing_response(response_text(response), size, iterative=step_size > 0)
        changed = 0
        for action in actions:
            x, y, color = action["x"], action["y"], action["color"]
            if rows[y][x] != color:
                changed += 1
            rows[y][x] = color
        stalled = step_size > 0 and not model_complete and changed == 0
        total_changes += changed
        action_record = {"step": step_number, "complete": model_complete, "changed": changed, "actions": actions}
        _write_json(run / "actions" / f"{tag}.json", action_record)
        snapshot_path = write_grid_png(run / "snapshots" / f"{tag}.png", size, rows=rows, colors=COLORS)
        history.append({
            "step": step_number,
            "prompt": f"prompts/{tag}.txt",
            "input": f"inputs/{tag}.png",
            "request": f"requests/{tag}.json",
            "response": f"responses/{tag}.json",
            "actions": f"actions/{tag}.json",
            "snapshot": f"snapshots/{tag}.png",
            "pixels_returned": len(actions),
            "pixels_changed": changed,
            "complete": model_complete,
            "request_started_utc": request_started_utc,
            "response_received_utc": response_received_utc,
            "latency_seconds": latency_seconds,
            "usage": response.get("usage", {}),
        })
        _write_json(run / "history.json", history)
        _write_json(run / "usage.json", _sum_usage(responses))
        if on_step is not None:
            on_step({"step": step_number, "actions": actions, "changed": changed, "complete": model_complete})
        if step_size == 0 or model_complete or stalled:
            break

    stop_reason = "single_call" if step_size == 0 else ("model_complete" if model_complete else ("stalled" if stalled else "max_steps"))
    filled = sum(value is not None for row in rows for value in row)
    output_path = write_pixel_art_png(run / "output.png", rows, COLORS)
    latest = ROOT / "artifacts" / "latest_agent_output.png"
    latest.parent.mkdir(exist_ok=True)
    shutil.copyfile(output_path, latest)
    result = {
        "run_id": run_id,
        "run_path": str(run),
        "steps": len(history),
        "step_size": step_size,
        "complete": bool(model_complete or step_size == 0),
        "stop_reason": stop_reason,
        "filled": filled,
        "changes": total_changes,
        "usage": _sum_usage(responses),
        "model": actual_model,
        "provider": actual_provider,
        "output_url": "/artifacts/latest_agent_output.png",
        "snapshot_urls": [f"/runs/{run_id}/{item['snapshot']}" for item in history],
    }
    _write_json(run / "result.json", result)
    return result


def run_team_drawing(
    *,
    subject: str,
    step_size: int,
    max_steps: int,
    size: int,
    members: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    artwork_size: int | None = None,
    center: tuple[int, int] | None = None,
    description: str | None = None,
    reference_tag: str | None = None,
    personal_history_enabled: bool = False,
    forum_enabled: bool = False,
    public_forum_enabled: bool = False,
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID,
    prompt_template_text: str | None = None,
    initial_rows: list[list[str | None]] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_retry: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    resume_run_id: str | None = None,
    opening_member_turns: int = 1,
    run_id_override: str | None = None,
    request_guard: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run simultaneous agents in synchronized rounds over one shared canvas."""
    if not 0 <= step_size <= size * size:
        raise CanvasError(f"step_size must be from 0 to {size * size}")
    if not 1 <= max_steps <= 256:
        raise CanvasError("max_steps must be from 1 to 256")
    if not 1 <= len(members) <= 5:
        raise CanvasError("a team must contain from 1 to 5 members")
    if opening_member_turns not in {0, 1}:
        raise CanvasError("opening_member_turns must be 0 or 1")
    if public_forum_enabled and not forum_enabled:
        raise CanvasError("public_forum_enabled requires forum_enabled")
    if forum_enabled and step_size == 0:
        raise CanvasError("forums require max pixels per round to be greater than zero")
    artwork_size = size if artwork_size is None else artwork_size
    if not 1 <= artwork_size <= size:
        raise CanvasError(f"artwork_size must be from 1 to {size}")
    try:
        assigned_bounds = artwork_bounds(size, artwork_size, center)
    except ValueError as exc:
        raise CanvasError(str(exc)) from exc
    rows = [row[:] for row in (initial_rows if initial_rows is not None else _blank_rows(size))]
    if len(rows) != size or any(len(row) != size for row in rows):
        raise CanvasError(f"initial canvas must be exactly {size}x{size}")

    if resume_run_id is not None and run_id_override is not None:
        raise CanvasError("resume_run_id and run_id_override cannot both be set")
    resuming = resume_run_id is not None
    if resuming:
        run_id = str(resume_run_id)
        run = _run_path(run_id)
    elif run_id_override is not None:
        run_id = str(run_id_override)
        if not run_id.startswith("drawing_") or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id
        ):
            raise CanvasError("invalid run id override")
        run = ROOT / "runs" / run_id
        if run.exists():
            raise CanvasError("run id override already exists")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_id = f"drawing_{stamp}"
        run = ROOT / "runs" / run_id
    for folder in ("inputs", "prompts", "requests", "responses", "actions", "snapshots", "retries"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    reference_path = write_reference_png(run / f"reference_{reference_tag}.png", reference_tag, artwork_size) if reference_tag else None

    normalized_members = []
    for index, member in enumerate(members, start=1):
        normalized_members.append({
            "id": index,
            "model": member["model"],
            "provider": member["provider"],
            "reasoning": member.get("reasoning", "disabled"),
            "reasoning_max_tokens": member.get("reasoning_max_tokens", 500),
        })
    config = {
        "subject": subject,
        "description": description,
        "reference_tag": reference_tag,
        "personal_history_enabled": personal_history_enabled,
        "forum_enabled": forum_enabled,
        "public_forum_enabled": public_forum_enabled,
        "prompt_template_id": prompt_template_id,
        "prompt_template": prompt_template_text,
        "canvas_size": size,
        "artwork_size": artwork_size,
        "center": None if center is None else {"x": center[0], "y": center[1]},
        "assigned_bounds": assigned_bounds,
        "team_size": len(normalized_members),
        "step_size": step_size,
        "max_steps": max_steps,
        "members": normalized_members,
        "parameters": {"temperature": temperature, "max_tokens": max_tokens},
        "opening_member_turns": opening_member_turns,
        "concurrency": "member 1 takes one ordinary opening turn; subsequent rounds use the same round-start image, concurrent requests, and completion-order application",
        "communication": "complete accessible forum history shown automatically; explicit write actions" if forum_enabled else "none",
    }
    if not resuming:
        _write_json(run / "config.json", config)
        _write_json(run / "initial_canvas.json", {"size": size, "pixels": rows})

    if resuming:
        restored = _resume_context(run, size, len(normalized_members))
        rows = restored["rows"]
        history = restored["history"]
        responses = restored["responses"]
        total_changes = restored["total_changes"]
        last_completed_round = restored["last_round"]
        frame_number = restored["frame_number"]
    else:
        history: list[dict[str, Any]] = []
        responses: list[dict[str, Any]] = []
        total_changes = 0
        last_completed_round = 0
        frame_number = 0
    round_limit = 1 if step_size == 0 else max_steps
    team_complete = False
    stalled = False
    cancelled = False
    rounds_executed = last_completed_round
    actual_members = {member["id"]: {"model": member["model"], "provider": member["provider"]} for member in normalized_members}
    personal_histories: dict[int, list[dict[str, Any]]] = {member["id"]: [] for member in normalized_members}
    forum_events = [item["forum_event"] for item in history if isinstance(item.get("forum_event"), dict)]
    team_forum = [event for event in forum_events if event.get("kind") == "write_team_forum"]
    public_forum = [event for event in forum_events if event.get("kind") == "write_public_forum"]
    forum_activities: dict[int, list[dict[str, Any]]] = {member["id"]: [] for member in normalized_members}
    for item in history:
        agent_id = item.get("agent")
        changes = item.get("changed_actions", [])
        if agent_id in personal_histories and changes:
            personal_histories[agent_id].append({"round": item.get("display_round", item["round"]), "changes": changes})
        event = item.get("forum_event")
        if agent_id in forum_activities and isinstance(event, dict) and str(event.get("kind", "")).startswith("write_"):
            forum_activities[agent_id].append(event)
    _write_json(run / "forums.json", {"team": team_forum, "public": public_forum if public_forum_enabled else None, "events": forum_events})
    _write_run_state(run, mode="team", status="running", last_completed_round=last_completed_round, rows=rows)

    def request_member(prepared: dict[str, Any]) -> dict[str, Any]:
        def validate(candidate: dict[str, Any]) -> None:
            _decode_agent_response(
                candidate,
                size=size,
                iterative=step_size > 0,
                forum_enabled=forum_enabled,
                public_forum_enabled=public_forum_enabled,
            )

        report_retry = _retry_logger(
            run,
            prepared["tag"],
            {
                "round": prepared["round"],
                "opening_turn": prepared.get("opening_turn", False),
                "team": 1,
                "member": prepared["agent"],
                "agent": prepared["agent"],
            },
            on_retry,
        )
        response_reused = False
        response = _read_json(prepared["response_path"]) if resuming else None
        if isinstance(response, dict):
            response_reused = True
            request_path = run / prepared["request_rel"]
            request_time = request_path.stat().st_mtime if request_path.exists() else prepared["response_path"].stat().st_mtime
            response_time = prepared["response_path"].stat().st_mtime
            request_started_utc = datetime.fromtimestamp(request_time, timezone.utc).isoformat().replace("+00:00", "Z")
            response_received_utc = datetime.fromtimestamp(response_time, timezone.utc).isoformat().replace("+00:00", "Z")
            latency_seconds = max(0.0, response_time - request_time)
        else:
            request_started_utc = _utc_timestamp()
            request_started = perf_counter()
            with request_guard(prepared["request"]) if request_guard is not None else nullcontext():
                response = chat_completion(
                    prepared["request"],
                    os.environ.get("OPENROUTER_API_KEY", ""),
                    validate_response=validate,
                    on_retry=report_retry,
                )
            latency_seconds = perf_counter() - request_started
            response_received_utc = _utc_timestamp()
            _write_json(prepared["response_path"], response)
        try:
            turn, returned_actions, complete = _decode_agent_response(
                response,
                size=size,
                iterative=step_size > 0,
                forum_enabled=forum_enabled,
                public_forum_enabled=public_forum_enabled,
            )
        except (CanvasError, RuntimeError) as exc:
            if not response_reused:
                raise
            report_retry({
                "kind": "invalid_output",
                "retry": 1,
                "max_retries": 3,
                "delay_seconds": 0.0,
                "message": str(exc),
            })
            _archive_invalid_response(prepared["response_path"])
            response_reused = False
            request_started_utc = _utc_timestamp()
            request_started = perf_counter()
            with request_guard(prepared["request"]) if request_guard is not None else nullcontext():
                response = chat_completion(
                    prepared["request"],
                    os.environ.get("OPENROUTER_API_KEY", ""),
                    validate_response=validate,
                    on_retry=report_retry,
                )
            latency_seconds = perf_counter() - request_started
            response_received_utc = _utc_timestamp()
            _write_json(prepared["response_path"], response)
            turn, returned_actions, complete = _decode_agent_response(
                response,
                size=size,
                iterative=step_size > 0,
                forum_enabled=forum_enabled,
                public_forum_enabled=public_forum_enabled,
            )
        actions = [action for action in returned_actions if _within_bounds(action, assigned_bounds)]
        rejected_actions = [action for action in returned_actions if not _within_bounds(action, assigned_bounds)]
        return {
            **prepared,
            "response": response,
            "actions": actions,
            "rejected_actions": rejected_actions,
            "complete": complete,
            "action_kind": turn["kind"],
            "forum": turn["forum"],
            "message": turn["message"],
            "request_started_utc": request_started_utc,
            "response_received_utc": response_received_utc,
            "latency_seconds": latency_seconds,
            "response_reused": response_reused,
        }

    opening_turn_done = any(item.get("opening_turn") is True for item in history)
    schedule: list[tuple[int, int, bool]] = []
    if opening_member_turns and not opening_turn_done:
        schedule.append((0, 0, True))
    schedule.extend((round_number, round_number, False) for round_number in range(last_completed_round + 1, round_limit + 1))

    for round_number, prompt_round_number, opening_turn in schedule:
        if not opening_turn:
            rounds_executed = round_number
        round_rows = [row[:] for row in rows]
        round_team_forum = [dict(post) for post in team_forum]
        round_public_forum = [dict(post) for post in public_forum]
        prepared_calls: list[dict[str, Any]] = []
        scheduled_members = normalized_members[:1] if opening_turn else normalized_members
        for member in scheduled_members:
            agent_id = member["id"]
            tag = f"round_{round_number:03d}_agent_{agent_id:02d}"
            prompt = constructed_prompt(
                size,
                subject,
                step_size,
                prompt_round_number,
                team_size=len(normalized_members),
                member_index=agent_id,
                artwork_size=artwork_size,
                center=center,
                description=description,
                reference_tag=reference_tag,
                reference_name=subject,
                personal_history=personal_histories[agent_id] if personal_history_enabled else None,
                forum_enabled=forum_enabled,
                public_forum_enabled=public_forum_enabled,
                forum_posts={"team": round_team_forum, "public": round_public_forum},
                forum_activity=forum_activities[agent_id],
                prompt_template_id=prompt_template_id,
                prompt_template_text=prompt_template_text,
            )
            prompt_path = run / "prompts" / f"{tag}.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            input_path = write_grid_png(run / "inputs" / f"{tag}.png", size, rows=round_rows, colors=COLORS)
            request = build_vision_request(
                prompt,
                input_path,
                forum_response_schema(size, step_size, public_forum_enabled, assigned_bounds)
                if forum_enabled
                else pixel_response_schema(size, step_size, assigned_bounds),
                model=member["model"],
                provider=member["provider"],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=member["reasoning"],
                reasoning_max_tokens=member["reasoning_max_tokens"],
                reference_image_path=reference_path,
            )
            request_path = run / "requests" / f"{tag}.json"
            response_path = run / "responses" / f"{tag}.json"
            if not (resuming and response_path.is_file()):
                _write_json(request_path, request_preview(request, input_path, reference_path))
            prepared_calls.append({
                "agent": agent_id,
                "round": round_number,
                "display_round": prompt_round_number,
                "opening_turn": opening_turn,
                "tag": tag,
                "request": request,
                "response_path": response_path,
                "prompt_rel": f"prompts/{tag}.txt",
                "input_rel": f"inputs/{tag}.png",
                "request_rel": f"requests/{tag}.json",
                "response_rel": f"responses/{tag}.json",
            })

        completed_results: list[dict[str, Any]] = []
        try:
            with ThreadPoolExecutor(max_workers=len(prepared_calls), thread_name_prefix="canvas-agent") as executor:
                futures = {executor.submit(request_member, prepared): prepared["agent"] for prepared in prepared_calls}
                for future in as_completed(futures):
                    completed_results.append(future.result())
        except Exception as exc:
            _write_run_state(
                run,
                mode="team",
                status="interrupted",
                last_completed_round=last_completed_round,
                rows=rows,
                error=str(exc),
            )
            raise
        completed_results.sort(key=lambda item: item["response_received_utc"])

        round_changes = 0
        round_forum_actions = 0
        round_completions: list[bool] = []
        for completion_order, completed in enumerate(completed_results, start=1):
            frame_number += 1
            agent_id = completed["agent"]
            response = completed["response"]
            actions = completed["actions"]
            complete = completed["complete"]
            action_kind, forum_target, forum_message = completed["action_kind"], completed["forum"], completed["message"]
            responses.append(response)
            actual_members[agent_id] = {
                "model": str(response.get("model", actual_members[agent_id]["model"])),
                "provider": str(response.get("provider", actual_members[agent_id]["provider"])),
            }
            changed = 0
            changed_actions: list[dict[str, Any]] = []
            for action in actions:
                x, y, color = action["x"], action["y"], action["color"]
                if rows[y][x] != color:
                    changed += 1
                    changed_actions.append({
                        "x": x,
                        "y": y,
                        "color": "erase" if color is None else color,
                    })
                rows[y][x] = color
            if changed_actions:
                personal_histories[agent_id].append({"round": prompt_round_number, "changes": changed_actions})
            total_changes += changed
            round_changes += changed
            forum_event = None
            if forum_enabled and action_kind == "color":
                forum_event = {
                    "round": prompt_round_number,
                    "team": 1,
                    "member": agent_id,
                    "agent": agent_id,
                    "kind": "color",
                    "forum": None,
                    "message": "",
                    "changed": changed,
                }
                forum_events.append(forum_event)
            elif action_kind.startswith("write_"):
                forum_event = {
                    "round": prompt_round_number,
                    "team": 1,
                    "member": agent_id,
                    "agent": agent_id,
                    "kind": action_kind,
                    "forum": forum_target,
                    "message": forum_message,
                }
                (team_forum if forum_target == "team" else public_forum).append(forum_event)
                forum_activities[agent_id].append(forum_event)
                forum_events.append(forum_event)
                round_forum_actions += 1
            round_completions.append(complete)

            action_record = {
                "round": round_number,
                "display_round": prompt_round_number,
                "opening_turn": opening_turn,
                "agent": agent_id,
                "completion_order": completion_order,
                "complete": complete,
                "action_kind": action_kind,
                "forum": forum_target,
                "message": forum_message,
                "forum_event": forum_event,
                "changed": changed,
                "changed_actions": changed_actions,
                "actions": actions,
                "rejected_actions": completed["rejected_actions"],
            }
            _write_json(run / "actions" / f"{completed['tag']}.json", action_record)
            snapshot_name = f"frame_{frame_number:03d}_round_{round_number:03d}_agent_{agent_id:02d}.png"
            write_grid_png(run / "snapshots" / snapshot_name, size, rows=rows, colors=COLORS)
            history_item = {
                "round": round_number,
                "display_round": prompt_round_number,
                "opening_turn": opening_turn,
                "agent": agent_id,
                "completion_order": completion_order,
                "prompt": completed["prompt_rel"],
                "input": completed["input_rel"],
                "request": completed["request_rel"],
                "response": completed["response_rel"],
                "actions": f"actions/{completed['tag']}.json",
                "snapshot": f"snapshots/{snapshot_name}",
                "pixels_returned": len(actions) + len(completed["rejected_actions"]),
                "pixels_rejected_outside_region": len(completed["rejected_actions"]),
                "pixels_changed": changed,
                "changed_actions": changed_actions,
                "complete": complete,
                "request_started_utc": completed["request_started_utc"],
                "response_received_utc": completed["response_received_utc"],
                "latency_seconds": completed["latency_seconds"],
                "response_reused": completed["response_reused"],
                "usage": response.get("usage", {}),
            }
            history.append(history_item)
            _write_json(run / "history.json", history)
            _write_json(run / "usage.json", _sum_usage(responses))
            _write_json(run / "forums.json", {"team": team_forum, "public": public_forum if public_forum_enabled else None, "events": forum_events})
            if on_step is not None:
                on_step({
                    "round": round_number,
                    "display_round": prompt_round_number,
                    "opening_turn": opening_turn,
                    "agent": agent_id,
                    "completion_order": completion_order,
                    "actions": actions,
                    "changed": changed,
                    "complete": complete,
                    "action_kind": action_kind,
                    "forum": forum_target,
                    "message": forum_message,
                    "forum_event": forum_event,
                    "tag": completed["tag"],
                    "response_reused": completed["response_reused"],
                    "usage": response.get("usage", {}),
                })

        if opening_turn:
            opening_turn_done = True
            _write_run_state(run, mode="team", status="running", last_completed_round=last_completed_round, rows=rows)
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            continue

        last_completed_round = round_number
        _write_run_state(run, mode="team", status="running", last_completed_round=last_completed_round, rows=rows)

        team_complete = all(round_completions)
        stalled = step_size > 0 and not team_complete and round_changes == 0 and round_forum_actions == 0
        if step_size == 0 or team_complete or stalled:
            break
        if should_cancel is not None and should_cancel():
            cancelled = True
            break

    stop_reason = "single_round" if step_size == 0 else ("team_complete" if team_complete else ("stalled" if stalled else ("cancelled" if cancelled else "max_steps")))
    filled = sum(value is not None for row in rows for value in row)
    _write_json(run / "final_canvas.json", {"size": size, "pixels": rows})
    output_path = write_pixel_art_png(run / "output.png", rows, COLORS)
    latest = ROOT / "artifacts" / "latest_agent_output.png"
    latest.parent.mkdir(exist_ok=True)
    shutil.copyfile(output_path, latest)
    result = {
        "run_id": run_id,
        "run_path": str(run),
        "team_size": len(normalized_members),
        "rounds": rounds_executed + int(opening_turn_done),
        "steps": rounds_executed + int(opening_turn_done),
        "configured_rounds_executed": rounds_executed,
        "opening_turns": int(opening_turn_done),
        "calls": len(history),
        "step_size": step_size,
        "artwork_size": artwork_size,
        "center": center,
        "assigned_bounds": assigned_bounds,
        "complete": bool(team_complete or step_size == 0),
        "cancelled": cancelled,
        "stop_reason": stop_reason,
        "filled": filled,
        "changes": total_changes,
        "forums": {
            "enabled": forum_enabled,
            "public_enabled": public_forum_enabled,
            "team_posts": len(team_forum),
            "public_posts": len(public_forum),
            "actions": len(forum_events),
        },
        "forum_events": forum_events,
        "usage": _sum_usage(responses),
        "members": [{"id": member_id, **actual_members[member_id]} for member_id in sorted(actual_members)],
        "output_url": "/artifacts/latest_agent_output.png",
        "snapshot_urls": [f"/runs/{run_id}/{item['snapshot']}" for item in history],
    }
    _write_json(run / "result.json", result)
    _write_run_state(
        run,
        mode="team",
        status="paused" if cancelled else "complete",
        last_completed_round=last_completed_round,
        rows=rows,
    )
    return result


def run_competing_teams(
    *,
    teams: list[dict[str, Any]],
    artwork_size: int,
    step_size: int,
    max_steps: int,
    size: int,
    temperature: float,
    max_tokens: int,
    personal_history_enabled: bool = False,
    forum_enabled: bool = False,
    public_forum_enabled: bool = False,
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID,
    prompt_template_text: str | None = None,
    initial_rows: list[list[str | None]] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_retry: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    resume_run_id: str | None = None,
    run_id_override: str | None = None,
    opening_member_turns: int = 1,
    request_guard: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run several teams concurrently over one labeled shared canvas."""
    if not 2 <= len(teams) <= 5:
        raise CanvasError("competing teams must contain from 2 to 5 teams")
    if not 1 <= artwork_size <= size:
        raise CanvasError(f"artwork_size must be from 1 to {size}")
    if not 0 <= step_size <= size * size:
        raise CanvasError(f"step_size must be from 0 to {size * size}")
    if not 1 <= max_steps <= 256:
        raise CanvasError("max_steps must be from 1 to 256")
    if public_forum_enabled and not forum_enabled:
        raise CanvasError("public_forum_enabled requires forum_enabled")
    if opening_member_turns not in {0, 1}:
        raise CanvasError("opening_member_turns must be 0 or 1")
    if forum_enabled and step_size == 0:
        raise CanvasError("forums require max pixels per round to be greater than zero")

    rows = [row[:] for row in (initial_rows if initial_rows is not None else _blank_rows(size))]
    if len(rows) != size or any(len(row) != size for row in rows):
        raise CanvasError(f"initial canvas must be exactly {size}x{size}")

    resuming = resume_run_id is not None
    if resuming:
        run_id = str(resume_run_id)
        run = _run_path(run_id)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_id = str(run_id_override) if run_id_override else f"drawing_{stamp}"
        run = ROOT / "runs" / run_id
    for folder in ("inputs", "prompts", "requests", "responses", "actions", "snapshots", "retries"):
        (run / folder).mkdir(parents=True, exist_ok=True)

    normalized_teams: list[dict[str, Any]] = []
    reference_paths: dict[int, Path | None] = {}
    next_agent_id = 1
    for team_index, team in enumerate(teams, start=1):
        subject = team["subject"]
        raw_center = team.get("center")
        if isinstance(raw_center, dict):
            raw_center = (raw_center.get("x"), raw_center.get("y"))
        center = None if raw_center is None else tuple(raw_center)
        try:
            assigned_bounds = artwork_bounds(size, artwork_size, center)
        except (TypeError, ValueError) as exc:
            raise CanvasError(f"team {team_index} {exc}") from exc
        members = []
        for member_index, member in enumerate(team["members"], start=1):
            members.append({
                "id": member_index,
                "agent_id": next_agent_id,
                "model": member["model"],
                "provider": member["provider"],
                "reasoning": member.get("reasoning", "disabled"),
                "reasoning_max_tokens": member.get("reasoning_max_tokens", 500),
            })
            next_agent_id += 1
        normalized = {
            "id": team_index,
            "subject": subject,
            "description": team.get("description"),
            "reference_tag": team.get("reference_tag"),
            "center": center,
            "assigned_bounds": assigned_bounds,
            "members": members,
        }
        normalized_teams.append(normalized)
        reference_tag = normalized["reference_tag"]
        reference_paths[team_index] = (
            write_reference_png(
                run / f"reference_team_{team_index:02d}_{reference_tag}.png",
                reference_tag,
                artwork_size,
            )
            if reference_tag
            else None
        )

    config = {
        "canvas_size": size,
        "artwork_size": artwork_size,
        "competing_teams": len(normalized_teams),
        "personal_history_enabled": personal_history_enabled,
        "forum_enabled": forum_enabled,
        "public_forum_enabled": public_forum_enabled,
        "prompt_template_id": prompt_template_id,
        "prompt_template": prompt_template_text,
        "step_size": step_size,
        "max_steps": max_steps,
        "teams": normalized_teams,
        "parameters": {"temperature": temperature, "max_tokens": max_tokens},
        "opening_member_turns": opening_member_turns,
        "concurrency": "member 1 of each team takes one ordinary opening turn; subsequent rounds use the same round-start image, concurrent requests, and completion-order application",
        "communication": "complete accessible forum history shown automatically; explicit write actions" if forum_enabled else "none",
    }
    if not resuming:
        _write_json(run / "config.json", config)
        _write_json(run / "initial_canvas.json", {"size": size, "pixels": rows})

    expected_agents = sum(len(team["members"]) for team in normalized_teams)
    if resuming:
        restored = _resume_context(run, size, expected_agents)
        rows = restored["rows"]
        history = restored["history"]
        responses = restored["responses"]
        total_changes = restored["total_changes"]
        last_completed_round = restored["last_round"]
        frame_number = restored["frame_number"]
    else:
        history: list[dict[str, Any]] = []
        responses: list[dict[str, Any]] = []
        total_changes = 0
        last_completed_round = 0
        frame_number = 0
    round_limit = 1 if step_size == 0 else max_steps
    all_complete = False
    stalled = False
    cancelled = False
    rounds_executed = last_completed_round
    personal_histories: dict[int, list[dict[str, Any]]] = {
        member["agent_id"]: []
        for team in normalized_teams
        for member in team["members"]
    }
    actual_members = {
        member["agent_id"]: {"model": member["model"], "provider": member["provider"]}
        for team in normalized_teams
        for member in team["members"]
    }
    forum_events = [item["forum_event"] for item in history if isinstance(item.get("forum_event"), dict)]
    team_forums: dict[int, list[dict[str, Any]]] = {team["id"]: [] for team in normalized_teams}
    public_forum: list[dict[str, Any]] = []
    forum_activities: dict[int, list[dict[str, Any]]] = {agent_id: [] for agent_id in actual_members}
    for item in history:
        agent_id = item.get("agent")
        changes = item.get("changed_actions", [])
        if agent_id in personal_histories and changes:
            personal_histories[agent_id].append({"round": item.get("display_round", item["round"]), "changes": changes})
        event = item.get("forum_event")
        if not isinstance(event, dict) or not str(event.get("kind", "")).startswith("write_"):
            continue
        if event.get("forum") == "public":
            public_forum.append(event)
        elif event.get("team") in team_forums:
            team_forums[event["team"]].append(event)
        if agent_id in forum_activities:
            forum_activities[agent_id].append(event)
    _write_json(run / "forums.json", {"teams": team_forums, "public": public_forum if public_forum_enabled else None, "events": forum_events})
    _write_run_state(run, mode="competing", status="running", last_completed_round=last_completed_round, rows=rows)

    def request_member(prepared: dict[str, Any]) -> dict[str, Any]:
        def validate(candidate: dict[str, Any]) -> None:
            _decode_agent_response(
                candidate,
                size=size,
                iterative=step_size > 0,
                forum_enabled=forum_enabled,
                public_forum_enabled=public_forum_enabled,
            )

        report_retry = _retry_logger(
            run,
            prepared["tag"],
            {
                "round": prepared["round"],
                "opening_turn": prepared.get("opening_turn", False),
                "team": prepared["team"],
                "member": prepared["member"],
                "agent": prepared["agent"],
            },
            on_retry,
        )
        response_reused = False
        response = _read_json(prepared["response_path"]) if resuming else None
        if isinstance(response, dict):
            response_reused = True
            request_path = run / prepared["request_rel"]
            request_time = request_path.stat().st_mtime if request_path.exists() else prepared["response_path"].stat().st_mtime
            response_time = prepared["response_path"].stat().st_mtime
            request_started_utc = datetime.fromtimestamp(request_time, timezone.utc).isoformat().replace("+00:00", "Z")
            response_received_utc = datetime.fromtimestamp(response_time, timezone.utc).isoformat().replace("+00:00", "Z")
            latency_seconds = max(0.0, response_time - request_time)
        else:
            request_started_utc = _utc_timestamp()
            request_started = perf_counter()
            with request_guard(prepared["request"]) if request_guard is not None else nullcontext():
                response = chat_completion(
                    prepared["request"],
                    os.environ.get("OPENROUTER_API_KEY", ""),
                    validate_response=validate,
                    on_retry=report_retry,
                )
            latency_seconds = perf_counter() - request_started
            response_received_utc = _utc_timestamp()
            _write_json(prepared["response_path"], response)
        try:
            turn, returned_actions, complete = _decode_agent_response(
                response,
                size=size,
                iterative=step_size > 0,
                forum_enabled=forum_enabled,
                public_forum_enabled=public_forum_enabled,
            )
        except (CanvasError, RuntimeError) as exc:
            if not response_reused:
                raise
            report_retry({
                "kind": "invalid_output",
                "retry": 1,
                "max_retries": 3,
                "delay_seconds": 0.0,
                "message": str(exc),
            })
            _archive_invalid_response(prepared["response_path"])
            response_reused = False
            request_started_utc = _utc_timestamp()
            request_started = perf_counter()
            with request_guard(prepared["request"]) if request_guard is not None else nullcontext():
                response = chat_completion(
                    prepared["request"],
                    os.environ.get("OPENROUTER_API_KEY", ""),
                    validate_response=validate,
                    on_retry=report_retry,
                )
            latency_seconds = perf_counter() - request_started
            response_received_utc = _utc_timestamp()
            _write_json(prepared["response_path"], response)
            turn, returned_actions, complete = _decode_agent_response(
                response,
                size=size,
                iterative=step_size > 0,
                forum_enabled=forum_enabled,
                public_forum_enabled=public_forum_enabled,
            )
        bounds = prepared["assigned_bounds"]
        actions = [action for action in returned_actions if _within_bounds(action, bounds)]
        rejected_actions = [action for action in returned_actions if not _within_bounds(action, bounds)]
        return {
            **prepared,
            "response": response,
            "actions": actions,
            "rejected_actions": rejected_actions,
            "complete": complete,
            "action_kind": turn["kind"],
            "forum": turn["forum"],
            "message": turn["message"],
            "request_started_utc": request_started_utc,
            "response_received_utc": response_received_utc,
            "latency_seconds": latency_seconds,
            "response_reused": response_reused,
        }

    opening_turn_done = any(item.get("opening_turn") is True for item in history)
    schedule: list[tuple[int, int, bool]] = []
    if opening_member_turns and not opening_turn_done:
        schedule.append((0, 0, True))
    schedule.extend((round_number, round_number, False) for round_number in range(last_completed_round + 1, round_limit + 1))

    for round_number, prompt_round_number, opening_turn in schedule:
        if not opening_turn:
            rounds_executed = round_number
        round_rows = [row[:] for row in rows]
        round_team_forums = {team_id: [dict(post) for post in posts] for team_id, posts in team_forums.items()}
        round_public_forum = [dict(post) for post in public_forum]
        prepared_calls: list[dict[str, Any]] = []
        for team in normalized_teams:
            team_id = team["id"]
            scheduled_members = team["members"][:1] if opening_turn else team["members"]
            for member in scheduled_members:
                member_id, agent_id = member["id"], member["agent_id"]
                tag = f"round_{round_number:03d}_team_{team_id:02d}_member_{member_id:02d}"
                prompt = constructed_prompt(
                    size,
                    team["subject"],
                    step_size,
                    prompt_round_number,
                    team_size=len(team["members"]),
                    member_index=member_id,
                    competing_teams=len(normalized_teams),
                    team_index=team_id,
                    team_objectives=[candidate["subject"] for candidate in normalized_teams],
                    artwork_size=artwork_size,
                    center=team["center"],
                    description=team["description"],
                    reference_tag=team["reference_tag"],
                    reference_name=team["subject"],
                    personal_history=personal_histories[agent_id] if personal_history_enabled else None,
                    forum_enabled=forum_enabled,
                    public_forum_enabled=public_forum_enabled,
                    forum_posts={"team": round_team_forums[team_id], "public": round_public_forum},
                    forum_activity=forum_activities[agent_id],
                    prompt_template_id=prompt_template_id,
                    prompt_template_text=prompt_template_text,
                )
                prompt_path = run / "prompts" / f"{tag}.txt"
                prompt_path.write_text(prompt + "\n", encoding="utf-8")
                input_path = write_grid_png(run / "inputs" / f"{tag}.png", size, rows=round_rows, colors=COLORS)
                reference_path = reference_paths[team_id]
                request = build_vision_request(
                    prompt,
                    input_path,
                    forum_response_schema(size, step_size, public_forum_enabled, team["assigned_bounds"])
                    if forum_enabled
                    else pixel_response_schema(size, step_size, team["assigned_bounds"]),
                    model=member["model"],
                    provider=member["provider"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=member["reasoning"],
                    reasoning_max_tokens=member["reasoning_max_tokens"],
                    reference_image_path=reference_path,
                )
                request_path = run / "requests" / f"{tag}.json"
                response_path = run / "responses" / f"{tag}.json"
                if not (resuming and response_path.is_file()):
                    _write_json(request_path, request_preview(request, input_path, reference_path))
                prepared_calls.append({
                    "agent": agent_id,
                    "team": team_id,
                    "member": member_id,
                    "round": round_number,
                    "display_round": prompt_round_number,
                    "opening_turn": opening_turn,
                    "tag": tag,
                    "request": request,
                    "response_path": response_path,
                    "prompt_rel": f"prompts/{tag}.txt",
                    "input_rel": f"inputs/{tag}.png",
                    "request_rel": f"requests/{tag}.json",
                    "response_rel": f"responses/{tag}.json",
                    "assigned_bounds": team["assigned_bounds"],
                })

        completed_results: list[dict[str, Any]] = []
        try:
            with ThreadPoolExecutor(max_workers=len(prepared_calls), thread_name_prefix="canvas-agent") as executor:
                futures = {executor.submit(request_member, prepared): prepared["agent"] for prepared in prepared_calls}
                for future in as_completed(futures):
                    completed_results.append(future.result())
        except Exception as exc:
            _write_run_state(
                run,
                mode="competing",
                status="interrupted",
                last_completed_round=last_completed_round,
                rows=rows,
                error=str(exc),
            )
            raise
        completed_results.sort(key=lambda item: item["response_received_utc"])

        round_changes = 0
        round_forum_actions = 0
        round_completions: list[bool] = []
        for completion_order, completed in enumerate(completed_results, start=1):
            frame_number += 1
            agent_id, team_id, member_id = completed["agent"], completed["team"], completed["member"]
            response, actions, complete = completed["response"], completed["actions"], completed["complete"]
            action_kind, forum_target, forum_message = completed["action_kind"], completed["forum"], completed["message"]
            responses.append(response)
            actual_members[agent_id] = {
                "model": str(response.get("model", actual_members[agent_id]["model"])),
                "provider": str(response.get("provider", actual_members[agent_id]["provider"])),
            }
            changed = 0
            changed_actions: list[dict[str, Any]] = []
            for action in actions:
                x, y, color = action["x"], action["y"], action["color"]
                if rows[y][x] != color:
                    changed += 1
                    changed_actions.append({"x": x, "y": y, "color": "erase" if color is None else color})
                rows[y][x] = color
            if changed_actions:
                personal_histories[agent_id].append({"round": prompt_round_number, "changes": changed_actions})
            total_changes += changed
            round_changes += changed
            forum_event = None
            if forum_enabled and action_kind == "color":
                forum_event = {
                    "round": prompt_round_number,
                    "team": team_id,
                    "member": member_id,
                    "agent": agent_id,
                    "kind": "color",
                    "forum": None,
                    "message": "",
                    "changed": changed,
                }
                forum_events.append(forum_event)
            elif action_kind.startswith("write_"):
                forum_event = {
                    "round": prompt_round_number,
                    "team": team_id,
                    "member": member_id,
                    "agent": agent_id,
                    "kind": action_kind,
                    "forum": forum_target,
                    "message": forum_message,
                }
                (team_forums[team_id] if forum_target == "team" else public_forum).append(forum_event)
                forum_activities[agent_id].append(forum_event)
                forum_events.append(forum_event)
                round_forum_actions += 1
            round_completions.append(complete)

            action_record = {
                "round": round_number,
                "display_round": prompt_round_number,
                "opening_turn": opening_turn,
                "team": team_id,
                "member": member_id,
                "agent": agent_id,
                "completion_order": completion_order,
                "complete": complete,
                "action_kind": action_kind,
                "forum": forum_target,
                "message": forum_message,
                "forum_event": forum_event,
                "changed": changed,
                "changed_actions": changed_actions,
                "actions": actions,
                "rejected_actions": completed["rejected_actions"],
            }
            _write_json(run / "actions" / f"{completed['tag']}.json", action_record)
            snapshot_name = f"frame_{frame_number:03d}_round_{round_number:03d}_team_{team_id:02d}_member_{member_id:02d}.png"
            write_grid_png(run / "snapshots" / snapshot_name, size, rows=rows, colors=COLORS)
            history_item = {
                **action_record,
                "prompt": completed["prompt_rel"],
                "input": completed["input_rel"],
                "request": completed["request_rel"],
                "response": completed["response_rel"],
                "actions_file": f"actions/{completed['tag']}.json",
                "snapshot": f"snapshots/{snapshot_name}",
                "pixels_returned": len(actions) + len(completed["rejected_actions"]),
                "pixels_rejected_outside_region": len(completed["rejected_actions"]),
                "pixels_changed": changed,
                "request_started_utc": completed["request_started_utc"],
                "response_received_utc": completed["response_received_utc"],
                "latency_seconds": completed["latency_seconds"],
                "response_reused": completed["response_reused"],
                "usage": response.get("usage", {}),
            }
            history.append(history_item)
            _write_json(run / "history.json", history)
            _write_json(run / "usage.json", _sum_usage(responses))
            _write_json(run / "forums.json", {"teams": team_forums, "public": public_forum if public_forum_enabled else None, "events": forum_events})
            if on_step is not None:
                on_step({
                    "round": round_number,
                    "display_round": prompt_round_number,
                    "opening_turn": opening_turn,
                    "team": team_id,
                    "member": member_id,
                    "agent": agent_id,
                    "completion_order": completion_order,
                    "actions": actions,
                    "changed": changed,
                    "complete": complete,
                    "action_kind": action_kind,
                    "forum": forum_target,
                    "message": forum_message,
                    "forum_event": forum_event,
                    "tag": completed["tag"],
                    "response_reused": completed["response_reused"],
                    "usage": response.get("usage", {}),
                })

        if opening_turn:
            opening_turn_done = True
            _write_run_state(run, mode="competing", status="running", last_completed_round=last_completed_round, rows=rows)
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            continue

        last_completed_round = round_number
        _write_run_state(run, mode="competing", status="running", last_completed_round=last_completed_round, rows=rows)

        all_complete = all(round_completions)
        stalled = step_size > 0 and not all_complete and round_changes == 0 and round_forum_actions == 0
        if step_size == 0 or all_complete or stalled:
            break
        if should_cancel is not None and should_cancel():
            cancelled = True
            break

    stop_reason = "single_round" if step_size == 0 else ("all_teams_complete" if all_complete else ("stalled" if stalled else ("cancelled" if cancelled else "max_steps")))
    filled = sum(value is not None for row in rows for value in row)
    _write_json(run / "final_canvas.json", {"size": size, "pixels": rows})
    output_path = write_pixel_art_png(run / "output.png", rows, COLORS)
    latest = ROOT / "artifacts" / "latest_agent_output.png"
    latest.parent.mkdir(exist_ok=True)
    shutil.copyfile(output_path, latest)
    result_teams = []
    for team in normalized_teams:
        result_teams.append({
            "id": team["id"],
            "subject": team["subject"],
            "center": team["center"],
            "assigned_bounds": team["assigned_bounds"],
            "members": [
                {"id": member["id"], "agent_id": member["agent_id"], **actual_members[member["agent_id"]]}
                for member in team["members"]
            ],
        })
    result = {
        "run_id": run_id,
        "run_path": str(run),
        "competing_teams": len(normalized_teams),
        "teams": result_teams,
        "rounds": rounds_executed + int(opening_turn_done),
        "steps": rounds_executed + int(opening_turn_done),
        "configured_rounds_executed": rounds_executed,
        "opening_turns_per_team": int(opening_turn_done),
        "calls": len(history),
        "step_size": step_size,
        "artwork_size": artwork_size,
        "complete": bool(all_complete or step_size == 0),
        "cancelled": cancelled,
        "stop_reason": stop_reason,
        "filled": filled,
        "changes": total_changes,
        "forums": {
            "enabled": forum_enabled,
            "public_enabled": public_forum_enabled,
            "team_posts": {str(team_id): len(posts) for team_id, posts in team_forums.items()},
            "public_posts": len(public_forum),
            "actions": len(forum_events),
        },
        "forum_events": forum_events,
        "usage": _sum_usage(responses),
        "output_url": "/artifacts/latest_agent_output.png",
        "snapshot_urls": [f"/runs/{run_id}/{item['snapshot']}" for item in history],
    }
    _write_json(run / "result.json", result)
    _write_run_state(
        run,
        mode="competing",
        status="paused" if cancelled else "complete",
        last_completed_round=last_completed_round,
        rows=rows,
    )
    return result


def resume_canvas(run_id: str) -> dict[str, Any]:
    """Load the last fully committed canvas for a resumable run."""
    info = resumable_run(run_id)
    if info is None:
        raise CanvasError("run is not resumable")
    run = _run_path(run_id)
    config = _read_json(run / "config.json")
    if not isinstance(config, dict) or not isinstance(config.get("canvas_size"), int):
        raise CanvasError("run configuration is invalid")
    if info["mode"] == "competing":
        expected_agents = sum(len(team.get("members", [])) for team in config["teams"] if isinstance(team, dict))
    else:
        expected_agents = len(config["members"])
    restored = _resume_context(run, config["canvas_size"], expected_agents)
    return {
        **info,
        "size": config["canvas_size"],
        "pixels": restored["rows"],
    }


def resume_saved_run(
    run_id: str,
    *,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_retry: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    request_guard: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Continue an interrupted or paused run in its original directory and configuration."""
    info = resumable_run(run_id)
    if info is None:
        raise CanvasError("run is not resumable")
    run = _run_path(run_id)
    config = _read_json(run / "config.json")
    if not isinstance(config, dict):
        raise CanvasError("run configuration is invalid")
    parameters = config.get("parameters", {})
    temperature = float(parameters.get("temperature", 0.2))
    max_tokens = int(parameters.get("max_tokens", 4096))
    common = {
        "artwork_size": int(config.get("artwork_size", config["canvas_size"])),
        "step_size": int(config["step_size"]),
        "max_steps": int(config["max_steps"]),
        "size": int(config["canvas_size"]),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "personal_history_enabled": bool(config.get("personal_history_enabled", False)),
        "forum_enabled": bool(config.get("forum_enabled", False)),
        "public_forum_enabled": bool(config.get("public_forum_enabled", False)),
        "opening_member_turns": int(config.get("opening_member_turns", 0)),
        "prompt_template_id": str(config.get("prompt_template_id", DEFAULT_PROMPT_TEMPLATE_ID)),
        "prompt_template_text": config.get("prompt_template"),
        "on_step": on_step,
        "on_retry": on_retry,
        "should_cancel": should_cancel,
        "request_guard": request_guard,
        "resume_run_id": run_id,
    }
    if info["mode"] == "competing":
        teams = []
        for team in config["teams"]:
            center = team.get("center")
            if isinstance(center, dict):
                center = (center.get("x"), center.get("y"))
            elif isinstance(center, list):
                center = tuple(center)
            teams.append({
                "subject": team["subject"],
                "description": team.get("description"),
                "reference_tag": team.get("reference_tag"),
                "center": center,
                "members": team["members"],
            })
        return run_competing_teams(teams=teams, **common)
    center = config.get("center")
    if isinstance(center, dict):
        center = (center.get("x"), center.get("y"))
    elif isinstance(center, list):
        center = tuple(center)
    return run_team_drawing(
        subject=config["subject"],
        members=config["members"],
        description=config.get("description"),
        reference_tag=config.get("reference_tag"),
        center=center,
        **common,
    )
