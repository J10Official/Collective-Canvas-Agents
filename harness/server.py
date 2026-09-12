"""Dependency-free local HTTP/SSE server for the collective canvas."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .agent_runner import resumable_run, resume_canvas, resume_saved_run, run_competing_teams, run_team_drawing
from .actions import normalize_action, parse_llm_response
from .grid_image import write_grid_png
from .openrouter_client import DEFAULT_MODEL, DEFAULT_PROVIDER, load_env
from .prompts import (
    DEFAULT_PROMPT_TEMPLATE_ID,
    constructed_prompt,
    prompt_template,
    prompt_template_records,
    validate_prompt_template,
)
from .references import DEFAULT_OBJECTS, available_objects, reference_exists, save_reference, write_reference_png
from .store import CanvasError, CanvasStore

ROOT = Path(__file__).resolve().parent
DEFAULT_STATE = ROOT / "data" / "canvas.json"
DEFAULT_RECORDING = ROOT / "artifacts" / "mapping_validation.webm"
INDEX = ROOT / "static" / "index.html"
GLM_MODEL = "z-ai/glm-5.3-flash"
GLM_PROVIDER = "relace/fp4"


def _normalize_center(value: Any, size: int, label: str = "center") -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        raise CanvasError(f"{label} must contain integer x and y coordinates or be null")
    x, y = value["x"], value["y"]
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, int)
        or not isinstance(y, int)
        or not 0 <= x < size
        or not 0 <= y < size
    ):
        raise CanvasError(f"{label} coordinates must be integers from 0 to {size - 1}")
    return x, y


class CanvasHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can allow two local servers to share this port and
    # randomly serve different code versions. Refuse duplicate backends.
    allow_reuse_address = False
    allow_reuse_port = False

    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class EventHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        channel: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.add(channel)
        return channel

    def unsubscribe(self, channel: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(channel)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
        for channel in subscribers:
            try:
                channel.put_nowait(event)
            except queue.Full:
                try:
                    channel.get_nowait()
                    channel.put_nowait({"type": "resync"})
                except (queue.Empty, queue.Full):
                    pass


def make_handler(store: CanvasStore, hub: EventHub, recording_path: Path = DEFAULT_RECORDING) -> type[BaseHTTPRequestHandler]:
    mutation_lock = threading.Lock()
    agent_run_lock = threading.Lock()
    active_runs_lock = threading.Lock()
    active_runs: dict[str, threading.Event] = {}

    def apply_pixel(action: dict[str, Any], source: str) -> dict[str, Any]:
        # Preserve state-version order in the event stream even under concurrent callers.
        with mutation_lock:
            event = store.set_pixel(action["x"], action["y"], action["color"], source)
            if event.get("changed", True):
                hub.publish(event)
            return event

    def apply_clear(source: str) -> dict[str, Any]:
        with mutation_lock:
            event = store.clear(source)
            hub.publish(event)
            return event

    def apply_snapshot(pixels: list[list[str | None]], size: int, source: str) -> dict[str, Any]:
        with mutation_lock:
            if store.width != size:
                store.resize(size, source)
            event = store.replace_pixels(pixels, source)
            hub.publish(event)
            return event

    class CanvasHandler(BaseHTTPRequestHandler):
        server_version = "CollectiveCanvas/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

        def _send_json(self, status: int, value: Any) -> None:
            body = json.dumps(value, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Any:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise CanvasError("invalid Content-Length") from exc
            if length <= 0 or length > 1_000_000:
                raise CanvasError("request body must contain at most 1 MB of JSON")
            try:
                return json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CanvasError("request body is not valid JSON") from exc

        def _read_bytes(self, maximum: int) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise CanvasError("invalid Content-Length") from exc
            if length <= 0 or length > maximum:
                raise CanvasError(f"request body must contain at most {maximum // 1_000_000} MB")
            return self.rfile.read(length)

        def do_GET(self) -> None:  # noqa: N802
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            if path == "/":
                body = INDEX.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                self._send_json(HTTPStatus.OK, store.snapshot())
            elif path == "/api/health":
                self._send_json(HTTPStatus.OK, {"ok": True, "version": store.snapshot()["version"]})
            elif path == "/api/resumable-run":
                self._send_json(HTTPStatus.OK, resumable_run() or {"available": False})
            elif path == "/api/prompt-templates":
                self._send_json(HTTPStatus.OK, {
                    "default_id": DEFAULT_PROMPT_TEMPLATE_ID,
                    "templates": prompt_template_records(),
                    "syntax": "Use {{variable}} placeholders and {{#condition}}...{{/condition}} conditional blocks.",
                })
            elif path == "/api/agent-defaults":
                load_env(ROOT / ".env")
                self._send_json(HTTPStatus.OK, {
                    "subject": "apple",
                    "description": DEFAULT_OBJECTS["apple"]["description"],
                    "objects": available_objects(),
                    "grid_size": store.width,
                    "reference_enabled": False,
                    "personal_history_enabled": False,
                    "forum_enabled": False,
                    "public_forum_enabled": False,
                    "competing_teams": 1,
                    "team_size": 2,
                    "artwork_size": min(16, store.width),
                    "step_size": 8,
                    "max_steps": 10,
                    "prompt_template_id": DEFAULT_PROMPT_TEMPLATE_ID,
                    "prompt": constructed_prompt(
                        store.width,
                        "apple",
                        8,
                        1,
                        team_size=2,
                        member_index=1,
                        artwork_size=min(16, store.width),
                    ),
                    "members": [
                        {"id": 1, "model": GLM_MODEL, "provider": GLM_PROVIDER, "reasoning": "low", "reasoning_max_tokens": 500},
                        {"id": 2, "model": GLM_MODEL, "provider": GLM_PROVIDER, "reasoning": "low", "reasoning_max_tokens": 500},
                    ],
                    "parameters": {"temperature": 0.2, "max_tokens": 4096},
                })
            elif path == "/api/grid-image":
                image = write_grid_png(ROOT / "artifacts" / f"empty_grid_{store.width}.png", store.width)
                self._send_file(image, "image/png")
            elif path == "/api/reference-image":
                query = parse_qs(parsed_url.query)
                tag = query.get("tag", [""])[0]
                try:
                    size = int(query.get("size", [str(store.width)])[0])
                    if not reference_exists(tag):
                        raise CanvasError("unknown reference tag")
                    image = write_reference_png(ROOT / "artifacts" / f"reference_{tag}_{size}.png", tag, size)
                    self._send_file(image, "image/png")
                except (ValueError, CanvasError) as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            elif path == "/events":
                self._serve_events()
            elif path == "/artifacts/latest_agent_output.png" and (ROOT / "artifacts" / "latest_agent_output.png").exists():
                self._send_file(ROOT / "artifacts" / "latest_agent_output.png", "image/png")
            elif path == "/artifacts/mapping_validation.webm" and recording_path.exists():
                self._send_file(recording_path, "video/webm")
            elif match := re.fullmatch(r"/runs/(drawing_[0-9TZ_]+)/snapshots/([A-Za-z0-9_-]+\.png)", path):
                asset = ROOT / "runs" / match.group(1) / "snapshots" / match.group(2)
                self._send_file(asset, "image/png") if asset.is_file() else self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            elif match := re.fullmatch(r"/runs/(drawing_[0-9TZ_]+)/video\.webm", path):
                asset = ROOT / "runs" / match.group(1) / "video.webm"
                self._send_file(asset, "video/webm") if asset.is_file() else self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _serve_events(self) -> None:
            channel = hub.subscribe()
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                initial = {"type": "snapshot", "state": store.snapshot()}
                self.wfile.write(f"data:{json.dumps(initial, separators=(',', ':'))}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        event = channel.get(timeout=15)
                        data = json.dumps(event, separators=(",", ":"))
                        self.wfile.write(f"data:{data}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b":keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError):
                pass
            finally:
                hub.unsubscribe(channel)

        def do_POST(self) -> None:  # noqa: N802
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            try:
                if path == "/api/recording":
                    if self.headers.get_content_type() != "video/webm":
                        raise CanvasError("recording must use Content-Type video/webm")
                    body = self._read_bytes(50_000_000)
                    recording_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = recording_path.with_suffix(".webm.tmp")
                    with temporary.open("wb") as handle:
                        handle.write(body)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, recording_path)
                    self._send_json(HTTPStatus.OK, {"saved": True, "bytes": len(body), "path": str(recording_path)})
                    return
                if path == "/api/run-video":
                    if self.headers.get_content_type() != "video/webm":
                        raise CanvasError("run video must use Content-Type video/webm")
                    run_id = parse_qs(parsed_url.query).get("run_id", [""])[0]
                    if not re.fullmatch(r"drawing_[0-9TZ_]+", run_id):
                        raise CanvasError("invalid run_id")
                    run = ROOT / "runs" / run_id
                    if not run.is_dir():
                        raise CanvasError("run_id does not exist")
                    body = self._read_bytes(50_000_000)
                    video = run / "video.webm"
                    temporary = run / "video.webm.tmp"
                    with temporary.open("wb") as handle:
                        handle.write(body)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, video)
                    self._send_json(HTTPStatus.OK, {"saved": True, "bytes": len(body), "video_url": f"/runs/{run_id}/video.webm"})
                    return
                payload = self._read_json()
                if path == "/api/cancel-run":
                    if not isinstance(payload, dict):
                        raise CanvasError("request must be an object")
                    run_token = payload.get("run_token")
                    if not isinstance(run_token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", run_token):
                        raise CanvasError("invalid run_token")
                    with active_runs_lock:
                        cancellation = active_runs.get(run_token)
                    if cancellation is not None:
                        cancellation.set()
                    self._send_json(HTTPStatus.OK, {
                        "cancel_requested": cancellation is not None,
                        "message": "The current round will finish; no later round will start." if cancellation is not None else "The run has already finished.",
                    })
                elif path == "/api/resume-run":
                    if not isinstance(payload, dict):
                        raise CanvasError("request must be an object")
                    requested_run_id = payload.get("run_id")
                    if requested_run_id is not None and (
                        not isinstance(requested_run_id, str)
                        or not re.fullmatch(r"drawing_[0-9TZ_]+", requested_run_id)
                    ):
                        raise CanvasError("invalid run_id")
                    info = resumable_run(requested_run_id)
                    if info is None:
                        raise CanvasError("there is no interrupted or paused run to resume")
                    checkpoint = resume_canvas(info["run_id"])
                    load_env(ROOT / ".env")
                    run_token = payload.get("run_token") or uuid.uuid4().hex
                    if not isinstance(run_token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", run_token):
                        raise CanvasError("run_token must contain 8 to 100 letters, numbers, underscores, or hyphens")
                    cancellation = threading.Event()
                    with active_runs_lock:
                        if run_token in active_runs:
                            raise CanvasError("run_token is already active")
                        active_runs[run_token] = cancellation
                    try:
                        with agent_run_lock:
                            apply_snapshot(checkpoint["pixels"], checkpoint["size"], "resume-checkpoint")

                            def on_resume_step(step: dict[str, Any]) -> None:
                                team_id = step.get("team", 1)
                                member_id = step.get("member", step["agent"])
                                for action in step["actions"]:
                                    apply_pixel(action, f"configured-team-{team_id}-member-{member_id}")
                                hub.publish({
                                    "type": "agent_step",
                                    "round": step["round"],
                                    "agent": step["agent"],
                                    "team": team_id,
                                    "member": member_id,
                                    "completion_order": step["completion_order"],
                                    "pixels": len(step["actions"]),
                                    "changed": step["changed"],
                                    "complete": step["complete"],
                                    "action_kind": step.get("action_kind", "color"),
                                    "forum": step.get("forum"),
                                    "message": step.get("message", ""),
                                    "forum_event": step.get("forum_event"),
                                })

                            def on_resume_retry(event: dict[str, Any]) -> None:
                                hub.publish({"type": "agent_retry", **event})

                            result = resume_saved_run(
                                info["run_id"],
                                on_step=on_resume_step,
                                on_retry=on_resume_retry,
                                should_cancel=cancellation.is_set,
                            )
                            result["run_token"] = run_token
                    finally:
                        with active_runs_lock:
                            active_runs.pop(run_token, None)
                    self._send_json(HTTPStatus.OK, result)
                elif path == "/api/store-reference":
                    if not isinstance(payload, dict):
                        raise CanvasError("request must be an object")
                    run_id = payload.get("run_id")
                    if not isinstance(run_id, str) or not re.fullmatch(r"drawing_[0-9TZ_]+", run_id):
                        raise CanvasError("invalid run_id")
                    run = ROOT / "runs" / run_id
                    config_path, canvas_path, result_path = run / "config.json", run / "final_canvas.json", run / "result.json"
                    if not config_path.is_file() or not canvas_path.is_file() or not result_path.is_file():
                        raise CanvasError("the completed run does not contain a reusable final canvas")
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    final_canvas = json.loads(canvas_path.read_text(encoding="utf-8"))
                    subject, description = config.get("subject"), config.get("description", "")
                    rows = final_canvas.get("pixels") if isinstance(final_canvas, dict) else None
                    if not isinstance(subject, str) or not isinstance(description, str) or not isinstance(rows, list):
                        raise CanvasError("the completed run metadata is invalid")
                    tag, object_record = save_reference(
                        subject=subject,
                        description=description,
                        rows=rows,
                        source_run_id=run_id,
                    )
                    self._send_json(HTTPStatus.OK, {"stored": True, "tag": tag, "object": object_record})
                elif path == "/api/resize":
                    if not isinstance(payload, dict):
                        raise CanvasError("request must be an object")
                    size = payload.get("size")
                    with agent_run_lock:
                        with mutation_lock:
                            event = store.resize(size, "ui-resize")
                            hub.publish(event)
                    self._send_json(HTTPStatus.OK, event)
                elif path == "/api/run-agent":
                    if not isinstance(payload, dict):
                        raise CanvasError("request must be an object")
                    personal_history_enabled = payload.get("personal_history_enabled", False)
                    forum_enabled = payload.get("forum_enabled", False)
                    public_forum_enabled = payload.get("public_forum_enabled", False)
                    step_size, max_steps = payload.get("step_size", 8), payload.get("max_steps", 10)
                    artwork_size = payload.get("artwork_size", min(16, store.width))
                    parameters = payload.get("parameters", {})
                    prompt_template_id = payload.get("prompt_template_id", DEFAULT_PROMPT_TEMPLATE_ID)
                    prompt_template_text = payload.get("prompt_template")
                    if not isinstance(personal_history_enabled, bool):
                        raise CanvasError("personal_history_enabled must be true or false")
                    if not isinstance(forum_enabled, bool):
                        raise CanvasError("forum_enabled must be true or false")
                    if not isinstance(public_forum_enabled, bool):
                        raise CanvasError("public_forum_enabled must be true or false")
                    if public_forum_enabled and not forum_enabled:
                        raise CanvasError("public_forum_enabled requires forum_enabled")
                    if forum_enabled and step_size == 0:
                        raise CanvasError("forums require max pixels per round to be greater than zero")
                    if isinstance(artwork_size, bool) or not isinstance(artwork_size, int) or not 1 <= artwork_size <= store.width:
                        raise CanvasError(f"artwork_size must be an integer from 1 to {store.width}")
                    if isinstance(step_size, bool) or not isinstance(step_size, int) or not 0 <= step_size <= store.width * store.height:
                        raise CanvasError(f"step_size must be an integer from 0 to {store.width * store.height}")
                    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 256:
                        raise CanvasError("max_steps must be an integer from 1 to 256")
                    if not isinstance(parameters, dict):
                        raise CanvasError("parameters must be an object")
                    if not isinstance(prompt_template_id, str) or not 1 <= len(prompt_template_id) <= 100:
                        raise CanvasError("prompt_template_id must contain from 1 to 100 characters")
                    try:
                        if prompt_template_text is None:
                            prompt_template_text = prompt_template(prompt_template_id)
                        validate_prompt_template(prompt_template_text)
                    except ValueError as exc:
                        raise CanvasError(str(exc)) from exc
                    temperature = parameters.get("temperature", 0.2)
                    max_tokens = parameters.get("max_tokens", 4096)
                    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
                        raise CanvasError("temperature must be from 0 to 2")
                    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 256 <= max_tokens <= 32_768:
                        raise CanvasError("max_tokens must be an integer from 256 to 32,768")
                    teams = payload.get("teams")
                    if teams is None:
                        legacy_members = payload.get("members") or [{
                            "model": payload.get("model", os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)),
                            "provider": payload.get("provider", os.environ.get("OPENROUTER_PROVIDER", DEFAULT_PROVIDER)),
                            "reasoning": parameters.get("reasoning", "disabled"),
                            "reasoning_max_tokens": parameters.get("reasoning_max_tokens", 500),
                        }]
                        teams = [{
                            "subject": payload.get("subject", "apple"),
                            "description": payload.get("description", ""),
                            "reference_tag": payload.get("reference_tag"),
                            "members": legacy_members,
                        }]
                    if not isinstance(teams, list) or not 1 <= len(teams) <= 5:
                        raise CanvasError("teams must contain from 1 to 5 team configurations")
                    normalized_teams = []
                    total_agents = 0
                    for team_index, team in enumerate(teams, start=1):
                        if not isinstance(team, dict):
                            raise CanvasError(f"team {team_index} must be an object")
                        subject = team.get("subject", "apple")
                        description = team.get("description", "")
                        reference_tag = team.get("reference_tag")
                        center = _normalize_center(team.get("center"), store.width, f"team {team_index} center")
                        members = team.get("members")
                        if not isinstance(subject, str) or not 1 <= len(subject.strip()) <= 100:
                            raise CanvasError(f"team {team_index} subject must contain 1 to 100 characters")
                        if not isinstance(description, str) or len(description) > 1_000:
                            raise CanvasError(f"team {team_index} description must contain at most 1,000 characters")
                        if reference_tag is not None and (not isinstance(reference_tag, str) or not reference_exists(reference_tag)):
                            raise CanvasError(f"team {team_index} reference_tag must identify an available reference or be null")
                        if not isinstance(members, list) or not 1 <= len(members) <= 5:
                            raise CanvasError(f"team {team_index} members must contain from 1 to 5 agent configurations")
                        normalized_members = []
                        for member_index, member in enumerate(members, start=1):
                            if not isinstance(member, dict):
                                raise CanvasError(f"team {team_index} member {member_index} must be an object")
                            model, provider = member.get("model"), member.get("provider")
                            reasoning = member.get("reasoning", "disabled")
                            reasoning_max_tokens = member.get("reasoning_max_tokens", 500)
                            if not isinstance(model, str) or not 1 <= len(model) <= 200 or any(char.isspace() for char in model):
                                raise CanvasError(f"team {team_index} member {member_index} model must be a non-empty model ID without spaces")
                            if not isinstance(provider, str) or not 1 <= len(provider) <= 200 or any(char.isspace() for char in provider):
                                raise CanvasError(f"team {team_index} member {member_index} provider must be a non-empty endpoint slug without spaces")
                            if reasoning not in {"disabled", "low", "high", "max"}:
                                raise CanvasError(f"team {team_index} member {member_index} reasoning must be disabled, low, high, or max")
                            if isinstance(reasoning_max_tokens, bool) or not isinstance(reasoning_max_tokens, int) or not 1 <= reasoning_max_tokens <= 32_768:
                                raise CanvasError(f"team {team_index} member {member_index} reasoning_max_tokens must be an integer from 1 to 32,768")
                            normalized_members.append({
                                "model": model,
                                "provider": provider,
                                "reasoning": reasoning,
                                "reasoning_max_tokens": reasoning_max_tokens,
                            })
                        total_agents += len(normalized_members)
                        normalized_teams.append({
                            "subject": subject.strip(),
                            "description": description.strip(),
                            "reference_tag": reference_tag,
                            "center": center,
                            "members": normalized_members,
                        })
                    if total_agents > 25:
                        raise CanvasError("an experiment may contain at most 25 agents")
                    load_env(ROOT / ".env")
                    run_token = payload.get("run_token") or uuid.uuid4().hex
                    if not isinstance(run_token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", run_token):
                        raise CanvasError("run_token must contain 8 to 100 letters, numbers, underscores, or hyphens")
                    cancellation = threading.Event()
                    with active_runs_lock:
                        if run_token in active_runs:
                            raise CanvasError("run_token is already active")
                        active_runs[run_token] = cancellation
                    try:
                        with agent_run_lock:
                            if payload.get("clear_first", True):
                                apply_clear("configured-agent")
                            initial_rows = store.snapshot()["pixels"]

                            def on_step(step: dict[str, Any]) -> None:
                                team_id = step.get("team", 1)
                                member_id = step.get("member", step["agent"])
                                for action in step["actions"]:
                                    apply_pixel(action, f"configured-team-{team_id}-member-{member_id}")
                                hub.publish({
                                    "type": "agent_step",
                                    "round": step["round"],
                                    "agent": step["agent"],
                                    "team": team_id,
                                    "member": member_id,
                                    "completion_order": step["completion_order"],
                                    "pixels": len(step["actions"]),
                                    "changed": step["changed"],
                                    "complete": step["complete"],
                                    "action_kind": step.get("action_kind", "color"),
                                    "forum": step.get("forum"),
                                    "message": step.get("message", ""),
                                    "forum_event": step.get("forum_event"),
                                })

                            def on_retry(event: dict[str, Any]) -> None:
                                hub.publish({"type": "agent_retry", **event})

                            common = {
                                "artwork_size": artwork_size,
                                "step_size": step_size,
                                "max_steps": max_steps,
                                "size": store.width,
                                "temperature": float(temperature),
                                "max_tokens": max_tokens,
                                "personal_history_enabled": personal_history_enabled,
                                "forum_enabled": forum_enabled,
                                "public_forum_enabled": public_forum_enabled,
                                "prompt_template_id": prompt_template_id,
                                "prompt_template_text": prompt_template_text,
                                "initial_rows": initial_rows,
                                "on_step": on_step,
                                "on_retry": on_retry,
                                "should_cancel": cancellation.is_set,
                            }
                            if len(normalized_teams) == 1:
                                team = normalized_teams[0]
                                result = run_team_drawing(
                                    subject=team["subject"],
                                    members=team["members"],
                                    description=team["description"],
                                    reference_tag=team["reference_tag"],
                                    center=team["center"],
                                    **common,
                                )
                                result["competing_teams"] = 1
                            else:
                                result = run_competing_teams(teams=normalized_teams, **common)
                            result["run_token"] = run_token
                    finally:
                        with active_runs_lock:
                            active_runs.pop(run_token, None)
                    self._send_json(HTTPStatus.OK, result)
                elif path == "/api/prompt-preview":
                    if not isinstance(payload, dict):
                        raise CanvasError("request must be an object")
                    subject, step_size = payload.get("subject", "apple"), payload.get("step_size", 8)
                    description = payload.get("description", DEFAULT_OBJECTS["apple"]["description"])
                    reference_tag = payload.get("reference_tag")
                    personal_history_enabled = payload.get("personal_history_enabled", False)
                    forum_enabled = payload.get("forum_enabled", False)
                    public_forum_enabled = payload.get("public_forum_enabled", False)
                    team_size = payload.get("team_size", 2)
                    competing_teams = payload.get("competing_teams", 1)
                    team_index = payload.get("team_index", 1)
                    team_objectives = payload.get("team_objectives")
                    artwork_size = payload.get("artwork_size", min(16, store.width))
                    prompt_template_id = payload.get("prompt_template_id", DEFAULT_PROMPT_TEMPLATE_ID)
                    prompt_template_text = payload.get("prompt_template")
                    center = _normalize_center(payload.get("center"), store.width)
                    if not isinstance(subject, str) or not 1 <= len(subject.strip()) <= 100:
                        raise CanvasError("subject must contain 1 to 100 characters")
                    if isinstance(step_size, bool) or not isinstance(step_size, int) or not 0 <= step_size <= store.width * store.height:
                        raise CanvasError(f"step_size must be an integer from 0 to {store.width * store.height}")
                    if not isinstance(description, str) or len(description) > 1_000:
                        raise CanvasError("description must be text containing at most 1,000 characters")
                    if reference_tag is not None and (not isinstance(reference_tag, str) or not reference_exists(reference_tag)):
                        raise CanvasError("reference_tag must identify an available reference or be null")
                    if not isinstance(personal_history_enabled, bool):
                        raise CanvasError("personal_history_enabled must be true or false")
                    if not isinstance(forum_enabled, bool):
                        raise CanvasError("forum_enabled must be true or false")
                    if not isinstance(public_forum_enabled, bool):
                        raise CanvasError("public_forum_enabled must be true or false")
                    if public_forum_enabled and not forum_enabled:
                        raise CanvasError("public_forum_enabled requires forum_enabled")
                    if forum_enabled and step_size == 0:
                        raise CanvasError("forums require max pixels per round to be greater than zero")
                    if isinstance(team_size, bool) or not isinstance(team_size, int) or not 1 <= team_size <= 5:
                        raise CanvasError("team_size must be an integer from 1 to 5")
                    if isinstance(competing_teams, bool) or not isinstance(competing_teams, int) or not 1 <= competing_teams <= 5:
                        raise CanvasError("competing_teams must be an integer from 1 to 5")
                    if isinstance(team_index, bool) or not isinstance(team_index, int) or not 1 <= team_index <= competing_teams:
                        raise CanvasError("team_index must identify a configured team")
                    if team_objectives is not None:
                        if (
                            not isinstance(team_objectives, list)
                            or len(team_objectives) != competing_teams
                            or any(not isinstance(item, str) or not 1 <= len(item.strip()) <= 100 for item in team_objectives)
                        ):
                            raise CanvasError("team_objectives must name each configured team's drawing objective")
                        team_objectives = [item.strip() for item in team_objectives]
                    if isinstance(artwork_size, bool) or not isinstance(artwork_size, int) or not 1 <= artwork_size <= store.width:
                        raise CanvasError(f"artwork_size must be an integer from 1 to {store.width}")
                    if not isinstance(prompt_template_id, str) or not 1 <= len(prompt_template_id) <= 100:
                        raise CanvasError("prompt_template_id must contain from 1 to 100 characters")
                    try:
                        if prompt_template_text is None:
                            prompt_template_text = prompt_template(prompt_template_id)
                        validate_prompt_template(prompt_template_text)
                    except ValueError as exc:
                        raise CanvasError(str(exc)) from exc
                    self._send_json(HTTPStatus.OK, {
                        "prompt": constructed_prompt(
                            store.width,
                            subject.strip(),
                            step_size,
                            1,
                            team_size=team_size,
                            member_index=1,
                            competing_teams=competing_teams,
                            team_index=team_index,
                            team_objectives=team_objectives,
                            artwork_size=artwork_size,
                            center=center,
                            description=description.strip(),
                            reference_tag=reference_tag,
                            reference_name=subject.strip(),
                            personal_history=[] if personal_history_enabled else None,
                            forum_enabled=forum_enabled,
                            public_forum_enabled=public_forum_enabled,
                            forum_posts={},
                            forum_activity=[],
                            prompt_template_id=prompt_template_id,
                            prompt_template_text=prompt_template_text,
                        )
                    })
                elif path == "/api/pixel":
                    action = normalize_action(payload, store.width, store.height)
                    event = apply_pixel(action, str(payload.get("source", "manual")))
                    self._send_json(HTTPStatus.OK, event)
                elif path == "/api/actions":
                    values = payload.get("actions") if isinstance(payload, dict) else None
                    if not isinstance(values, list) or not values:
                        raise CanvasError("actions must be a non-empty array")
                    if len(values) > 256:
                        raise CanvasError("a command may contain at most 256 actions")
                    source = str(payload.get("source", "command"))
                    actions = [normalize_action(value, store.width, store.height) for value in values]
                    events = [apply_pixel(action, source) for action in actions]
                    self._send_json(HTTPStatus.OK, {"events": events, "version": store.snapshot()["version"]})
                elif path == "/api/llm-response":
                    if not isinstance(payload, dict):
                        raise CanvasError("request must be an object")
                    actions = parse_llm_response(payload.get("raw_response"), width=store.width, height=store.height)
                    source = str(payload.get("source", "llm"))
                    events = [apply_pixel(action, source) for action in actions]
                    self._send_json(HTTPStatus.OK, {"events": events, "version": store.snapshot()["version"]})
                elif path == "/api/clear":
                    source = str(payload.get("source", "manual")) if isinstance(payload, dict) else "manual"
                    event = apply_clear(source)
                    self._send_json(HTTPStatus.OK, event)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except CanvasError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # keep malformed commands from killing the server
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"internal error: {exc}"})

    return CanvasHandler


def create_server(
    host: str,
    port: int,
    state_path: Path = DEFAULT_STATE,
    recording_path: Path = DEFAULT_RECORDING,
    width: int = 32,
    height: int = 32,
) -> ThreadingHTTPServer:
    server = CanvasHTTPServer((host, port), make_handler(CanvasStore(state_path, width, height), EventHub(), recording_path))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live collective canvas")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--size", type=int, default=32, help="logical grid width and height (default: 32)")
    parser.add_argument("--state", type=Path, help="state file (default: data/canvas_SIZE.json)")
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--open", action="store_true", help="open the canvas in the default browser")
    args = parser.parse_args()
    state_path = args.state or (DEFAULT_STATE if args.size == 32 else ROOT / "data" / f"canvas_{args.size}.json")
    server = create_server(args.host, args.port, state_path, args.recording, args.size, args.size)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Collective Canvas running at {url}")
    print(f"Grid: {args.size}x{args.size}")
    print(f"State file: {state_path.resolve()}")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
