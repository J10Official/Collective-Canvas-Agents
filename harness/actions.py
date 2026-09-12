"""Validation and parsing for manual, command-line, and LLM actions."""

from __future__ import annotations

import json
import re
from typing import Any

from .store import COLORS, HEIGHT, WIDTH, CanvasError

MAX_ACTIONS_PER_RESPONSE = 4


def _json_text(raw: str) -> str:
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else text


def normalize_action(value: Any, width: int = WIDTH, height: int = HEIGHT) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CanvasError("each action must be a JSON object")
    kind = value.get("action", "place")
    if kind not in {"place", "erase"}:
        raise CanvasError("action must be 'place' or 'erase'")
    x, y = value.get("x"), value.get("y")
    if not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < width:
        raise CanvasError(f"x must be an integer from 0 to {width - 1}")
    if not isinstance(y, int) or isinstance(y, bool) or not 0 <= y < height:
        raise CanvasError(f"y must be an integer from 0 to {height - 1}")
    color = None if kind == "erase" else value.get("color")
    if kind == "place" and color not in COLORS:
        raise CanvasError(f"color must be one of: {', '.join(COLORS)}")
    return {"action": kind, "x": x, "y": y, "color": color}


def parse_llm_response(
    raw: str,
    max_actions: int = MAX_ACTIONS_PER_RESPONSE,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> list[dict[str, Any]]:
    """Parse the strict action contract: {"actions": [{...}, ...]}."""
    if not isinstance(raw, str) or not raw.strip():
        raise CanvasError("raw_response must be a non-empty string")
    try:
        payload = json.loads(_json_text(raw))
    except json.JSONDecodeError as exc:
        raise CanvasError(f"LLM response is not valid JSON: {exc.msg}") from exc
    values = payload.get("actions") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise CanvasError("LLM response must contain an 'actions' array")
    if not 1 <= len(values) <= max_actions:
        raise CanvasError(f"LLM response must contain 1 to {max_actions} actions")
    return [normalize_action(value, width, height) for value in values]


def parse_drawing_response(raw: str, size: int, iterative: bool = False) -> tuple[list[dict[str, Any]], bool]:
    """Parse a structured drawing step and reject repeated cells."""
    if not isinstance(raw, str) or not raw.strip():
        raise CanvasError("model output was empty")
    try:
        value = json.loads(_json_text(raw))
    except json.JSONDecodeError as exc:
        raise CanvasError(f"model output was not valid JSON: {exc.msg}") from exc
    pixels = value.get("pixels") if isinstance(value, dict) else None
    if not isinstance(pixels, list) or (not iterative and not pixels):
        requirement = "a pixels array" if iterative else "a non-empty pixels array"
        raise CanvasError(f"model output must contain {requirement}")
    if any(not isinstance(pixel, dict) for pixel in pixels):
        raise CanvasError("every pixels entry must contain x, y, and color")
    actions = [
        normalize_action(
            {**pixel, "action": "erase" if pixel.get("color") == "erase" else "place"},
            size,
            size,
        )
        for pixel in pixels
    ]
    coordinates = [(action["x"], action["y"]) for action in actions]
    if len(coordinates) != len(set(coordinates)):
        raise CanvasError("model output repeated at least one coordinate")
    complete = not actions if iterative else True
    return actions, complete


def parse_forum_turn(
    raw: str,
    size: int,
    *,
    iterative: bool = True,
    public_forum_enabled: bool = False,
) -> dict[str, Any]:
    """Parse one mutually exclusive color or forum-write action."""
    if not isinstance(raw, str) or not raw.strip():
        raise CanvasError("model output was empty")
    try:
        value = json.loads(_json_text(raw))
    except json.JSONDecodeError as exc:
        raise CanvasError(f"model output was not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict) or set(value) != {"action", "pixels", "message"}:
        raise CanvasError("forum-mode output must contain exactly action, pixels, and message")
    kind, pixels, message = value["action"], value["pixels"], value["message"]
    allowed = {"color", "write_team_forum"}
    if public_forum_enabled:
        allowed.add("write_public_forum")
    if kind not in allowed:
        raise CanvasError(f"action must be one of: {', '.join(sorted(allowed))}")
    if not isinstance(pixels, list):
        raise CanvasError("pixels must be an array")
    if not isinstance(message, str):
        raise CanvasError("message must be a string")

    if kind == "color":
        actions, complete = parse_drawing_response(
            json.dumps({"pixels": pixels}),
            size,
            iterative=iterative,
        )
        if message:
            raise CanvasError("message must be empty for a color action")
        return {"kind": kind, "actions": actions, "forum": None, "message": "", "complete": complete}

    if pixels:
        raise CanvasError("pixels must be empty for forum actions")
    forum = "public" if "public" in kind else "team"
    cleaned = message.strip()
    if not 1 <= len(cleaned) <= 500:
        raise CanvasError("a forum post must contain from 1 to 500 characters")
    return {"kind": kind, "actions": [], "forum": forum, "message": cleaned, "complete": False}


def parse_pixel_drawing(raw: str, size: int) -> list[dict[str, Any]]:
    """Parse the original one-call drawing response."""
    return parse_drawing_response(raw, size)[0]
