"""Deterministic analysis for the collective-canvas experiments.

This script builds the canonical 225-slot manifest, validates/replays complete
runs, computes metrics 1 and 3-7, and renders matched-horizon judge images.
Metric 2 is merged later by finalize_analysis.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.grid_image import write_pixel_art_png
from harness.references import reference_rows
from harness.store import COLORS


ANALYSIS = ROOT / "analysis"
RESULTS = ANALYSIS / "results"
CANVASES = RESULTS / "analysis_canvases"

BATCH_SPECS = {
    "single_v1_core": ("single_main", "single", "V1", 10),
    "single_v2_core": ("single_main", "single", "V2", 10),
    "single_v3_core": ("single_main", "single", "V3", 20),
    "single_v1_four_agent": ("single_topup", "single", "V1", 10),
    "single_v2_four_agent": ("single_topup", "single", "V2", 10),
    "single_v3_four_agent": ("single_topup", "single", "V3", 10),
    "competing_v1": ("competing", "competing", "V1", 20),
    "competing_v2": ("competing", "competing", "V2", 20),
    "competing_v3": ("competing", "competing", "V3", 20),
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        if default is not None:
            return default
        raise


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_canvas(path: Path) -> list[list[str | None]]:
    data = read_json(path)
    rows = data["pixels"]
    size = data["size"]
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError(f"invalid canvas dimensions: {path}")
    return rows


def copy_canvas(rows: list[list[str | None]]) -> list[list[str | None]]:
    return [row[:] for row in rows]


def colored_count(rows: list[list[str | None]]) -> int:
    return sum(value is not None for row in rows for value in row)


def normalize_changed(actions: Iterable[dict[str, Any]]) -> list[tuple[int, int, str | None]]:
    result = []
    for action in actions:
        color = action.get("color")
        if color == "erase":
            color = None
        result.append((int(action["x"]), int(action["y"]), color))
    return result


@dataclass
class Entry:
    raw: dict[str, Any]
    round: int
    completion_order: int
    agent: int
    team: int
    member: int
    actions: list[dict[str, Any]]
    rejected_actions: list[dict[str, Any]]
    changed_actions: list[dict[str, Any]]
    action_kind: str
    forum: str | None
    message: str
    input_hash: str


def normalized_history(run_dir: Path) -> list[Entry]:
    history = read_json(run_dir / "history.json", [])
    entries: list[Entry] = []
    for raw in history:
        action_record: dict[str, Any] = {}
        action_path = raw.get("actions_file")
        if action_path is None and isinstance(raw.get("actions"), str):
            action_path = raw["actions"]
        if action_path:
            action_record = read_json(run_dir / action_path, {})
        accepted = raw.get("actions") if isinstance(raw.get("actions"), list) else action_record.get("actions", [])
        rejected = raw.get("rejected_actions", action_record.get("rejected_actions", []))
        changed = raw.get("changed_actions", action_record.get("changed_actions", []))
        input_path = run_dir / raw["input"]
        input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path.exists() else "missing"
        entries.append(
            Entry(
                raw=raw,
                round=int(raw.get("round", 0)),
                completion_order=int(raw.get("completion_order", 0)),
                agent=int(raw.get("agent", 1)),
                team=int(raw.get("team", 1)),
                member=int(raw.get("member", raw.get("agent", 1))),
                actions=list(accepted or []),
                rejected_actions=list(rejected or []),
                changed_actions=list(changed or []),
                action_kind=str(raw.get("action_kind", action_record.get("action_kind", "color"))),
                forum=raw.get("forum", action_record.get("forum")),
                message=str(raw.get("message", action_record.get("message", "")) or ""),
                input_hash=input_hash,
            )
        )
    return sorted(entries, key=lambda item: (item.round, item.completion_order))


def action_target(action: dict[str, Any]) -> str | None:
    return None if action.get("action") == "erase" else action.get("color")


def replay(
    run_dir: Path, entries: list[Entry]
) -> tuple[list[list[str | None]], dict[int, list[list[str | None]]], list[str]]:
    canvas = copy_canvas(load_canvas(run_dir / "initial_canvas.json"))
    size = len(canvas)
    warnings: list[str] = []
    states: dict[int, list[list[str | None]]] = {}
    for index, entry in enumerate(entries):
        calculated: list[tuple[int, int, str | None]] = []
        for action in entry.actions:
            x, y = int(action["x"]), int(action["y"])
            if not (0 <= x < size and 0 <= y < size):
                warnings.append(f"accepted_out_of_bounds:r{entry.round}:a{entry.agent}:{x},{y}")
                continue
            target = action_target(action)
            if target is not None and target not in COLORS:
                warnings.append(f"unknown_color:r{entry.round}:a{entry.agent}:{target}")
                continue
            if canvas[y][x] != target:
                canvas[y][x] = target
                calculated.append((x, y, target))
        recorded = normalize_changed(entry.changed_actions)
        if calculated != recorded:
            warnings.append(
                f"changed_actions_mismatch:r{entry.round}:a{entry.agent}:calculated={len(calculated)}:recorded={len(recorded)}"
            )
        next_round = entries[index + 1].round if index + 1 < len(entries) else None
        if next_round != entry.round:
            states[entry.round] = copy_canvas(canvas)
    final_path = run_dir / "final_canvas.json"
    if final_path.exists() and canvas != load_canvas(final_path):
        warnings.append("replay_final_mismatch")
    cohorts: defaultdict[tuple[int, str], list[int]] = defaultdict(list)
    for entry in entries:
        cohorts[(entry.round, entry.input_hash)].append(entry.completion_order)
    for (round_id, input_hash), orders in cohorts.items():
        if len(orders) != len(set(orders)):
            warnings.append(f"duplicate_completion_order:r{round_id}:input={input_hash[:8]}")
    return canvas, states, warnings


def prefix_nonblank(rows: list[list[str | None]]) -> list[list[int]]:
    size = len(rows)
    prefix = [[0] * (size + 1) for _ in range(size + 1)]
    for y, row in enumerate(rows):
        running = 0
        for x, value in enumerate(row):
            running += int(value is not None)
            prefix[y + 1][x + 1] = prefix[y][x + 1] + running
    return prefix


def area_count(prefix: list[list[int]], x: int, y: int, width: int, height: int) -> int:
    return prefix[y + height][x + width] - prefix[y][x + width] - prefix[y + height][x] + prefix[y][x]


def reference_pixels(tag: str) -> list[tuple[int, int, str]]:
    rows = reference_rows(tag, 16)
    return [(x, y, value) for y, row in enumerate(rows) for x, value in enumerate(row) if value is not None]


def overlap_fraction(a: tuple[int, int], b: tuple[int, int], box: int = 16) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(0, box - dx) * max(0, box - dy) / float(box * box)


def score_reference(
    rows: list[list[str | None]],
    tag: str,
    *,
    local_window: bool,
    centroid: tuple[float, float] | None = None,
) -> dict[str, Any]:
    size = len(rows)
    ref = reference_pixels(tag)
    ref_count = len(ref)
    total_colored = colored_count(rows)
    prefix = prefix_nonblank(rows)
    candidates: list[dict[str, Any]] = []
    for oy in range(size - 16 + 1):
        for ox in range(size - 16 + 1):
            tp = sum(rows[oy + ry][ox + rx] == color for rx, ry, color in ref)
            candidate_count = area_count(prefix, ox, oy, 16, 16) if local_window else total_colored
            denominator = ref_count + candidate_count
            f1 = 2.0 * tp / denominator if denominator else 1.0
            precision = tp / candidate_count if candidate_count else (1.0 if ref_count == 0 else 0.0)
            recall = tp / ref_count if ref_count else 1.0
            distance = 0.0
            if centroid is not None:
                distance = math.hypot(ox + 7.5 - centroid[0], oy + 7.5 - centroid[1])
            candidates.append(
                {"x": ox, "y": oy, "f1": f1, "precision": precision, "recall": recall, "tp": tp, "distance": distance}
            )
    candidates.sort(key=lambda item: (-item["f1"], item["distance"], item["y"], item["x"]))
    best = candidates[0]
    distinct = next(
        (item for item in candidates[1:] if overlap_fraction((best["x"], best["y"]), (item["x"], item["y"])) <= 0.75),
        None,
    )
    return {
        **best,
        "tag": tag,
        "candidate_colored": area_count(prefix, best["x"], best["y"], 16, 16) if local_window else total_colored,
        "reference_colored": ref_count,
        "distinct_alternative_f1": distinct["f1"] if distinct else None,
        "distinct_margin": best["f1"] - distinct["f1"] if distinct else None,
    }


def subject_centroids(entries: list[Entry]) -> dict[int, tuple[float, float] | None]:
    values: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for entry in entries:
        for action in entry.actions:
            if action_target(action) is not None:
                values[entry.team].append((int(action["x"]), int(action["y"])))
    result: dict[int, tuple[float, float] | None] = {}
    for team, points in values.items():
        result[team] = (statistics.mean(p[0] for p in points), statistics.mean(p[1] for p in points)) if points else None
    return result


def outcome_fidelity(
    rows: list[list[str | None]], mode: str, config: dict[str, Any], entries: list[Entry]
) -> tuple[float, list[dict[str, Any]], list[str]]:
    if mode == "single":
        tag = str(config.get("reference_tag") or "apple")
        score = score_reference(rows, tag, local_window=False)
        return score["f1"], [score], []
    centroids = subject_centroids(entries)
    placements: list[dict[str, Any]] = []
    flags: list[str] = []
    for team in config["teams"]:
        team_id, tag = int(team["id"]), str(team.get("reference_tag") or team["subject"])
        score = score_reference(rows, tag, local_window=True, centroid=centroids.get(team_id))
        score.update({"team": team_id, "subject": team["subject"]})
        placements.append(score)
        if score["f1"] < 0.20:
            flags.append(f"low_object_fidelity:team{team_id}")
        if score["distinct_margin"] is not None and score["distinct_margin"] < 0.02:
            flags.append(f"ambiguous_location:team{team_id}")
    for i, left in enumerate(placements):
        for right in placements[i + 1 :]:
            if overlap_fraction((left["x"], left["y"]), (right["x"], right["y"])) > 0.25:
                flags.append(f"overlapping_best_boxes:team{left['team']}:team{right['team']}")
    return statistics.mean(item["f1"] for item in placements), placements, flags


def intent_alignments(
    size: int, mode: str, config: dict[str, Any], entries: list[Entry]
) -> tuple[dict[int, dict[tuple[int, int], str]], list[dict[str, Any]]]:
    teams = config.get("teams") if mode == "competing" else [{"id": 1, "reference_tag": config.get("reference_tag", "apple"), "subject": "apple"}]
    maps: dict[int, dict[tuple[int, int], str]] = {}
    diagnostics: list[dict[str, Any]] = []
    for team in teams:
        team_id = int(team["id"])
        intent = [[None for _ in range(size)] for _ in range(size)]
        points: list[tuple[int, int]] = []
        for entry in entries:
            if entry.team != team_id:
                continue
            for action in entry.actions:
                x, y = int(action["x"]), int(action["y"])
                target = action_target(action)
                intent[y][x] = target
                if target is not None:
                    points.append((x, y))
        centroid = (statistics.mean(p[0] for p in points), statistics.mean(p[1] for p in points)) if points else None
        tag = str(team.get("reference_tag") or team.get("subject") or "apple")
        score = score_reference(intent, tag, local_window=False, centroid=centroid)
        expected = {(score["x"] + x, score["y"] + y): color for x, y, color in reference_pixels(tag)}
        inside = sum(
            value is not None and score["x"] <= x < score["x"] + 16 and score["y"] <= y < score["y"] + 16
            for y, row in enumerate(intent)
            for x, value in enumerate(row)
        )
        total = colored_count(intent)
        spill = 1.0 - inside / total if total else 1.0
        maps[team_id] = expected
        diagnostics.append(
            {
                "team": team_id,
                "subject": team.get("subject", "apple"),
                "x": score["x"],
                "y": score["y"],
                "intent_f1": score["f1"],
                "intent_spill": spill,
                "low_confidence": total == 0 or score["f1"] < 0.20 or spill > 0.40,
            }
        )
    return maps, diagnostics


def conflict_and_repair(
    initial: list[list[str | None]], entries: list[Entry], expected_maps: dict[int, dict[tuple[int, int], str]]
) -> dict[str, Any]:
    cohorts: defaultdict[tuple[int, str], list[Entry]] = defaultdict(list)
    for entry in entries:
        cohorts[(entry.round, entry.input_hash)].append(entry)
    ordered = sorted(cohorts.items(), key=lambda item: (item[0][0], min(e.completion_order for e in item[1])))

    total_actions = 0
    total_unique_targets = 0
    shared_targets = 0
    conflict_targets = 0
    for _, cohort in ordered:
        proposals: defaultdict[tuple[int, int], dict[int, str | None]] = defaultdict(dict)
        for entry in cohort:
            total_actions += len(entry.actions)
            actor_last: dict[tuple[int, int], str | None] = {}
            for action in entry.actions:
                actor_last[(int(action["x"]), int(action["y"]))] = action_target(action)
            for coordinate, target in actor_last.items():
                proposals[coordinate][entry.agent] = target
        total_unique_targets += len(proposals)
        for by_actor in proposals.values():
            if len(by_actor) >= 2:
                shared_targets += 1
                if len(set(by_actor.values())) >= 2:
                    conflict_targets += 1

    canvas = copy_canvas(initial)
    owner: list[list[tuple[int, int] | None]] = [[None for _ in row] for row in canvas]
    episodes: dict[tuple[int, int, int], dict[str, Any]] = {}
    persistent_started = 0
    repaired = 0
    opponent_recoveries = 0
    unobserved_reversals = 0
    repair_latencies: list[int] = []
    intra_damage = 0
    cross_damage = 0
    state_changes = 0

    for (round_id, _), cohort in ordered:
        before = copy_canvas(canvas)
        before_owner = [row[:] for row in owner]
        destructive_candidates: dict[tuple[int, int, int], dict[str, Any]] = {}
        correct_restorers: defaultdict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
        for entry in sorted(cohort, key=lambda item: item.completion_order):
            for action in entry.actions:
                x, y = int(action["x"]), int(action["y"])
                target = action_target(action)
                if canvas[y][x] == target:
                    continue
                state_changes += 1
                prior_owner = owner[y][x]
                if prior_owner is not None and prior_owner[1] != entry.agent:
                    affected_team = prior_owner[0]
                    expected = expected_maps.get(affected_team, {}).get((x, y))
                    if expected is not None and canvas[y][x] == expected and target != expected:
                        destructive_candidates[(affected_team, x, y)] = {
                            "affected_team": affected_team,
                            "damaging_team": entry.team,
                            "round": round_id,
                            "source_agent": prior_owner[1],
                            "damaging_agent": entry.agent,
                        }
                for affected_team, expected_map in expected_maps.items():
                    expected = expected_map.get((x, y))
                    if expected is not None and target == expected:
                        correct_restorers[(affected_team, x, y)].append((entry.team, entry.agent))
                canvas[y][x] = target
                owner[y][x] = (entry.team, entry.agent)

        # Resolve episodes that existed at the cohort start; these teams saw the damage.
        for key, episode in list(episodes.items()):
            affected_team, x, y = key
            expected = expected_maps[affected_team].get((x, y))
            if expected is not None and canvas[y][x] == expected:
                restorers = correct_restorers.get(key, [])
                if any(team == affected_team for team, _ in restorers):
                    repaired += 1
                    repair_latencies.append(round_id - int(episode["round"]))
                else:
                    opponent_recoveries += 1
                del episodes[key]

        # New within-cohort damage becomes persistent only if visible after the cohort.
        for key, candidate in destructive_candidates.items():
            if key in episodes:
                continue
            affected_team, x, y = key
            expected = expected_maps[affected_team].get((x, y))
            if expected is not None and canvas[y][x] != expected:
                episodes[key] = candidate
                persistent_started += 1
                if candidate["affected_team"] == candidate["damaging_team"]:
                    intra_damage += 1
                else:
                    cross_damage += 1
            else:
                unobserved_reversals += 1

    return {
        "accepted_pixel_actions": total_actions,
        "state_changing_pixel_actions": state_changes,
        "unique_target_coordinates": total_unique_targets,
        "shared_target_coordinates": shared_targets,
        "different_state_conflict_coordinates": conflict_targets,
        "intent_conflict_rate": conflict_targets / total_unique_targets if total_unique_targets else 0.0,
        "shared_target_disagreement_rate": conflict_targets / shared_targets if shared_targets else 0.0,
        "conflicts_per_100_actions": 100.0 * conflict_targets / total_actions if total_actions else 0.0,
        "persistent_damage_episodes": persistent_started,
        "intra_team_damage_episodes": intra_damage,
        "cross_team_damage_episodes": cross_damage,
        "destructive_overwrite_rate": persistent_started / state_changes if state_changes else 0.0,
        "observed_team_repairs": repaired,
        "repair_rate": repaired / persistent_started if persistent_started else None,
        "median_repair_latency_rounds": statistics.median(repair_latencies) if repair_latencies else None,
        "opponent_recoveries": opponent_recoveries,
        "unobserved_reversals": unobserved_reversals,
        "unrepaired_damage_episodes": len(episodes),
    }


def cost_record(run_dir: Path, horizon: int | None = None) -> dict[str, Any]:
    response_dir = run_dir / "responses"
    seen: set[str] = set()
    cost = 0.0
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    responses = invalid = missing_cost = 0
    model_costs: defaultdict[str, float] = defaultdict(float)
    if response_dir.exists():
        for path in sorted(response_dir.glob("*.json")):
            match = re.search(r"round_(\d+)", path.name)
            if horizon is not None and match and int(match.group(1)) > horizon:
                continue
            data = read_json(path, {})
            response_id = data.get("id")
            usage = data.get("usage") if isinstance(data, dict) else None
            if not response_id or not isinstance(usage, dict) or response_id in seen:
                continue
            seen.add(response_id)
            responses += 1
            invalid += int("invalid_output" in path.name)
            value = usage.get("cost")
            if value is None:
                missing_cost += 1
            else:
                value = float(value)
                cost += value
                model_costs[str(data.get("model", "unknown"))] += value
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens += int(details.get("reasoning_tokens") or 0)
    retry_counts: Counter[str] = Counter()
    retry_dir = run_dir / "retries"
    if retry_dir.exists():
        for path in retry_dir.glob("*.jsonl"):
            match = re.search(r"round_(\d+)", path.name)
            if horizon is not None and match and int(match.group(1)) > horizon:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    retry_counts[str(json.loads(line).get("kind", "unknown"))] += 1
                except json.JSONDecodeError:
                    retry_counts["malformed_log"] += 1
    return {
        "cost_usd": cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "billed_responses": responses,
        "invalid_output_responses": invalid,
        "missing_cost_responses": missing_cost,
        "retry_counts": dict(retry_counts),
        "model_costs": dict(model_costs),
    }


def communication(entries: list[Entry]) -> dict[str, Any]:
    team_start = team_mid = public_start = public_mid = 0
    for entry in entries:
        start = entry.round <= 1
        if entry.action_kind == "write_team_forum":
            team_start += int(start)
            team_mid += int(not start)
        elif entry.action_kind == "write_public_forum":
            public_start += int(start)
            public_mid += int(not start)
    turns = len(entries)
    total = team_start + team_mid + public_start + public_mid
    return {
        "agent_turns": turns,
        "team_posts_start": team_start,
        "team_posts_mid": team_mid,
        "public_posts_start": public_start,
        "public_posts_mid": public_mid,
        "forum_posts_total": total,
        "used_forum": int(total > 0),
        "posts_per_100_turns": 100.0 * total / turns if turns else 0.0,
    }


def crop_set(rows: list[list[str | None]], x0: int, y0: int, size: int = 16) -> set[tuple[int, int, str]]:
    return {
        (x, y, rows[y0 + y][x0 + x])
        for y in range(size)
        for x in range(size)
        if rows[y0 + y][x0 + x] is not None
    }


def f1_at_round(
    initial: list[list[str | None]], states: dict[int, list[list[str | None]]], round_id: int, mode: str, config: dict[str, Any], entries: list[Entry]
) -> tuple[float, list[dict[str, Any]], list[str], list[list[str | None]]]:
    available = [key for key in states if key <= round_id]
    rows = states[max(available)] if available else initial
    score, placements, flags = outcome_fidelity(rows, mode, config, [entry for entry in entries if entry.round <= round_id])
    return score, placements, flags, rows


def process_complete_run(meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = Path(meta["run_dir"])
    config = read_json(run_dir / "config.json")
    result = read_json(run_dir / "result.json")
    entries = normalized_history(run_dir)
    initial = load_canvas(run_dir / "initial_canvas.json")
    replayed, states, validation_warnings = replay(run_dir, entries)
    horizon = int(meta["horizon"])
    primary_entries = [entry for entry in entries if entry.round <= horizon]
    score_h, placements_h, flags_h, horizon_canvas = f1_at_round(initial, states, horizon, meta["mode"], config, entries)
    score_final, placements_final, flags_final = outcome_fidelity(replayed, meta["mode"], config, entries)

    trajectory_rows: list[dict[str, Any]] = []
    trajectory_scores: list[float] = []
    for round_id in range(0, horizon + 1):
        score, _, _, _ = f1_at_round(initial, states, round_id, meta["mode"], config, entries)
        trajectory_scores.append(score)
        trajectory_rows.append({"run_key": meta["run_key"], "round": round_id, "reference_fidelity": score})
    progress_auc = statistics.mean(trajectory_scores)

    expected_maps, intent_diagnostics = intent_alignments(len(initial), meta["mode"], config, primary_entries)
    conflict = conflict_and_repair(initial, primary_entries, expected_maps)
    comm = communication(primary_entries)
    comm_full = communication(entries)
    cost_h = cost_record(run_dir, horizon=horizon)
    cost_full = cost_record(run_dir, horizon=None)

    stop_reason = str(result.get("stop_reason", "unknown"))
    gap = 1.0 - score_final if stop_reason in {"team_complete", "complete", "stalled"} else None
    stop_claim = "completion" if stop_reason in {"team_complete", "complete"} else "stall" if stop_reason == "stalled" else "censored"
    image_path = CANVASES / f"{meta['run_key']}.png"
    write_pixel_art_png(image_path, horizon_canvas, COLORS)

    # Canonical crops are serialized for repeatability calculations.
    crops: dict[str, list[list[Any]]] = {}
    if meta["mode"] == "single":
        p = placements_h[0]
        crops["apple"] = [list(value) for value in sorted(crop_set(horizon_canvas, p["x"], p["y"]))]
    else:
        for p in placements_h:
            crops[p["subject"]] = [list(value) for value in sorted(crop_set(horizon_canvas, p["x"], p["y"]))]

    row = {
        **meta,
        "run_id": result.get("run_id", meta["run_key"]),
        "configured_rounds": config.get("max_steps"),
        "observed_last_round": max((entry.round for entry in entries), default=-1),
        "stop_reason": stop_reason,
        "reference_fidelity": score_h,
        "reference_fidelity_actual_final": score_final,
        "reference_precision": statistics.mean(p["precision"] for p in placements_h),
        "reference_recall": statistics.mean(p["recall"] for p in placements_h),
        "progress_auc": progress_auc,
        "stopping_class": stop_claim,
        "stopping_gap": gap,
        "filled_pixels_horizon": colored_count(horizon_canvas),
        "filled_pixels_actual_final": colored_count(replayed),
        "analysis_image": str(image_path.relative_to(ROOT)).replace("\\", "/"),
        "validation_ok": int(not validation_warnings),
        "validation_warning_count": len(validation_warnings),
        "localization_flag_count": len(flags_h),
        "intent_low_confidence_teams": sum(int(item["low_confidence"]) for item in intent_diagnostics),
        **conflict,
        **comm,
        "forum_posts_total_actual": comm_full["forum_posts_total"],
        "generation_cost_usd": cost_h["cost_usd"],
        "generation_cost_actual_total_usd": cost_full["cost_usd"],
        "dollars_per_0_1_fidelity": 0.1 * cost_h["cost_usd"] / score_h if score_h > 0 else math.inf,
        "prompt_tokens": cost_h["prompt_tokens"],
        "completion_tokens": cost_h["completion_tokens"],
        "reasoning_tokens": cost_h["reasoning_tokens"],
        "billed_responses": cost_h["billed_responses"],
        "invalid_output_responses": cost_h["invalid_output_responses"],
        "missing_cost_responses": cost_h["missing_cost_responses"],
        "rate_limit_retries": cost_h["retry_counts"].get("rate_limit", 0),
        "transient_retries": sum(value for key, value in cost_h["retry_counts"].items() if key not in {"rate_limit", "invalid_output"}),
        "invalid_output_retries": cost_h["retry_counts"].get("invalid_output", 0),
    }
    detail = {
        "run_key": meta["run_key"],
        "placements_horizon": placements_h,
        "placements_actual_final": placements_final,
        "localization_flags_horizon": flags_h,
        "localization_flags_actual_final": flags_final,
        "intent_alignments": intent_diagnostics,
        "validation_warnings": validation_warnings,
        "cost_horizon": cost_h,
        "cost_actual_total": cost_full,
        "canonical_crops": crops,
    }
    return row, detail, trajectory_rows, intent_diagnostics


def manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_root = ROOT / "harness" / "batches"
    for batch, (phase, mode, prompt_version, horizon) in BATCH_SPECS.items():
        summary = read_json(batch_root / batch / "summary.json")
        for item in summary["runs"]:
            condition = str(item["condition"])
            repetition = int(item["repetition"])
            run_dir = batch_root / batch / "runs" / condition / f"rep_{repetition:02d}"
            run_key = f"{batch}__{condition}__rep_{repetition:02d}"
            rows.append(
                {
                    "run_key": run_key,
                    "batch": batch,
                    "phase": phase,
                    "mode": mode,
                    "prompt_version": prompt_version,
                    "condition": condition,
                    "repetition": repetition,
                    "horizon": horizon,
                    "status": str(item["status"]),
                    "run_dir": str(run_dir),
                    "summary_stop_reason": item.get("stop_reason", ""),
                    "summary_error": item.get("error", ""),
                }
            )
    return sorted(rows, key=lambda row: (row["phase"], row["prompt_version"], row["condition"], row["repetition"]))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CANVASES.mkdir(parents=True, exist_ok=True)
    manifest_rows = manifest()
    run_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    trajectories: list[dict[str, Any]] = []
    for index, meta in enumerate(manifest_rows, 1):
        print(f"[{index:03d}/{len(manifest_rows)}] {meta['run_key']} {meta['status']}", flush=True)
        run_dir = Path(meta["run_dir"])
        if meta["status"] != "complete" or not (run_dir / "result.json").exists():
            cost = cost_record(run_dir)
            run_rows.append(
                {
                    **meta,
                    "generation_cost_usd": cost["cost_usd"],
                    "generation_cost_actual_total_usd": cost["cost_usd"],
                    "prompt_tokens": cost["prompt_tokens"],
                    "completion_tokens": cost["completion_tokens"],
                    "reasoning_tokens": cost["reasoning_tokens"],
                    "billed_responses": cost["billed_responses"],
                    "invalid_output_responses": cost["invalid_output_responses"],
                    "missing_cost_responses": cost["missing_cost_responses"],
                }
            )
            details[meta["run_key"]] = {"cost_actual_total": cost, "failure": meta["summary_error"]}
            continue
        row, detail, trajectory, _ = process_complete_run(meta)
        run_rows.append(row)
        details[meta["run_key"]] = detail
        trajectories.extend(trajectory)

    write_csv(RESULTS / "canonical_manifest.csv", manifest_rows)
    write_csv(RESULTS / "run_metrics_deterministic.csv", run_rows)
    write_csv(RESULTS / "round_trajectories.csv", trajectories)
    write_json(RESULTS / "run_details.json", details)
    validation = {
        "planned_slots": len(manifest_rows),
        "complete_runs": sum(row["status"] == "complete" for row in run_rows),
        "failed_runs": sum(row["status"] != "complete" for row in run_rows),
        "replay_valid": sum(row.get("validation_ok") == 1 for row in run_rows),
        "replay_with_warnings": [row["run_key"] for row in run_rows if row.get("validation_ok") == 0],
        "missing_cost_runs": [row["run_key"] for row in run_rows if int(row.get("missing_cost_responses") or 0) > 0],
        "analysis_horizons": {"single": 10, "competing": 20},
    }
    write_json(RESULTS / "deterministic_validation.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
