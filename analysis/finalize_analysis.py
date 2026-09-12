"""Merge VLM scores, aggregate all seven metrics, and write the result report."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import struct
import sys
import zlib
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.grid_image import _png_bytes

RESULTS = ROOT / "analysis" / "results"
JUDGES = RESULTS / "judges"


TASK_LABELS = {
    "single": "Single-team shared image",
    "competing": "Four-team shared-canvas competition",
}

PARTICIPANT_STRUCTURE_LABELS = {
    ("single", "homogeneous_gemini"): "2 Gemini agents",
    ("single", "homogeneous_qwen"): "2 Qwen agents",
    ("single", "homogeneous_luna"): "2 Luna agents",
    ("single", "heterogeneous"): "4 mixed agents (GLM, Gemini, Qwen, Luna)",
    ("single", "homogeneous4_gemini"): "4 Gemini agents",
    ("single", "homogeneous4_glm"): "4 GLM agents",
    ("single", "homogeneous4_qwen"): "4 Qwen agents",
    ("single", "homogeneous4_luna"): "4 Luna agents",
    ("competing", "all_gemini"): "4 Gemini agents",
    ("competing", "four_heterogeneous_teams"): "4 mixed agents (GLM, Gemini, Qwen, Luna)",
    ("competing", "homogeneous_family_per_team"): "4 homogeneous agents; one model family per team",
}

PARTICIPANT_STRUCTURE_ORDER = [
    "2 Gemini agents",
    "2 Qwen agents",
    "2 Luna agents",
    "4 Gemini agents",
    "4 GLM agents",
    "4 Qwen agents",
    "4 Luna agents",
    "4 mixed agents (GLM, Gemini, Qwen, Luna)",
    "4 homogeneous agents; one model family per team",
]


def task_id(row: dict[str, Any]) -> str:
    return "single" if row["mode"] == "single" else "competing"


def participant_structure(row: dict[str, Any]) -> str:
    key = (task_id(row), row["condition"])
    if key not in PARTICIPANT_STRUCTURE_LABELS:
        raise KeyError(f"unmapped participant structure: {key}")
    return PARTICIPANT_STRUCTURE_LABELS[key]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def ci_mean(values: list[float], seed: int, samples: int = 5000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = [statistics.mean(rng.choice(values) for _ in values) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for index in order[i:j]:
            ranks[index] = rank
        i = j
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return math.nan
    a, b = average_ranks(left), average_ranks(right)
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return numerator / denominator if denominator else 1.0


def color_iou(left: set[tuple[Any, ...]], right: set[tuple[Any, ...]]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def repeatability_for_group(run_keys: list[str], details: dict[str, Any], mode: str) -> float | None:
    pairs = list(combinations(run_keys, 2))
    if not pairs:
        return None
    scores: list[float] = []
    for left, right in pairs:
        lcrops = details[left]["canonical_crops"]
        rcrops = details[right]["canonical_crops"]
        if mode == "single":
            scores.append(color_iou(set(map(tuple, lcrops["apple"])), set(map(tuple, rcrops["apple"]))))
        else:
            tags = sorted(set(lcrops) & set(rcrops))
            scores.append(
                statistics.mean(color_iou(set(map(tuple, lcrops[tag])), set(map(tuple, rcrops[tag]))) for tag in tags)
            )
    return statistics.mean(scores)


def pooled_repair(group: list[dict[str, Any]]) -> float | None:
    damage = sum(int(float(row.get("persistent_damage_episodes") or 0)) for row in group)
    repaired = sum(int(float(row.get("observed_team_repairs") or 0)) for row in group)
    return repaired / damage if damage else None


def aggregate_conditions(rows: list[dict[str, Any]], details: dict[str, Any]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(task_id(row), row["prompt_version"], participant_structure(row))].append(row)
    result = []
    metrics = [
        "reference_fidelity",
        "visual_quality",
        "progress_auc",
        "intent_conflict_rate",
        "destructive_overwrite_rate",
        "posts_per_100_turns",
        "generation_cost_usd",
        "dollars_per_0_1_fidelity",
    ]
    for group_index, (key, group) in enumerate(sorted(groups.items())):
        complete = [row for row in group if row["status"] == "complete"]
        record: dict[str, Any] = {
            "task": key[0],
            "task_label": TASK_LABELS[key[0]],
            "prompt_version": key[1],
            "participant_structure": key[2],
            "planned_n": len(group),
            "complete_n": len(complete),
            "technical_failure_rate": 1.0 - len(complete) / len(group),
        }
        for metric_index, metric in enumerate(metrics):
            values = [value for row in complete if (value := number(row.get(metric))) is not None and math.isfinite(value)]
            if not values:
                continue
            low, high = ci_mean(values, 20260912 + 100 * group_index + metric_index)
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_median"] = statistics.median(values)
            record[f"{metric}_q1"] = percentile(values, 0.25)
            record[f"{metric}_q3"] = percentile(values, 0.75)
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        record["pooled_repair_rate"] = pooled_repair(complete)
        record["unique_target_coordinates"] = sum(int(float(row.get("unique_target_coordinates") or 0)) for row in complete)
        record["different_state_conflict_coordinates"] = sum(int(float(row.get("different_state_conflict_coordinates") or 0)) for row in complete)
        record["pooled_intent_conflict_rate"] = (
            record["different_state_conflict_coordinates"] / record["unique_target_coordinates"]
            if record["unique_target_coordinates"] else 0.0
        )
        record["state_changing_pixel_actions"] = sum(int(float(row.get("state_changing_pixel_actions") or 0)) for row in complete)
        record["persistent_damage_episodes"] = sum(int(float(row.get("persistent_damage_episodes") or 0)) for row in complete)
        record["pooled_destructive_overwrite_rate"] = (
            record["persistent_damage_episodes"] / record["state_changing_pixel_actions"]
            if record["state_changing_pixel_actions"] else 0.0
        )
        record["observed_team_repairs"] = sum(int(float(row.get("observed_team_repairs") or 0)) for row in complete)
        record["used_forum_rate"] = statistics.mean(float(row.get("used_forum") or 0) for row in complete) if complete else None
        keys = [row["run_key"] for row in complete]
        record["canonical_color_iou_repeatability"] = repeatability_for_group(keys, details, key[0])
        record["canonical_diversity"] = 1.0 - record["canonical_color_iou_repeatability"] if record["canonical_color_iou_repeatability"] is not None else None
        result.append(record)
    return result


def prompt_summary(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in conditions:
        groups[(row["task"], row["prompt_version"])].append(row)
    metrics = [
        "reference_fidelity_mean",
        "visual_quality_mean",
        "progress_auc_mean",
        "intent_conflict_rate_mean",
        "destructive_overwrite_rate_mean",
        "posts_per_100_turns_mean",
        "generation_cost_usd_mean",
        "canonical_color_iou_repeatability",
        "technical_failure_rate",
    ]
    output = []
    for (task, version), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "task": task,
            "task_label": TASK_LABELS[task],
            "prompt_version": version,
            "participant_structure_blocks": len(group),
        }
        for metric in metrics:
            values = [value for item in group if (value := number(item.get(metric))) is not None]
            row[metric.replace("_mean", "")] = statistics.mean(values) if values else None
        damage = sum(int(item["persistent_damage_episodes"]) for item in group)
        repaired = sum(int(item["observed_team_repairs"]) for item in group)
        row["pooled_repair_rate"] = repaired / damage if damage else None
        output.append(row)
    return output


def communication_effects(rows: list[dict[str, Any]], samples: int = 5000) -> list[dict[str, Any]]:
    complete = [row for row in rows if row["status"] == "complete"]
    metrics = [
        "forum_posts_total",
        "posts_per_100_turns",
        "reference_fidelity",
        "visual_quality",
        "intent_conflict_rate",
        "destructive_overwrite_rate",
        "progress_auc",
        "generation_cost_usd",
    ]
    output = []
    for task in sorted({task_id(row) for row in complete}):
        task_rows = [row for row in complete if task_id(row) == task]
        blocks = sorted({participant_structure(row) for row in task_rows})
        for contrast in ("V2", "V3"):
            for metric_index, metric in enumerate(metrics):
                usable_blocks = []
                for block in blocks:
                    base = [number(row.get(metric)) for row in task_rows if participant_structure(row) == block and row["prompt_version"] == "V1"]
                    test = [number(row.get(metric)) for row in task_rows if participant_structure(row) == block and row["prompt_version"] == contrast]
                    base = [x for x in base if x is not None]
                    test = [x for x in test if x is not None]
                    if base and test:
                        usable_blocks.append((base, test))
                observed = statistics.mean(statistics.mean(test) - statistics.mean(base) for base, test in usable_blocks)
                rng = random.Random(20270000 + metric_index + (100 if contrast == "V3" else 0) + len(output))
                boot = []
                for _ in range(samples):
                    block_effects = []
                    for base, test in usable_blocks:
                        boot_base = statistics.mean(rng.choice(base) for _ in base)
                        boot_test = statistics.mean(rng.choice(test) for _ in test)
                        block_effects.append(boot_test - boot_base)
                    boot.append(statistics.mean(block_effects))
                output.append(
                    {
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "contrast": f"{contrast}-V1",
                        "metric": metric,
                        "participant_structure_blocks": len(usable_blocks),
                        "effect": observed,
                        "ci_low": percentile(boot, 0.025),
                        "ci_high": percentile(boot, 0.975),
                    }
                )
    return output


def judge_agreement(scores: list[dict[str, str]]) -> dict[str, Any]:
    by_stage: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in scores:
        by_stage[row["stage"]][row["run_key"]] = row
    primary = by_stage["primary"]
    output: dict[str, Any] = {}
    for stage in ("grok_repeat", "kimi_audit"):
        other = by_stage[stage]
        keys = sorted(set(primary) & set(other))
        left = [float(primary[key]["visual_quality"]) for key in keys]
        right = [float(other[key]["visual_quality"]) for key in keys]
        output[stage] = {
            "n": len(keys),
            "visual_quality_spearman": spearman(left, right),
            "visual_quality_mean_absolute_difference": statistics.mean(abs(a - b) for a, b in zip(left, right)),
            "primary_mean": statistics.mean(left),
            "other_mean": statistics.mean(right),
        }
    return output


def stopping_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            groups[(task_id(row), row["prompt_version"])].append(row)
    output = []
    for (task, version), group in sorted(groups.items()):
        record: dict[str, Any] = {"task": task, "task_label": TASK_LABELS[task], "prompt_version": version, "n": len(group)}
        for reason in ("team_complete", "stalled", "max_steps"):
            subset = [row for row in group if row["stop_reason"] == reason]
            record[f"{reason}_n"] = len(subset)
            gaps = [value for row in subset if (value := number(row.get("stopping_gap"))) is not None]
            record[f"{reason}_gap_mean"] = statistics.mean(gaps) if gaps else None
            record[f"{reason}_gap_median"] = statistics.median(gaps) if gaps else None
        gains = [float(row["reference_fidelity_actual_final"]) - float(row["reference_fidelity"]) for row in group]
        record["post_horizon_fidelity_gain_mean"] = statistics.mean(gains)
        output.append(record)
    return output


def conflict_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            groups[(task_id(row), row["prompt_version"])].append(row)
    output = []
    for (task, version), group in sorted(groups.items()):
        unique_targets = sum(int(float(row.get("unique_target_coordinates") or 0)) for row in group)
        conflicts = sum(int(float(row.get("different_state_conflict_coordinates") or 0)) for row in group)
        changes = sum(int(float(row.get("state_changing_pixel_actions") or 0)) for row in group)
        damage = sum(int(float(row.get("persistent_damage_episodes") or 0)) for row in group)
        repairs = sum(int(float(row.get("observed_team_repairs") or 0)) for row in group)
        output.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "prompt_version": version,
                "n": len(group),
                "unique_target_coordinates": unique_targets,
                "different_state_conflict_coordinates": conflicts,
                "pooled_intent_conflict_rate": conflicts / unique_targets if unique_targets else 0.0,
                "state_changing_pixel_actions": changes,
                "persistent_damage_episodes": damage,
                "pooled_destructive_overwrite_rate": damage / changes if changes else 0.0,
                "observed_team_repairs": repairs,
                "pooled_repair_rate": repairs / damage if damage else None,
                "cross_team_damage_episodes": sum(int(float(row.get("cross_team_damage_episodes") or 0)) for row in group),
                "intra_team_damage_episodes": sum(int(float(row.get("intra_team_damage_episodes") or 0)) for row in group),
                "unobserved_reversals": sum(int(float(row.get("unobserved_reversals") or 0)) for row in group),
                "opponent_recoveries": sum(int(float(row.get("opponent_recoveries") or 0)) for row in group),
                "unrepaired_damage_episodes": sum(int(float(row.get("unrepaired_damage_episodes") or 0)) for row in group),
            }
        )
    return output


def forum_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            groups[(task_id(row), row["prompt_version"])].append(row)
    output = []
    for (task, version), group in sorted(groups.items()):
        turns = sum(int(float(row.get("agent_turns") or 0)) for row in group)
        record: dict[str, Any] = {
            "task": task,
            "task_label": TASK_LABELS[task],
            "prompt_version": version,
            "n": len(group),
            "agent_turns": turns,
        }
        for field in ("team_posts_start", "team_posts_mid", "public_posts_start", "public_posts_mid", "forum_posts_total"):
            count = sum(int(float(row.get(field) or 0)) for row in group)
            record[field] = count
            record[f"{field}_per_100_turns"] = 100.0 * count / turns if turns else 0.0
        record["runs_using_forum"] = sum(int(float(row.get("used_forum") or 0)) for row in group)
        record["runs_using_forum_rate"] = record["runs_using_forum"] / len(group)
        output.append(record)
    return output


def object_metrics(rows: list[dict[str, Any]], details: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    by_key = {row["run_key"]: row for row in rows}
    for run_key, detail in details.items():
        meta = by_key.get(run_key)
        if not meta or meta["status"] != "complete":
            continue
        for placement in detail.get("placements_horizon", []):
            output.append(
                {
                    "run_key": run_key,
                    "task": task_id(meta),
                    "task_label": TASK_LABELS[task_id(meta)],
                    "prompt_version": meta["prompt_version"],
                    "participant_structure": participant_structure(meta),
                    "team": placement.get("team", 1),
                    "subject": placement.get("subject", placement.get("tag", "apple")),
                    "x": placement["x"],
                    "y": placement["y"],
                    "reference_fidelity": placement["f1"],
                    "reference_precision": placement["precision"],
                    "reference_recall": placement["recall"],
                    "distinct_margin": placement.get("distinct_margin"),
                }
            )
    return output


def wide_configuration_task_matrix(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "planned_n",
        "complete_n",
        "technical_failure_rate",
        "reference_fidelity_mean",
        "visual_quality_mean",
        "progress_auc_mean",
        "pooled_intent_conflict_rate",
        "pooled_destructive_overwrite_rate",
        "pooled_repair_rate",
        "posts_per_100_turns_mean",
        "generation_cost_usd_mean",
        "canonical_color_iou_repeatability",
    ]
    lookup = {
        (row["participant_structure"], row["task"], row["prompt_version"]): row for row in conditions
    }
    output = []
    for structure in PARTICIPANT_STRUCTURE_ORDER:
        if not any(key[0] == structure for key in lookup):
            continue
        record: dict[str, Any] = {"participant_structure": structure}
        for task in ("single", "competing"):
            for version in ("V1", "V2", "V3"):
                source = lookup.get((structure, task, version))
                for metric in metrics:
                    record[f"{task}_{version}_{metric}"] = source.get(metric) if source else None
        output.append(record)
    return output


def cost_by_model(details: dict[str, Any]) -> list[dict[str, Any]]:
    totals: defaultdict[str, float] = defaultdict(float)
    for detail in details.values():
        for model, cost in detail.get("cost_actual_total", {}).get("model_costs", {}).items():
            totals[model] += float(cost)
    return [{"model": model, "cost_usd": cost} for model, cost in sorted(totals.items(), key=lambda item: -item[1])]


def png_rgb(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(path)
    offset, width, height, compressed = 8, 0, 0, bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        name = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if name == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif name == b"IDAT":
            compressed.extend(body)
        elif name == b"IEND":
            break
    raw = zlib.decompress(bytes(compressed))
    stride = width * 3
    rows = []
    cursor = 0
    for _ in range(height):
        if raw[cursor] != 0:
            raise ValueError("contact-sheet source PNG uses an unsupported filter")
        cursor += 1
        rows.append(raw[cursor : cursor + stride])
        cursor += stride
    return width, height, b"".join(rows)


def contact_sheet(flagged: list[dict[str, Any]]) -> None:
    if not flagged:
        return
    thumb, columns, padding = 192, 4, 16
    rows_n = math.ceil(len(flagged) / columns)
    width = columns * thumb + (columns + 1) * padding
    height = rows_n * thumb + (rows_n + 1) * padding
    output = bytearray((248, 246, 240) * (width * height))
    for index, row in enumerate(flagged):
        source_path = ROOT / row["analysis_image"]
        sw, sh, source = png_rgb(source_path)
        x0 = padding + (index % columns) * (thumb + padding)
        y0 = padding + (index // columns) * (thumb + padding)
        for y in range(thumb):
            sy = min(sh - 1, y * sh // thumb)
            for x in range(thumb):
                sx = min(sw - 1, x * sw // thumb)
                src = (sy * sw + sx) * 3
                dst = ((y0 + y) * width + (x0 + x)) * 3
                output[dst : dst + 3] = source[src : src + 3]
    (RESULTS / "flagged_localization_contact_sheet.png").write_bytes(_png_bytes(width, height, output))
    (RESULTS / "flagged_localization_contact_sheet.json").write_text(
        json.dumps([{"position": i + 1, "run_key": row["run_key"], "flags": row["localization_flag_count"]} for i, row in enumerate(flagged)], indent=2),
        encoding="utf-8",
    )


def fmt(value: Any, digits: int = 3) -> str:
    number_value = number(value)
    return "—" if number_value is None else f"{number_value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def build_report(
    rows: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    agreement: dict[str, Any],
    stopping: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    forums: list[dict[str, Any]],
    model_costs: list[dict[str, Any]],
) -> str:
    complete = [row for row in rows if row["status"] == "complete"]
    failures = [row for row in rows if row["status"] != "complete"]
    judge_ledger = json.loads((JUDGES / "ledger.json").read_text(encoding="utf-8"))
    total_generation = sum(float(row.get("generation_cost_actual_total_usd") or 0) for row in rows)
    invalid = sum(int(float(row.get("invalid_output_responses") or 0)) for row in rows)

    condition_lookup = {
        (row["participant_structure"], row["task"], row["prompt_version"]): row for row in conditions
    }
    matrix_columns = [(task, version) for task in ("single", "competing") for version in ("V1", "V2", "V3")]
    matrix_headers = [
        "Participant structure",
        "Single V1", "Single V2", "Single V3",
        "Competition V1", "Competition V2", "Competition V3",
    ]

    def matrix(cell: Callable[[dict[str, Any]], str]) -> str:
        table_rows = []
        for structure in PARTICIPANT_STRUCTURE_ORDER:
            if not any((structure, task, version) in condition_lookup for task, version in matrix_columns):
                continue
            values = []
            for task, version in matrix_columns:
                row = condition_lookup.get((structure, task, version))
                values.append(cell(row) if row else "—")
            table_rows.append([structure, *values])
        return markdown_table(matrix_headers, table_rows)

    performance_matrix = matrix(
        lambda row: f"{fmt(row['reference_fidelity_mean'])} / {fmt(row['visual_quality_mean'])} / {fmt(row['progress_auc_mean'])}"
    )
    coordination_matrix = matrix(
        lambda row: (
            f"{fmt(row['pooled_intent_conflict_rate'])} / {fmt(row['pooled_destructive_overwrite_rate'])} / "
            f"{fmt(row['pooled_repair_rate'])} / {fmt(row['posts_per_100_turns_mean'])}"
        )
    )
    efficiency_matrix = matrix(
        lambda row: (
            f"${fmt(row['generation_cost_usd_mean'])} / {fmt(row['canonical_color_iou_repeatability'])} / "
            f"{row['complete_n']}/{row['planned_n']}"
        )
    )

    effect_rows = []
    wanted = {"forum_posts_total", "reference_fidelity", "visual_quality", "destructive_overwrite_rate"}
    outcome_names = {
        "forum_posts_total": "Forum posts/run",
        "reference_fidelity": "Reference fidelity",
        "visual_quality": "VLM quality",
        "destructive_overwrite_rate": "Destructive overwrite",
    }
    for row in effects:
        if row["metric"] in wanted:
            effect_rows.append(
                [
                    TASK_LABELS[row["task"]], row["contrast"], outcome_names[row["metric"]],
                    fmt(row["effect"]), f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
                    str(row["participant_structure_blocks"]),
                ]
            )

    def effect(task: str, contrast: str, metric: str) -> dict[str, Any]:
        return next(row for row in effects if row["task"] == task and row["contrast"] == contrast and row["metric"] == metric)

    competing_v3_quality = effect("competing", "V3-V1", "visual_quality")
    competing_v3_damage = effect("competing", "V3-V1", "destructive_overwrite_rate")
    competing_v3_fidelity = effect("competing", "V3-V1", "reference_fidelity")
    single_v3_quality = effect("single", "V3-V1", "visual_quality")
    single_v3_fidelity = effect("single", "V3-V1", "reference_fidelity")

    def condition(task: str, version: str, structure: str) -> dict[str, Any]:
        return condition_lookup[(structure, task, version)]

    mixed_v1 = condition("competing", "V1", "4 mixed agents (GLM, Gemini, Qwen, Luna)")
    mixed_v3 = condition("competing", "V3", "4 mixed agents (GLM, Gemini, Qwen, Luna)")
    family_v1 = condition("competing", "V1", "4 homogeneous agents; one model family per team")
    family_v3 = condition("competing", "V3", "4 homogeneous agents; one model family per team")
    gemini_competing = [condition("competing", version, "4 Gemini agents") for version in ("V1", "V2", "V3")]

    stopping_rows = [
        [
            TASK_LABELS[row["task"]], row["prompt_version"], str(row["team_complete_n"]),
            fmt(row["team_complete_gap_mean"]), str(row["stalled_n"]), fmt(row["stalled_gap_mean"]),
            str(row["max_steps_n"]), fmt(row["post_horizon_fidelity_gain_mean"], 4),
        ]
        for row in stopping
    ]
    forum_rows = [
        [
            TASK_LABELS[row["task"]], row["prompt_version"], str(row["team_posts_start"]),
            str(row["team_posts_mid"]), str(row["public_posts_start"]), str(row["public_posts_mid"]),
            fmt(row["forum_posts_total_per_100_turns"]), fmt(row["runs_using_forum_rate"]),
        ]
        for row in forums
    ]
    model_cost_rows = [
        [row["model"], f"${row['cost_usd']:.4f}", f"{100 * row['cost_usd'] / total_generation:.1f}%"]
        for row in model_costs
    ]

    prompt_rows = [
        [
            TASK_LABELS[row["task"]], row["prompt_version"], str(row["participant_structure_blocks"]),
            fmt(row["reference_fidelity"]), fmt(row["visual_quality"]), fmt(row["progress_auc"]),
            fmt(row["destructive_overwrite_rate"]), fmt(row["posts_per_100_turns"]),
            f"${fmt(row['generation_cost_usd'])}", fmt(row["canonical_color_iou_repeatability"]),
        ]
        for row in prompts
    ]

    return f"""# Collective Canvas: results by participant structure and task

## Core findings

The main result is an interaction between **task difficulty** and **communication protocol**. On the four-team competition task, V3 reduced destructive overwriting by **{fmt(abs(competing_v3_damage['effect']))}** (95% interval **[{fmt(competing_v3_damage['ci_low'])}, {fmt(competing_v3_damage['ci_high'])}]**) relative to V1. Blinded visual quality was **{fmt(competing_v3_quality['effect'])}** higher, although its interval **[{fmt(competing_v3_quality['ci_low'])}, {fmt(competing_v3_quality['ci_high'])}]** just includes zero. Exact reference fidelity increased by **{fmt(competing_v3_fidelity['effect'])}**, with a wider interval of **[{fmt(competing_v3_fidelity['ci_low'])}, {fmt(competing_v3_fidelity['ci_high'])}]**. V2 generated more messages but no measurable task-level improvement.

The V3 effect is concentrated in participant structures that faced real coordination problems. For four mixed-agent teams, fidelity moved from **{fmt(mixed_v1['reference_fidelity_mean'])}** to **{fmt(mixed_v3['reference_fidelity_mean'])}**, visual quality from **{fmt(mixed_v1['visual_quality_mean'])}** to **{fmt(mixed_v3['visual_quality_mean'])}**, and destructive overwriting from **{fmt(mixed_v1['pooled_destructive_overwrite_rate'])}** to **{fmt(mixed_v3['pooled_destructive_overwrite_rate'])}**. For the family-specialized homogeneous teams, the corresponding changes were **{fmt(family_v1['reference_fidelity_mean'])}→{fmt(family_v3['reference_fidelity_mean'])}**, **{fmt(family_v1['visual_quality_mean'])}→{fmt(family_v3['visual_quality_mean'])}**, and **{fmt(family_v1['pooled_destructive_overwrite_rate'])}→{fmt(family_v3['pooled_destructive_overwrite_rate'])}**.

Four-Gemini teams were already at ceiling: across V1–V3 they achieved **{min(float(row['reference_fidelity_mean']) for row in gemini_competing):.3f}–{max(float(row['reference_fidelity_mean']) for row in gemini_competing):.3f}** fidelity, **1.000** judged quality, and **{min(float(row['canonical_color_iou_repeatability']) for row in gemini_competing):.3f}–{max(float(row['canonical_color_iou_repeatability']) for row in gemini_competing):.3f}** repeatability. Communication therefore had little outcome headroom in this configuration.

On the single-team task, **model composition mattered more than the communication prompt**. Gemini teams were nearly perfect, mixed teams were the next strongest, and the other homogeneous families were less accurate and less repeatable. V3 did not reliably improve the final image: its participant-structure-balanced effects were **{fmt(single_v3_fidelity['effect'])}** for fidelity (**[{fmt(single_v3_fidelity['ci_low'])}, {fmt(single_v3_fidelity['ci_high'])}]**) and **{fmt(single_v3_quality['effect'])}** for judged quality (**[{fmt(single_v3_quality['ci_low'])}, {fmt(single_v3_quality['ci_high'])}]**). Increasing a homogeneous team from two to four agents also produced no consistent gain across model families.

The participant structure also sets the cost–performance frontier. In V3 competition, four-Gemini teams reached **{fmt(gemini_competing[2]['reference_fidelity_mean'])}** fidelity and **{fmt(gemini_competing[2]['visual_quality_mean'])}** quality for **${fmt(gemini_competing[2]['generation_cost_usd_mean'])}** per match; mixed teams reached **{fmt(mixed_v3['reference_fidelity_mean'])}** and **{fmt(mixed_v3['visual_quality_mean'])}** for **${fmt(mixed_v3['generation_cost_usd_mean'])}**. The Gemini ceiling is stronger, while the mixed configuration exposes more of the coordination phenomenon at lower cost.

V3 primarily **prevented damage** rather than improving recovery after damage. Across competing runs, persistent damage episodes fell from **{next(row['persistent_damage_episodes'] for row in conflicts if row['task'] == 'competing' and row['prompt_version'] == 'V1')}** under V1 to **{next(row['persistent_damage_episodes'] for row in conflicts if row['task'] == 'competing' and row['prompt_version'] == 'V3')}** under V3. The observed repair fraction did not rise, because far fewer repairable incidents occurred.

## Metric matrices

Rows are participant structures. Columns are the two tasks, split by prompt version. A dash means that participant structure was not run on that task. The single-team task contains one team drawing one apple; the competition task contains four teams drawing four different objects on the same 32×32 canvas, and each row describes the membership of each team. “Top-up” was only a batch-execution label and is not an analysis category.

### Output performance

Each cell is **exact reference fidelity / blinded VLM quality / progress AUC**. All are on a 0–1 scale; higher is better.

{performance_matrix}

### Coordination behavior

Each cell is **intent-conflict rate / destructive-overwrite rate / observed repair rate / forum posts per 100 turns**. Lower is better for the first two; higher repair means a larger fraction of observed damage was later restored by the affected team. A repair dash means no qualifying damage occurred.

{coordination_matrix}

### Efficiency and reliability

Each cell is **mean generation cost / cross-repetition exact-color IoU / completed runs over planned runs**. Higher IoU means more repeatable outputs.

{efficiency_matrix}

## Prompt intervention estimates

These are participant-structure-balanced differences from V1, with 5,000-resample run-level bootstrap intervals. Positive fidelity and quality are favorable; negative destructive overwrite is favorable.

{markdown_table(['Task','Contrast','Outcome','Effect','95% interval','Structures'], effect_rows)}

## Communication timing

Start means R0–R1; mid-run means R2 onward. V3 is the only prompt that produced substantial mid-run communication in the competition task.

{markdown_table(['Task','Prompt','Team start','Team mid','Public start','Public mid','Posts/100 turns','Runs using forum'], forum_rows)}

## Progress and stopping calibration

Completion and stall gaps are `1 - final fidelity`; smaller is better. Maximum-round stops are censored rather than treated as completion claims.

{markdown_table(['Task','Prompt','Completed','Completion gap','Stalled','Stall gap','Max-round','Post-horizon gain'], stopping_rows)}

The extra V3 single-team rounds beyond the common 10-round horizon added only **{fmt(next(row['post_horizon_fidelity_gain_mean'] for row in stopping if row['task'] == 'single' and row['prompt_version'] == 'V3'), 4)}** mean fidelity. The matched horizon therefore captures essentially all realized output quality.

## Task-level descriptive aggregates

These values give each participant structure equal weight. They are descriptive summaries; the prompt-intervention table above gives the uncertainty estimates.

{markdown_table(['Task','Prompt','Structures','Fidelity','VLM quality','Progress AUC','Destructive rate','Posts/100','Cost/run','Repeatability'], prompt_rows)}

## Data and validation

- Planned experimental slots: **{len(rows)}**; completed and behaviorally scored: **{len(complete)}**; provider-failed: **{len(failures)}**.
- Single-team completed runs: **{sum(r['mode'] == 'single' and r['status'] == 'complete' for r in rows)}**; competing-team completed matches: **{sum(r['mode'] == 'competing' and r['status'] == 'complete' for r in rows)}**.
- All **{len(complete)}** completed histories replayed exactly to their saved canvases. The five technical failures were 529 provider-overload failures and remain in completion and cost accounting.
- Experimental generation cost, including failed work and billed invalid attempts: **${total_generation:.4f}**. Billed invalid-output responses: **{invalid}**.
- Primary comparisons use matched horizons: 10 ordinary rounds for single-team and 20 for competing-team runs.
- “Observed repair” requires the affected team to see damage and later restore the reference-consistent pixel. Simultaneous reversals and opponent restorations do not count.

## Experimental cost by model

{markdown_table(['Model','Billed cost','Share'], model_cost_rows)}

Prompt-version cost differences should not be read causally because versions ran at different times and providers showed different invalid-output behavior.

## Judge reliability

- Grok repeat sample: **n={agreement['grok_repeat']['n']}**, visual-quality rank correlation **{agreement['grok_repeat']['visual_quality_spearman']:.3f}**, mean absolute difference **{agreement['grok_repeat']['visual_quality_mean_absolute_difference']:.3f}** on the normalized 0–1 scale.
- Kimi independent audit: **n={agreement['kimi_audit']['n']}**, cross-model rank correlation **{agreement['kimi_audit']['visual_quality_spearman']:.3f}**, mean absolute difference **{agreement['kimi_audit']['visual_quality_mean_absolute_difference']:.3f}**.
- The separate 12-canvas competing calibration passed: rank correlation **{judge_ledger['calibration']['visual_quality_spearman']:.3f}**, component MAD **{judge_ledger['calibration']['component_mean_absolute_difference']:.3f}** points on the 0–4 scales.

## Limits requiring care

- Fourteen competing matches triggered an automatic placement ambiguity or overlap flag. Visual inspection confirmed genuinely crowded or overlapping drawings rather than obvious object-assignment errors; no score was manually altered.
- Pixel logs establish observed, team-attributable repair but cannot establish private intent.
- Prompt versions were executed at different times. The matched comparisons are informative for this experiment, while provider-time drift remains a nuisance variable.
- Participant-count comparisons are descriptive because the two-agent and four-agent batches were not randomized together.
"""


def main() -> None:
    deterministic = read_csv(RESULTS / "run_metrics_deterministic.csv")
    judge_scores = read_csv(JUDGES / "judge_scores.csv")
    primary_scores = {row["run_key"]: row for row in judge_scores if row["stage"] == "primary"}
    merged: list[dict[str, Any]] = []
    for row in deterministic:
        value: dict[str, Any] = dict(row)
        score = primary_scores.get(row["run_key"])
        if score:
            for key, item in score.items():
                if key not in {"stage", "judge_model", "run_key", "mode", "reason"}:
                    value[key] = item
            value["judge_reason"] = score.get("reason", "")
        merged.append(value)
    complete_without_judge = [row["run_key"] for row in merged if row["status"] == "complete" and number(row.get("visual_quality")) is None]
    if complete_without_judge:
        raise RuntimeError(f"missing primary judge scores for {len(complete_without_judge)} complete runs")

    details = json.loads((RESULTS / "run_details.json").read_text(encoding="utf-8"))
    conditions = aggregate_conditions(merged, details)
    configuration_matrix = wide_configuration_task_matrix(conditions)
    prompts = prompt_summary(conditions)
    effects = communication_effects(merged)
    agreement = judge_agreement(judge_scores)
    stopping = stopping_summary(merged)
    conflicts = conflict_summary(merged)
    forums = forum_summary(merged)
    objects = object_metrics(merged, details)
    model_costs = cost_by_model(details)
    flagged = [row for row in merged if row["status"] == "complete" and int(float(row.get("localization_flag_count") or 0)) > 0]
    contact_sheet(flagged)

    write_csv(RESULTS / "run_metrics_all.csv", merged)
    write_csv(RESULTS / "condition_metrics.csv", conditions)
    write_csv(RESULTS / "configuration_task_matrix.csv", configuration_matrix)
    write_csv(RESULTS / "prompt_summary.csv", prompts)
    write_csv(RESULTS / "communication_effects.csv", effects)
    write_csv(RESULTS / "stopping_summary.csv", stopping)
    write_csv(RESULTS / "conflict_repair_summary.csv", conflicts)
    write_csv(RESULTS / "forum_summary.csv", forums)
    write_csv(RESULTS / "object_metrics.csv", objects)
    write_csv(RESULTS / "cost_by_model.csv", model_costs)
    (RESULTS / "judge_agreement.json").write_text(json.dumps(agreement, indent=2), encoding="utf-8")
    report = build_report(merged, conditions, prompts, effects, agreement, stopping, conflicts, forums, model_costs)
    (RESULTS / "RESULTS_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "runs": len(merged),
        "complete": sum(row["status"] == "complete" for row in merged),
        "condition_rows": len(conditions),
        "communication_effect_rows": len(effects),
        "object_metric_rows": len(objects),
        "flagged_localization_runs": len(flagged),
        "judge_agreement": agreement,
        "report": str(RESULTS / "RESULTS_REPORT.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
