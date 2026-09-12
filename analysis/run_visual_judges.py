"""Run blinded multimodal judges for metric 2.

The job is resumable, uses 20 concurrent requests, never changes models or
providers on error, and only applies the long backoff schedule to real HTTP 429s.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import random
import statistics
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
RESULTS = ANALYSIS / "results"
JUDGES = RESULTS / "judges"
RESPONSES = JUDGES / "responses"
LOG = JUDGES / "judge.log"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "x-ai/grok-4.6"
AUDIT_MODEL = "moonshotai/kimi-k3"
CONCURRENCY = 20
RATE_BACKOFF = (10, 10, 20, 20, 60, 60, 120)
TOTAL_JUDGE_GUARD_USD = 4.50
REFERENCE_TAGS = ("apple", "orange", "lemon", "eggplant")
_PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    with _PRINT_LOCK:
        print(line, flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_env() -> None:
    path = ROOT / "harness" / ".env"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


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


def image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def score_fields() -> dict[str, Any]:
    return {
        "recognizability": {"type": "integer", "minimum": 0, "maximum": 4},
        "reference_fidelity": {"type": "integer", "minimum": 0, "maximum": 4},
        "visual_coherence": {"type": "integer", "minimum": 0, "maximum": 4},
        "reason": {"type": "string", "maxLength": 120},
    }


def response_schema(mode: str, candidate_count: int) -> dict[str, Any]:
    if mode == "single":
        item = {
            "type": "object",
            "properties": score_fields(),
            "required": ["recognizability", "reference_fidelity", "visual_coherence", "reason"],
            "additionalProperties": False,
        }
    else:
        object_item = {
            "type": "object",
            "properties": {
                "recognizability": {"type": "integer", "minimum": 0, "maximum": 4},
                "reference_fidelity": {"type": "integer", "minimum": 0, "maximum": 4},
            },
            "required": ["recognizability", "reference_fidelity"],
            "additionalProperties": False,
        }
        item = {
            "type": "object",
            "properties": {
                "objects": {"type": "array", "minItems": 4, "maxItems": 4, "items": object_item},
                "separation_coherence": {"type": "integer", "minimum": 0, "maximum": 4},
                "reason": {"type": "string", "maxLength": 140},
            },
            "required": ["objects", "separation_coherence", "reason"],
            "additionalProperties": False,
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"canvas_{mode}_visual_scores",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "minItems": candidate_count,
                        "maxItems": candidate_count,
                        "items": item,
                    }
                },
                "required": ["candidates"],
                "additionalProperties": False,
            },
        },
    }


def prompt_for(mode: str) -> str:
    if mode == "single":
        return (
            "You are a blinded visual evaluator of pixel art. The first image is the 16x16 reference apple. "
            "The later images are independently generated 32x32 canvases. Score every candidate independently, "
            "in exactly the order presented; do not rank them or let one alter another's score. Ignore translation. "
            "Use integers 0-4: recognizability = how clearly it depicts an apple; reference_fidelity = shape and "
            "color agreement with the reference; visual_coherence = clean, connected, intentional pixel art. "
            "A 4 is excellent, 3 good with small defects, 2 recognizable but materially flawed, 1 barely recognizable, "
            "0 absent or wrong. Return score objects in presentation order. Return only the required JSON."
        )
    return (
        "You are a blinded visual evaluator of a shared pixel-art canvas. The first four images are references in "
        "this fixed order: apple, orange, lemon, eggplant. The later images are independently generated 32x32 shared "
        "canvases on which four teams attempted those four objects. Score every candidate independently and in exactly "
        "the order presented. For each candidate, return four object scores in the fixed reference order. Use integers "
        "0-4: recognizability = how clearly that object is visible; reference_fidelity = shape and color agreement with "
        "its reference. separation_coherence = whether all four objects remain individually legible and spatially "
        "organized despite overlap or competition. A 4 is excellent, 3 good with small defects, 2 materially flawed, "
        "1 barely visible, 0 absent or wrong. Return only the required JSON."
    )


def validate_scores(mode: str, parsed: dict[str, Any], candidate_count: int) -> None:
    values = parsed.get("candidates")
    if not isinstance(values, list) or len(values) != candidate_count:
        raise ValueError(f"expected {candidate_count} candidates")
    for candidate in values:
        if not isinstance(candidate, dict):
            raise ValueError("candidate is not an object")
        if mode == "single":
            keys = ("recognizability", "reference_fidelity", "visual_coherence")
            for key in keys:
                if not isinstance(candidate.get(key), int) or not 0 <= candidate[key] <= 4:
                    raise ValueError(f"invalid {key}")
        else:
            objects = candidate.get("objects")
            if not isinstance(objects, list) or len(objects) != 4:
                raise ValueError("competing candidate must contain four object scores")
            for obj in objects:
                for key in ("recognizability", "reference_fidelity"):
                    if not isinstance(obj.get(key), int) or not 0 <= obj[key] <= 4:
                        raise ValueError(f"invalid object {key}")
            if not isinstance(candidate.get("separation_coherence"), int) or not 0 <= candidate["separation_coherence"] <= 4:
                raise ValueError("invalid separation_coherence")


def build_payload(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = task["mode"]
    candidates = [ROOT / value for value in task["images"]]
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_for(mode)}]
    if mode == "single":
        references = [ROOT / "harness" / "reference_images" / "apple_16.png"]
    else:
        references = [ROOT / "harness" / "reference_images" / f"{tag}_16.png" for tag in REFERENCE_TAGS]
    for index, path in enumerate(references, 1):
        label = "REFERENCE" if mode == "single" else f"REFERENCE {index}: {REFERENCE_TAGS[index - 1]}"
        content += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": image_data_url(path)}}]
    for index, path in enumerate(candidates, 1):
        content += [{"type": "text", "text": f"CANDIDATE {index}"}, {"type": "image_url", "image_url": {"url": image_data_url(path)}}]
    payload = {
        "model": task["model"],
        "messages": [{"role": "user", "content": content}],
        "response_format": response_schema(mode, len(candidates)),
        "reasoning": {"effort": "low", "exclude": True},
        "temperature": 0,
        "max_tokens": 1800 if mode == "single" else 3000,
        "stream": False,
    }
    preview = json.loads(json.dumps(payload))
    paths = references + candidates
    for item, path in zip((x for x in preview["messages"][0]["content"] if x.get("type") == "image_url"), paths, strict=True):
        item["image_url"]["url"] = f"<base64 PNG: {path.relative_to(ROOT).as_posix()}, {path.stat().st_size} bytes>"
    return payload, preview


def parse_content(response: dict[str, Any], mode: str, candidate_count: int) -> dict[str, Any]:
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("response contained no text")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(text)
    validate_scores(mode, parsed, candidate_count)
    return parsed


def execute_task(task: dict[str, Any], api_key: str) -> dict[str, Any]:
    path = RESPONSES / f"{task['task_id']}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("status") == "complete":
            return saved
    payload, preview = build_payload(task)
    invalid_attempts = 0
    rate_attempts = 0
    transient_attempts = 0
    while True:
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "Collective Canvas Metrics",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response_handle:
                response = json.load(response_handle)
            try:
                scores = parse_content(response, task["mode"], len(task["run_keys"]))
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                invalid_path = RESPONSES / f"{task['task_id']}.invalid_{invalid_attempts + 1:02d}.json"
                invalid_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
                if invalid_attempts >= 3:
                    raise RuntimeError(f"invalid output after four attempts: {exc}") from exc
                invalid_attempts += 1
                log(f"{task['task_id']} invalid output; immediate retry {invalid_attempts}/3: {exc}")
                continue
            saved = {
                "status": "complete",
                "task": task,
                "request_preview": preview,
                "response": response,
                "scores": scores,
                "invalid_retries": invalid_attempts,
                "rate_limit_retries": rate_attempts,
                "transient_retries": transient_attempts,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            return saved
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and rate_attempts < len(RATE_BACKOFF):
                delay = RATE_BACKOFF[rate_attempts]
                rate_attempts += 1
                log(f"{task['task_id']} real HTTP 429; backoff {delay}s ({rate_attempts}/{len(RATE_BACKOFF)})")
                time.sleep(delay)
                continue
            if exc.code in {408, 425, 500, 502, 503, 504, 529} and transient_attempts < 5:
                delay = (2, 5, 10, 20, 40)[transient_attempts]
                transient_attempts += 1
                log(f"{task['task_id']} transient HTTP {exc.code}; retry in {delay}s")
                time.sleep(delay)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            if transient_attempts < 5:
                delay = (2, 5, 10, 20, 40)[transient_attempts]
                transient_attempts += 1
                log(f"{task['task_id']} connection/timeout; retry in {delay}s: {exc}")
                time.sleep(delay)
                continue
            raise


def existing_cost() -> float:
    total = 0.0
    seen: set[str] = set()
    files = list(RESPONSES.glob("*.json")) if RESPONSES.exists() else []
    files += list((ANALYSIS / "metric_design_validation").glob("*.json"))
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        response = data.get("response", data)
        response_id = response.get("id") if isinstance(response, dict) else None
        usage = response.get("usage") if isinstance(response, dict) else None
        if response_id and isinstance(usage, dict) and response_id not in seen:
            seen.add(response_id)
            total += float(usage.get("cost") or 0)
    return total


def run_tasks(tasks: list[dict[str, Any]], api_key: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending = [task for task in tasks if not (RESPONSES / f"{task['task_id']}.json").exists()]
    completed_saved = [task for task in tasks if (RESPONSES / f"{task['task_id']}.json").exists()]
    for task in completed_saved:
        results.append(json.loads((RESPONSES / f"{task['task_id']}.json").read_text(encoding="utf-8")))
    for start in range(0, len(pending), CONCURRENCY):
        current_cost = existing_cost()
        if current_cost >= TOTAL_JUDGE_GUARD_USD:
            raise RuntimeError(f"judge cost guard reached ${current_cost:.4f}")
        wave = pending[start : start + CONCURRENCY]
        log(f"starting wave {start // CONCURRENCY + 1}: {len(wave)} requests, accumulated judge cost ${current_cost:.4f}")
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(execute_task, task, api_key): task for task in wave}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    cost = float(result["response"].get("usage", {}).get("cost") or 0)
                    log(f"complete {task['task_id']} model={task['model']} cost=${cost:.4f}")
                except Exception as exc:
                    failure = {"status": "failed", "task": task, "error": str(exc)}
                    (RESPONSES / f"{task['task_id']}.failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
                    log(f"FAILED {task['task_id']}: {exc}")
                    raise
    return results


def batched_tasks(prefix: str, rows: list[dict[str, str]], mode: str, model: str, size: int, rng: random.Random) -> list[dict[str, Any]]:
    ordered = rows[:]
    rng.shuffle(ordered)
    tasks = []
    for index in range(0, len(ordered), size):
        group = ordered[index : index + size]
        tasks.append(
            {
                "task_id": f"{prefix}_{mode}_{index // size:03d}",
                "stage": prefix,
                "mode": mode,
                "model": model,
                "run_keys": [row["run_key"] for row in group],
                "images": [row["analysis_image"] for row in group],
            }
        )
    return tasks


def quantile_sample(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: float(row["reference_fidelity"]))
    if len(ordered) <= count:
        return ordered
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return [ordered[index] for index in indices]


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return math.nan
    a, b = average_ranks(left), average_ranks(right)
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return numerator / denominator if denominator else 1.0


def flatten_result(saved: dict[str, Any]) -> list[dict[str, Any]]:
    task = saved["task"]
    rows = []
    for run_key, score in zip(task["run_keys"], saved["scores"]["candidates"], strict=True):
        if task["mode"] == "single":
            rows.append(
                {
                    "stage": task["stage"],
                    "judge_model": task["model"],
                    "run_key": run_key,
                    "mode": "single",
                    "recognizability": score["recognizability"],
                    "subjective_reference_fidelity": score["reference_fidelity"],
                    "visual_coherence": score["visual_coherence"],
                    "visual_quality": (score["recognizability"] + score["visual_coherence"]) / 8.0,
                    "reason": score.get("reason", ""),
                }
            )
        else:
            object_scores = score["objects"]
            row: dict[str, Any] = {
                "stage": task["stage"],
                "judge_model": task["model"],
                "run_key": run_key,
                "mode": "competing",
                "separation_coherence": score["separation_coherence"],
                "mean_object_recognizability": statistics.mean(obj["recognizability"] for obj in object_scores),
                "mean_subjective_reference_fidelity": statistics.mean(obj["reference_fidelity"] for obj in object_scores),
                "reason": score.get("reason", ""),
            }
            row["visual_quality"] = (row["mean_object_recognizability"] + score["separation_coherence"]) / 8.0
            for tag, obj in zip(REFERENCE_TAGS, object_scores, strict=True):
                row[f"{tag}_recognizability"] = obj["recognizability"]
                row[f"{tag}_subjective_fidelity"] = obj["reference_fidelity"]
            rows.append(row)
    return rows


def calibration_report(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    flat_a = {row["run_key"]: row for saved in first for row in flatten_result(saved)}
    flat_b = {row["run_key"]: row for saved in second for row in flatten_result(saved)}
    keys = sorted(set(flat_a) & set(flat_b))
    dimensions = ["separation_coherence", "mean_object_recognizability", "mean_subjective_reference_fidelity"]
    differences = [abs(float(flat_a[key][dimension]) - float(flat_b[key][dimension])) for key in keys for dimension in dimensions]
    left = [float(flat_a[key]["visual_quality"]) for key in keys]
    right = [float(flat_b[key]["visual_quality"]) for key in keys]
    report = {
        "n_canvases": len(keys),
        "component_mean_absolute_difference": statistics.mean(differences),
        "visual_quality_spearman": correlation(left, right),
        "passes_mad": statistics.mean(differences) <= 0.5,
        "passes_rank": correlation(left, right) >= 0.75,
    }
    report["passed"] = report["passes_mad"] and report["passes_rank"] and len(keys) == 12
    return report


def stratified_audit(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["prompt_version"], row["condition"])].append(row)
    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        selected.extend(group[: max(1, round(0.20 * len(group)))])
    return selected


def main() -> None:
    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    RESPONSES.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_csv(RESULTS / "run_metrics_deterministic.csv") if row["status"] == "complete"]
    single = [row for row in rows if row["mode"] == "single"]
    competing = [row for row in rows if row["mode"] == "competing"]
    rng = random.Random(20260912)

    # Calibrate the more complex four-object rubric before production.
    calibration = quantile_sample(competing, 12)
    cal_a = batched_tasks("calibration_forward", calibration, "competing", PRIMARY_MODEL, 3, rng)
    cal_b_rows = list(reversed(calibration))
    cal_b = batched_tasks("calibration_reverse", cal_b_rows, "competing", PRIMARY_MODEL, 3, random.Random(20260913))
    first = run_tasks(cal_a, api_key)
    second = run_tasks(cal_b, api_key)
    report = calibration_report(first, second)
    (JUDGES / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"competing calibration: {json.dumps(report, sort_keys=True)}")
    if not report["passed"]:
        raise RuntimeError("competing visual-judge calibration did not pass preregistered thresholds")

    primary_tasks = batched_tasks("primary", single, "single", PRIMARY_MODEL, 5, random.Random(20260914))
    primary_tasks += batched_tasks("primary", competing, "competing", PRIMARY_MODEL, 3, random.Random(20260915))
    primary = run_tasks(primary_tasks, api_key)

    audit_rows = stratified_audit(rows, 20260916)
    audit_single = [row for row in audit_rows if row["mode"] == "single"]
    audit_competing = [row for row in audit_rows if row["mode"] == "competing"]
    repeat_tasks = batched_tasks("grok_repeat", audit_single, "single", PRIMARY_MODEL, 5, random.Random(20260917))
    repeat_tasks += batched_tasks("grok_repeat", audit_competing, "competing", PRIMARY_MODEL, 3, random.Random(20260918))
    audit_tasks = batched_tasks("kimi_audit", audit_single, "single", AUDIT_MODEL, 5, random.Random(20260919))
    audit_tasks += batched_tasks("kimi_audit", audit_competing, "competing", AUDIT_MODEL, 3, random.Random(20260920))
    repeats = run_tasks(repeat_tasks, api_key)
    audits = run_tasks(audit_tasks, api_key)

    score_rows = [row for saved in primary + repeats + audits for row in flatten_result(saved)]
    write_csv(JUDGES / "judge_scores.csv", score_rows)
    ledger = {
        "primary_model": PRIMARY_MODEL,
        "audit_model": AUDIT_MODEL,
        "concurrency": CONCURRENCY,
        "primary_canvases": len(primary and [row for saved in primary for row in flatten_result(saved)]),
        "repeat_canvases": len([row for saved in repeats for row in flatten_result(saved)]),
        "audit_canvases": len([row for saved in audits for row in flatten_result(saved)]),
        "cost_including_metric_design_validation_usd": existing_cost(),
        "hard_guard_usd": TOTAL_JUDGE_GUARD_USD,
        "calibration": report,
    }
    (JUDGES / "ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    log(f"judge work complete: {json.dumps(ledger, sort_keys=True)}")


if __name__ == "__main__":
    main()
