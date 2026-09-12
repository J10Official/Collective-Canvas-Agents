# Collective Canvas

Collective Canvas tests when language-model agents use communication to coordinate work in a shared visual environment. Agents repeatedly receive the same labeled pixel canvas and can either edit up to eight pixels, post to an available forum, or skip. The repository supports one team drawing an apple and four competing teams drawing an apple, orange, lemon, and eggplant.

The completed report is available in [`docs/Collective_Canvas_Report.pdf`](docs/Collective_Canvas_Report.pdf). Balanced run-level and aggregate results are in [`results/`](results/).

## Requirements

- Windows with Python 3.11 or newer
- An OpenRouter API key for model runs
- A modern browser for the interactive interface

The harness and deterministic analysis use the Python standard library. No installation step is required.

## Setup

1. Copy `harness/.env.example` to `harness/.env`.
2. Put the key after `OPENROUTER_API_KEY=` in `harness/.env`.
3. Do not commit `.env`; it is ignored by Git.

## Interactive interface

Run [`start_canvas.cmd`](start_canvas.cmd). It starts the server at <http://127.0.0.1:8767/> and opens the interface. The UI configures canvas size, objects, reference images, team composition, model routes, generation settings, forums, prompt version, stopping rules, and resumable runs.

The canvas can also be edited from another terminal:

```powershell
python -m harness.cli place 4 8 blue
python -m harness.cli erase 4 8
python -m harness.cli random --count 100 --delay 0.04 --seed 7
python -m harness.cli clear
```

## Registered batches

Every command saves prompts, request previews, input images, raw responses, parsed actions, snapshots, usage, costs, final canvases, terminal logs, and resumable state under `harness/batches/`. Provider fallbacks are disabled. Re-running a command skips completed jobs and resumes interrupted jobs.

| Task | Prompt | Command | Runs |
|---|---|---|---:|
| Single team: two-agent homogeneous and four-agent mixed teams | V1: forum available | `run_single_v1_core.cmd` | 40 |
| Single team: four-agent homogeneous teams | V1 | `run_single_v1_four_agent.cmd` | 20 |
| Single team: two-agent homogeneous and four-agent mixed teams | V2: communication suggestions | `run_single_v2_core.cmd` | 40 |
| Single team: four-agent homogeneous teams | V2 | `run_single_v2_four_agent.cmd` | 20 |
| Single team: two-agent homogeneous and four-agent mixed teams | V3: explicit protocol | `run_single_v3_core.cmd` | 40 |
| Single team: four-agent homogeneous teams | V3 | `run_single_v3_four_agent.cmd` | 20 |
| Four competing teams; three participant structures | V1 | `run_competing_v1.cmd` | 15 |
| Four competing teams; three participant structures | V2 | `run_competing_v2.cmd` | 15 |
| Four competing teams; three participant structures | V3 | `run_competing_v3.cmd` | 15 |

The core and four-agent commands are separate because they reproduce the two collection phases used in the study. The balanced analysis uses five repetitions per final configuration; the core batches intentionally retain the original ten repetitions.

The model routes are:

- `google/gemini-3.8-flash` via `google-ai-studio/flex`
- `qwen/qwen3.8-27b` via `reka/fp8`
- `z-ai/glm-5.3-flash` via the provider recorded in each configuration
- `openai/gpt-5.6-luna` via `openai/flex`

Review the cost limit and concurrency fields in [`configs/`](configs/) before starting paid batches.

Validate any batch without making API calls:

```powershell
python -m harness.single_batch --config configs\single_v1_core.json --validate-only
python -m harness.competing_batch --config configs\competing_v1.json --validate-only
```

Run a full local scheduler test with dummy model responses:

```powershell
python -m harness.single_batch --config configs\single_v1_core.json --dummy
python -m harness.competing_batch --config configs\competing_v1.json --dummy
```

## Prompts and outputs

The three prompt templates are in `harness/prompt_versions/`. V1 states that the forum exists, V2 gives situations where communication may help, and V3 specifies a communication protocol. Runtime variables such as team identity, object, round, coordinate bounds, canvas size, and forum history are rendered separately for every agent call.

Each successful run contains `config.json`, `history.json`, `usage.json`, `forums.json`, `final_canvas.json`, `output.png`, and per-call records under `inputs/`, `prompts/`, `requests/`, `responses/`, `actions/`, and `snapshots/`. Retry records and partial-round state are retained so transient failures do not discard earlier work.

## Analysis

The final seven-metric definition is in [`analysis/SEVEN_METRIC_PROTOCOL.md`](analysis/SEVEN_METRIC_PROTOCOL.md). The pipeline is split into deterministic measurement, optional visual judging, and aggregation:

```powershell
python -m analysis.metrics_pipeline
python -m analysis.run_visual_judges
python -m analysis.finalize_analysis
```

The visual-judge step uses OpenRouter and incurs API cost. The scripts read generated batches from `harness/batches/` and write derived files under `analysis/results/`.

## Tests

```powershell
python -m unittest harness.test_harness -v
```

## Repository layout

```text
configs/                  Registered experimental configurations
harness/                  Server, UI, agents, prompts, batch runners, references
analysis/                 Final metric pipeline and protocol
results/                  Balanced results used in the report
docs/                     Final report and concise results narrative
run_*.cmd                 Reproducible batch entry points
start_canvas.cmd          Interactive interface entry point
```
