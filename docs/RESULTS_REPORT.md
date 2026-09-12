# Collective Canvas: results by participant structure and task

## Core findings

The main result is an interaction between **task difficulty** and **communication protocol**. On the four-team competition task, V3 reduced destructive overwriting by **0.060** (95% interval **[-0.111, -0.023]**) relative to V1. Blinded visual quality was **0.095** higher, although its interval **[-0.001, 0.188]** just includes zero. Exact reference fidelity increased by **0.039**, with a wider interval of **[-0.018, 0.092]**. V2 generated more messages but no measurable task-level improvement.

The V3 effect is concentrated in participant structures that faced real coordination problems. For four mixed-agent teams, fidelity moved from **0.661** to **0.760**, visual quality from **0.613** to **0.828**, and destructive overwriting from **0.129** to **0.047**. For the family-specialized homogeneous teams, the corresponding changes were **0.636→0.672**, **0.555→0.625**, and **0.146→0.038**.

Four-Gemini teams were already at ceiling: across V1–V3 they achieved **0.965–0.984** fidelity, **1.000** judged quality, and **0.924–0.954** repeatability. Communication therefore had little outcome headroom in this configuration.

On the single-team task, **model composition mattered more than the communication prompt**. Gemini teams were nearly perfect, mixed teams were the next strongest, and the other homogeneous families were less accurate and less repeatable. V3 did not reliably improve the final image: its participant-structure-balanced effects were **0.003** for fidelity (**[-0.018, 0.024]**) and **0.042** for judged quality (**[-0.023, 0.106]**). Increasing a homogeneous team from two to four agents also produced no consistent gain across model families.

The participant structure also sets the cost–performance frontier. In V3 competition, four-Gemini teams reached **0.965** fidelity and **1.000** quality for **$1.010** per match; mixed teams reached **0.760** and **0.828** for **$0.661**. The Gemini ceiling is stronger, while the mixed configuration exposes more of the coordination phenomenon at lower cost.

V3 primarily **prevented damage** rather than improving recovery after damage. Across competing runs, persistent damage episodes fell from **1407** under V1 to **338** under V3. The observed repair fraction did not rise, because far fewer repairable incidents occurred.

## Metric matrices

Rows are participant structures. Columns are the two tasks, split by prompt version. A dash means that participant structure was not run on that task. The single-team task contains one team drawing one apple; the competition task contains four teams drawing four different objects on the same 32×32 canvas, and each row describes the membership of each team. “Top-up” was only a batch-execution label and is not an analysis category.

### Output performance

Each cell is **exact reference fidelity / blinded VLM quality / progress AUC**. All are on a 0–1 scale; higher is better.

| Participant structure | Single V1 | Single V2 | Single V3 | Competition V1 | Competition V2 | Competition V3 |
|---|---|---|---|---|---|---|
| 2 Gemini agents | 0.995 / 1.000 / 0.728 | 1.000 / 1.000 / 0.765 | 0.989 / 0.975 / 0.761 | — | — | — |
| 2 Qwen agents | 0.874 / 0.725 / 0.645 | 0.812 / 0.562 / 0.601 | 0.848 / 0.625 / 0.636 | — | — | — |
| 2 Luna agents | 0.844 / 0.800 / 0.659 | 0.862 / 0.713 / 0.663 | 0.854 / 0.700 / 0.620 | — | — | — |
| 4 Gemini agents | 0.977 / 0.950 / 0.719 | 1.000 / 1.000 / 0.784 | 1.000 / 1.000 / 0.851 | 0.984 / 1.000 / 0.850 | 0.968 / 1.000 / 0.864 | 0.965 / 1.000 / 0.892 |
| 4 GLM agents | 0.849 / 0.525 / 0.710 | 0.746 / 0.450 / 0.649 | 0.797 / 0.775 / 0.664 | — | — | — |
| 4 Qwen agents | 0.805 / 0.625 / 0.679 | 0.792 / 0.475 / 0.636 | 0.835 / 0.550 / 0.671 | — | — | — |
| 4 Luna agents | 0.848 / 0.575 / 0.696 | 0.869 / 0.625 / 0.709 | 0.881 / 0.775 / 0.670 | — | — | — |
| 4 mixed agents (GLM, Gemini, Qwen, Luna) | 0.900 / 0.700 / 0.693 | 0.917 / 0.738 / 0.697 | 0.912 / 0.838 / 0.751 | 0.661 / 0.613 / 0.585 | 0.715 / 0.719 / 0.608 | 0.760 / 0.828 / 0.650 |
| 4 homogeneous agents; one model family per team | — | — | — | 0.636 / 0.555 / 0.590 | 0.578 / 0.453 / 0.505 | 0.672 / 0.625 / 0.590 |

### Coordination behavior

Each cell is **intent-conflict rate / destructive-overwrite rate / observed repair rate / forum posts per 100 turns**. Lower is better for the first two; higher repair means a larger fraction of observed damage was later restored by the affected team. A repair dash means no qualifying damage occurred.

| Participant structure | Single V1 | Single V2 | Single V3 | Competition V1 | Competition V2 | Competition V3 |
|---|---|---|---|---|---|---|
| 2 Gemini agents | 0.000 / 0.000 / — / 4.336 | 0.000 / 0.000 / — / 6.448 | 0.000 / 0.001 / 0.000 / 6.213 | — | — | — |
| 2 Qwen agents | 0.002 / 0.009 / 0.375 / 0.000 | 0.002 / 0.001 / 0.000 / 3.626 | 0.002 / 0.007 / 0.333 / 5.167 | — | — | — |
| 2 Luna agents | 0.004 / 0.020 / 0.263 / 0.000 | 0.002 / 0.009 / 0.429 / 0.000 | 0.001 / 0.001 / 0.000 / 7.670 | — | — | — |
| 4 Gemini agents | 0.000 / 0.000 / — / 1.818 | 0.000 / 0.011 / 1.000 / 3.163 | 0.000 / 0.002 / 1.000 / 5.371 | 0.010 / 0.024 / 0.963 / 1.778 | 0.009 / 0.028 / 0.986 / 1.524 | 0.003 / 0.014 / 0.962 / 3.142 |
| 4 GLM agents | 0.031 / 0.050 / 0.622 / 0.976 | 0.018 / 0.036 / 0.444 / 2.439 | 0.014 / 0.015 / 0.727 / 3.586 | — | — | — |
| 4 Qwen agents | 0.014 / 0.015 / 0.556 / 0.000 | 0.019 / 0.037 / 0.708 / 3.902 | 0.006 / 0.009 / 0.200 / 3.129 | — | — | — |
| 4 Luna agents | 0.013 / 0.033 / 0.450 / 0.000 | 0.026 / 0.032 / 0.526 / 0.488 | 0.009 / 0.023 / 0.200 / 11.184 | — | — | — |
| 4 mixed agents (GLM, Gemini, Qwen, Luna) | 0.026 / 0.041 / 0.746 / 0.244 | 0.030 / 0.044 / 0.706 / 1.707 | 0.016 / 0.023 / 0.727 / 4.937 | 0.061 / 0.129 / 0.819 / 1.667 | 0.048 / 0.087 / 0.802 / 2.623 | 0.036 / 0.047 / 0.759 / 7.562 |
| 4 homogeneous agents; one model family per team | — | — | — | 0.055 / 0.146 / 0.803 / 0.386 | 0.092 / 0.220 / 0.627 / 1.929 | 0.037 / 0.038 / 0.669 / 7.099 |

### Efficiency and reliability

Each cell is **mean generation cost / cross-repetition exact-color IoU / completed runs over planned runs**. Higher IoU means more repeatable outputs.

| Participant structure | Single V1 | Single V2 | Single V3 | Competition V1 | Competition V2 | Competition V3 |
|---|---|---|---|---|---|---|
| 2 Gemini agents | $0.047 / 0.982 / 10/10 | $0.040 / 1.000 / 10/10 | $0.040 / 0.963 / 10/10 | — | — | — |
| 2 Qwen agents | $0.156 / 0.750 / 10/10 | $0.031 / 0.571 / 10/10 | $0.033 / 0.658 / 10/10 | — | — | — |
| 2 Luna agents | $0.015 / 0.655 / 10/10 | $0.015 / 0.722 / 10/10 | $0.013 / 0.631 / 10/10 | — | — | — |
| 4 Gemini agents | $0.099 / 0.918 / 5/5 | $0.079 / 1.000 / 5/5 | $0.062 / 1.000 / 5/5 | $1.389 / 0.954 / 5/5 | $1.470 / 0.924 / 5/5 | $1.010 / 0.951 / 5/5 |
| 4 GLM agents | $0.024 / 0.625 / 5/5 | $0.028 / 0.603 / 5/5 | $0.025 / 0.653 / 5/5 | — | — | — |
| 4 Qwen agents | $0.058 / 0.630 / 5/5 | $0.067 / 0.617 / 5/5 | $0.063 / 0.665 / 5/5 | — | — | — |
| 4 Luna agents | $0.031 / 0.650 / 5/5 | $0.030 / 0.665 / 5/5 | $0.023 / 0.784 / 5/5 | — | — | — |
| 4 mixed agents (GLM, Gemini, Qwen, Luna) | $0.136 / 0.761 / 10/10 | $0.063 / 0.780 / 10/10 | $0.058 / 0.779 / 10/10 | $0.508 / 0.403 / 5/5 | $0.522 / 0.462 / 4/5 | $0.661 / 0.517 / 4/5 |
| 4 homogeneous agents; one model family per team | — | — | — | $0.530 / 0.411 / 4/5 | $0.566 / 0.267 / 4/5 | $0.710 / 0.423 / 4/5 |

## Prompt intervention estimates

These are participant-structure-balanced differences from V1, with 5,000-resample run-level bootstrap intervals. Positive fidelity and quality are favorable; negative destructive overwrite is favorable.

| Task | Contrast | Outcome | Effect | 95% interval | Structures |
|---|---|---|---|---|---|
| Four-team shared-canvas competition | V2-V1 | Forum posts/run | 2.500 | [1.050, 3.917] | 3 |
| Four-team shared-canvas competition | V2-V1 | Reference fidelity | -0.007 | [-0.063, 0.049] | 3 |
| Four-team shared-canvas competition | V2-V1 | VLM quality | 0.002 | [-0.069, 0.066] | 3 |
| Four-team shared-canvas competition | V2-V1 | Destructive overwrite | 0.014 | [-0.042, 0.062] | 3 |
| Four-team shared-canvas competition | V3-V1 | Forum posts/run | 13.817 | [9.383, 18.917] | 3 |
| Four-team shared-canvas competition | V3-V1 | Reference fidelity | 0.039 | [-0.018, 0.092] | 3 |
| Four-team shared-canvas competition | V3-V1 | VLM quality | 0.095 | [-0.001, 0.188] | 3 |
| Four-team shared-canvas competition | V3-V1 | Destructive overwrite | -0.060 | [-0.111, -0.023] | 3 |
| Single-team shared image | V2-V1 | Forum posts/run | 0.550 | [0.400, 0.713] | 8 |
| Single-team shared image | V2-V1 | Reference fidelity | -0.012 | [-0.030, 0.006] | 8 |
| Single-team shared image | V2-V1 | VLM quality | -0.042 | [-0.108, 0.019] | 8 |
| Single-team shared image | V2-V1 | Destructive overwrite | 0.001 | [-0.007, 0.009] | 8 |
| Single-team shared image | V3-V1 | Forum posts/run | 1.387 | [1.037, 1.762] | 8 |
| Single-team shared image | V3-V1 | Reference fidelity | 0.003 | [-0.018, 0.024] | 8 |
| Single-team shared image | V3-V1 | VLM quality | 0.042 | [-0.023, 0.106] | 8 |
| Single-team shared image | V3-V1 | Destructive overwrite | -0.012 | [-0.017, -0.006] | 8 |

## Communication timing

Start means R0–R1; mid-run means R2 onward. V3 is the only prompt that produced substantial mid-run communication in the competition task.

| Task | Prompt | Team start | Team mid | Public start | Public mid | Posts/100 turns | Runs using forum |
|---|---|---|---|---|---|---|---|
| Four-team shared-canvas competition | V1 | 14 | 17 | 26 | 2 | 1.330 | 1.000 |
| Four-team shared-canvas competition | V2 | 17 | 28 | 33 | 5 | 1.993 | 1.000 |
| Four-team shared-canvas competition | V3 | 18 | 94 | 59 | 49 | 6.077 | 1.000 |
| Single-team shared image | V1 | 11 | 2 | 0 | 0 | 0.734 | 0.217 |
| Single-team shared image | V2 | 40 | 3 | 0 | 0 | 2.491 | 0.650 |
| Single-team shared image | V3 | 65 | 26 | 0 | 0 | 5.659 | 1.000 |

## Progress and stopping calibration

Completion and stall gaps are `1 - final fidelity`; smaller is better. Maximum-round stops are censored rather than treated as completion claims.

| Task | Prompt | Completed | Completion gap | Stalled | Stall gap | Max-round | Post-horizon gain |
|---|---|---|---|---|---|---|---|
| Four-team shared-canvas competition | V1 | 0 | — | 4 | 0.010 | 10 | 0.0000 |
| Four-team shared-canvas competition | V2 | 0 | — | 1 | 0.010 | 12 | 0.0000 |
| Four-team shared-canvas competition | V3 | 0 | — | 4 | 0.032 | 9 | 0.0000 |
| Single-team shared image | V1 | 11 | 0.000 | 6 | 0.124 | 43 | 0.0000 |
| Single-team shared image | V2 | 19 | 0.025 | 3 | 0.280 | 38 | 0.0000 |
| Single-team shared image | V3 | 25 | 0.053 | 19 | 0.151 | 16 | 0.0000 |

The extra V3 single-team rounds beyond the common 10-round horizon added only **0.0000** mean fidelity. The matched horizon therefore captures essentially all realized output quality.

## Task-level descriptive aggregates

These values give each participant structure equal weight. They are descriptive summaries; the prompt-intervention table above gives the uncertainty estimates.

| Task | Prompt | Structures | Fidelity | VLM quality | Progress AUC | Destructive rate | Posts/100 | Cost/run | Repeatability |
|---|---|---|---|---|---|---|---|---|---|
| Four-team shared-canvas competition | V1 | 3 | 0.761 | 0.722 | 0.675 | 0.094 | 1.277 | $0.809 | 0.589 |
| Four-team shared-canvas competition | V2 | 3 | 0.753 | 0.724 | 0.659 | 0.107 | 2.026 | $0.853 | 0.551 |
| Four-team shared-canvas competition | V3 | 3 | 0.799 | 0.818 | 0.711 | 0.033 | 5.934 | $0.794 | 0.630 |
| Single-team shared image | V1 | 8 | 0.887 | 0.738 | 0.691 | 0.021 | 0.922 | $0.071 | 0.747 |
| Single-team shared image | V2 | 8 | 0.875 | 0.695 | 0.688 | 0.022 | 2.722 | $0.044 | 0.745 |
| Single-team shared image | V3 | 8 | 0.890 | 0.780 | 0.703 | 0.009 | 5.907 | $0.040 | 0.767 |

## Data and validation

- Planned experimental slots: **225**; completed and behaviorally scored: **220**; provider-failed: **5**.
- Single-team completed runs: **180**; competing-team completed matches: **40**.
- All **220** completed histories replayed exactly to their saved canvases. The five technical failures were 529 provider-overload failures and remain in completion and cost accounting.
- Experimental generation cost, including failed work and billed invalid attempts: **$46.3683**. Billed invalid-output responses: **625**.
- Primary comparisons use matched horizons: 10 ordinary rounds for single-team and 20 for competing-team runs.
- “Observed repair” requires the affected team to see damage and later restore the reference-consistent pixel. Simultaneous reversals and opponent restorations do not count.

## Experimental cost by model

| Model | Billed cost | Share |
|---|---|---|
| google/gemini-3.8-flash | $31.8128 | 68.6% |
| qwen/qwen3.8-27b | $8.8326 | 19.0% |
| openai/gpt-5.6-luna | $3.3848 | 7.3% |
| z-ai/glm-5.3-flash | $2.3381 | 5.0% |

Prompt-version cost differences should not be read causally because versions ran at different times and providers showed different invalid-output behavior.

## Judge reliability

- Grok repeat sample: **n=45**, visual-quality rank correlation **0.931**, mean absolute difference **0.056** on the normalized 0–1 scale.
- Kimi independent audit: **n=45**, cross-model rank correlation **0.754**, mean absolute difference **0.102**.
- The separate 12-canvas competing calibration passed: rank correlation **0.842**, component MAD **0.146** points on the 0–4 scales.

## Limits requiring care

- Fourteen competing matches triggered an automatic placement ambiguity or overlap flag. Visual inspection confirmed genuinely crowded or overlapping drawings rather than obvious object-assignment errors; no score was manually altered.
- Pixel logs establish observed, team-attributable repair but cannot establish private intent.
- Prompt versions were executed at different times. The matched comparisons are informative for this experiment, while provider-time drift remains a nuisance variable.
- Participant-count comparisons are descriptive because the two-agent and four-agent batches were not randomized together.
