# Seven-metric analysis protocol

## Scope and common rules

- The statistical unit is one completed run (single-team) or one completed match (competing teams). Pixels, calls, agents, and repeated judge ratings are not independent samples.
- Failed runs are excluded from behavioral scores and included in the technical failure rate. A run with a valid, genuinely blank final canvas is scored normally.
- Report every run as a point, condition median and IQR, condition mean, and a run-level bootstrap 95% interval (5,000 resamples). Emphasize effect sizes and intervals rather than binary significance.
- Compare single-team and competing-team experiments separately. They are different tasks.
- The main prompt comparison uses a fixed horizon: 10 ordinary rounds for single-team runs and 20 for competing-team matches. Round 0 is retained as the initialization turn. Later single-team V3 rounds are an extension analysis.
- Before scoring, replay each run and assert: valid dimensions and palette; accepted plus rejected action counts reconcile; `pixels_changed == len(changed_actions)`; completion orders are unique within a concurrent cohort; the replayed canvas equals `final_canvas.json`; and successful-response usage reconciles with `usage.json`.

## 1. Reference fidelity

**Question:** How closely does the final drawing match its target, independent of where it was placed?

### Single-team calculation

Read the canonical 16x16 reference directly from `harness/references.py`. Let `R_t` be its set of colored `(x, y, color)` triples after translation `t`, and let `C` be all colored triples on the final 32x32 canvas. Test all 17x17 = 289 valid translations.

For each translation:

`TP_t = |R_t intersection C|`

`F1_t = 2 TP_t / (|R_t| + |C|)`

The run's fidelity is `max_t F1_t`. A wrong color at a target cell counts as both a missing correct pixel and an extra wrong pixel. Stray pixels anywhere on the canvas reduce precision. Save the winning translation, precision, recall, and score.

### Competing-team calculation

Score the final visual result wherever each assigned object is best represented:

1. For each object and each valid translation, let `C_t` contain the final colored pixels inside that translated 16x16 window. Compute exact-color F1 between `C_t` and the translated reference.
2. Give the object its maximum F1 over all 289 positions. This measures the final outcome and does not require the object to remain where its team began drawing.
3. Macro-average the four object maxima to obtain match fidelity.

The four objects have different dominant colors, which makes one region unlikely to score highly as several objects. Still, flag a case when selected object boxes overlap by more than 25% of the smaller box, an object's best and best spatially distinct alternatives differ by less than 0.02 F1, or an object F1 is below 0.20. Audit 15 stratified matches plus all flagged cases using a contact sheet. If exact ties remain, use proximity to that team's edit centroid only as a tie-breaker. Retain visual-location versus team-centroid agreement as an attribution diagnostic, not part of fidelity.

**Primary outputs:** single-team F1; four per-object competing F1s; competing macro-F1.

## 2. Blinded visual quality

**Question:** Does the result look like coherent, recognizable pixel art to a strong vision model, including defects that exact pixel matching misses?

### Judges and blinding

- Primary judge: `x-ai/grok-4.6` on every valid output.
- Independent audit judge: `moonshotai/kimi-k3` on a stratified 20% sample.
- Same-model repeat: Grok 4.6 repeats that same 20% sample in a different order.
- Do not expose prompt version, model composition, condition name, cost, forum use, or deterministic metrics.
- Randomize candidate order. Return an array in presentation order; do not ask the model to copy candidate letter labels.
- Use temperature 0, low reasoning, strict JSON, and record the exact model, provider, request, response, usage, and cost.

### Single-team rubric

Send the apple reference once, followed by five candidate canvases. Score each candidate independently from 0 to 4 on:

1. **Recognizability:** how clearly it depicts the requested object.
2. **Reference fidelity:** subjective agreement in shape and colors; retained as a diagnostic because metric 1 is the primary fidelity measure.
3. **Visual coherence:** whether the object is connected, clean, and intentional rather than fragmented or malformed.

The visual-quality score is `(recognizability + visual coherence) / 8`, yielding 0 to 1. This avoids counting fidelity twice in the seven-metric suite.

### Competing-team rubric

Send one labeled sheet containing the four references, followed by three candidate shared canvases. Return, for every candidate:

- recognizability and subjective fidelity for each of the four assigned objects, both 0–4;
- whole-canvas separation/coherence, 0–4: are all four objects legible and spatially organized despite competition?

The match visual-quality score is `(mean object recognizability + separation/coherence) / 8`. Subjective object fidelity remains diagnostic.

### Acceptance checks and budget

Calibrate on 12 outputs spanning deterministic fidelity and visible failure modes. Accept the production protocol only if:

- valid-schema rate is at least 98%;
- same-model repeat Spearman correlation is at least 0.75;
- same-model mean absolute score difference is at most 0.5 on the 0–4 scales.

Cross-model disagreement is reported and is not silently averaged away. Grok remains the preregistered primary judge.

Set a hard judge-spend guard at $4.50 and reserve $0.50 for retries. The single-team pilot cost about $0.018 per five-image Grok batch and $0.027 per five-image Kimi batch, so the expected full analysis is roughly $1.5–$2.5 even with audits. The metric-design validation itself cost $0.0916.

**Primary output:** normalized visual-quality score. **Diagnostics:** rubric components, repeat agreement, cross-model agreement, invalid-output rate.

## 3. Conflict and repair

**Question:** When agents target the same cells or damage each other's correct work, how often do they recover?

Reconstruct concurrency cohorts using both the round number and a hash of the input canvas image. Agents in one cohort acted from the same observed state. Application order is recorded by `completion_order`.

### Simultaneous intent conflict

For each cohort and coordinate, collect all proposed final states, treating erase as blank.

- Two agents proposing the same color are redundant, not conflicting.
- A coordinate is a conflict if agents propose at least two different final states.

`intent conflict rate = conflicting unique coordinates / all unique coordinates targeted in concurrency cohorts`

Also report the conflict rate among shared-target coordinates and the count per 100 accepted pixel actions. The first is prevalence across all work; the other two distinguish rare collisions from frequent disagreement.

### Destructive overwrite and repair

Repair requires a team-specific target location, unlike outcome fidelity. Infer that location by replaying only the team's accepted actions on a blank intent canvas and finding the translation of its reference with the highest exact-color F1 against that intent. Flag low-confidence team alignments for review. Then track a damage episode for each `(affected team, coordinate)`:

1. The episode begins when an agent changes a pixel last made reference-correct by a different agent (or team) into a state incorrect for the affected target.
2. Repeated wrong edits while the pixel remains damaged do not create extra episodes.
3. Damage must remain after all actions in the current cohort. A same-cohort reversal occurred without observing the damage and is classified as an unobserved reversal, not repair.
4. An **observed team repair** occurs only when the affected team later receives an input canvas showing the damaged state and one of its members restores the reference-correct color. Restoration by an opponent does not count.

`destructive overwrite rate = persistent damage episodes / state-changing pixel edits`

`repair rate = observed team-repaired persistent episodes / persistent damage episodes`

`repair latency = number of observable rounds from damage to repair`

Report intra-team and cross-team episodes separately. Erasing a correct pixel is destructive; erasing an incorrect intrusion can repair. Rejected or out-of-bounds actions are validity failures, not conflicts.

Pixel data cannot establish private intent. Therefore, do not label an observed team repair as “deliberate.” It establishes three behavioral facts: the team was responsible for the restoration, it had been shown the damage, and it restored the target-consistent value. Report same-cohort reversals and opponent restorations separately as accidental/unattributed recovery diagnostics.

**Primary outputs:** intent conflict rate, destructive overwrite rate, repair rate, median repair latency.

## 4. Communication effectiveness

**Question:** Does stronger communication prompting produce useful communication and better coordination?

Use prompt version as the intervention. Do not claim that an individual message caused a later pixel change merely because it came first.

### Uptake

From forum events, calculate:

- fraction of runs/teams with at least one post;
- posts per 100 agent turns;
- start communication: posts in R0–R1;
- mid-run communication: posts in R2 and later;
- team-forum and public-forum rates separately.

### Outcome effect

Within each team-composition block shared by V1, V2, and V3, calculate version differences in:

- final reference fidelity;
- destructive overwrite rate;
- visual quality;
- generation cost.

For each outcome, give every composition block equal weight:

`effect(Vk) = mean over blocks [ mean(outcome | Vk, block) - mean(outcome | V1, block) ]`

Bootstrap complete runs within each version-by-block cell. Analyze single-team and competing-team experiments separately. If a prompt increases posting without improving fidelity, quality, or conflict, describe it as increased communication without demonstrated benefit.

**Primary outputs:** communication uptake, V2−V1 and V3−V1 fidelity effects, and corresponding conflict effects.

## 5. Progress and stopping calibration

**Question:** How quickly does the canvas improve, and do agents stop when the drawing is actually finished?

Reconstruct one checkpoint after Round 0 and one after every complete ordinary round. Do not use each member's post-application snapshot as a separate time step. Let `q_r` be reference fidelity at the checkpoint after round `r`.

For fixed horizon `H` (10 single-team, 20 competing):

`progress AUC = mean(q_0, q_1, ..., q_H)`

If a valid run stops early, carry its final canvas forward through `H`; this rewards finishing early without treating absent calls as missing data. Failed runs are not carried forward.

Stopping is evaluated by reason:

- all agents returned empty: `completion gap = 1 - fidelity_at_stop`;
- harness stall rule: `stall gap = 1 - fidelity_at_stop`;
- maximum rounds reached: right-censored, because the agents did not claim completion.

Show the median trajectory by condition alongside AUC. Keep the final two-round gain as a diagnostic for whether a stall detector fired on a plateau; do not create an arbitrary binary “premature stop” threshold.

**Primary outputs:** progress AUC, final fidelity, stop-reason proportions, completion/stall gap.

## 6. Cost efficiency

**Question:** What did each behavior and quality level cost?

Sum `usage.cost` from every actual OpenRouter response JSON, including invalid-output responses saved before retries. Deduplicate by OpenRouter response ID. Transport errors and rate-limit responses with no model response cost zero. If a response lacks cost, flag it rather than silently treating it as zero. Cross-check the successful-response subtotal against `usage.json`.

For every run report:

- total generation USD;
- input, output, and reasoning tokens;
- calls and retry counts;
- dollars per 0.1 reference-fidelity achieved: `0.1 * cost / fidelity` (infinite if fidelity is zero).

Use raw cost as the primary measure and the ratio only as a quality-adjusted diagnostic. Compare ratios within matched task horizons and model-composition blocks. Report judge expenditure separately from experimental generation expenditure.

**Primary output:** USD per run/match. **Diagnostics:** token composition, retries, dollars per 0.1 fidelity.

## 7. Repetition diversity and reliability

**Question:** Does the same configuration reliably reach a similar quality and visual solution across repetitions?

### Outcome reliability

For every exact condition, report the median and IQR of final fidelity and visual quality, plus the technical completion rate. Do not hide failures or replace them with zero-quality canvases.

### Visual repeatability

Canonicalize placement before comparing images:

- Single-team: use the winning translation from metric 1 and extract the corresponding 16x16 candidate window.
- Competing-team: extract each object's 16x16 window using that team's inferred target translation.

Represent a crop as its set of `(local x, local y, color)` triples. For every pair of repetitions in the same condition:

`color IoU = |A intersection B| / |A union B|`

If both are blank, define IoU as 1; if only one is blank, define it as 0. The condition's repeatability is the mean pairwise color IoU. Diversity is `1 - repeatability`. In competing matches, compute IoU separately for each object and macro-average the four object values.

Pairwise comparisons are not treated as independent observations. Bootstrap whole runs and recompute the condition statistic. Retain centroid-position spread as a placement diagnostic, separate from visual repeatability.

**Primary outputs:** fidelity IQR, mean canonical color-IoU repeatability, and technical failure rate.

## Validation already performed

Two independent log-schema and judge-design audits checked the protocol against the actual artifacts. A five-canvas judge pilot deliberately spanned deterministic fidelity from approximately 0.58 to 1.00.

- Grok 4.6 produced valid JSON twice and, after image-order reversal, reproduced 14 of 15 component ratings exactly. The only difference was one point, giving component MAD 0.067 and composite rank correlation 0.973.
- The first shuffled Kimi K3 request revealed that letter IDs could be reassigned to presentation positions. The corrected order-based schema removed that failure. Kimi preserved the same broad quality ordering but was somewhat more generous on recognizability.
- Four paid validation calls cost $0.091591335 in total; failed HTTP validation requests were not billed model responses.

## Remaining uncertainty

1. **Competing-team localization:** final-outcome best-fit scoring follows your chosen definition, but heavily mixed or overlapping objects can admit several plausible placements. The overlap/tie flags and manual audit are required. Repair uses a separate team-intent alignment because attribution cannot be recovered from appearance alone.
2. **Competing-canvas VLM rubric:** the single-team judge schema is validated; the four-object rubric still needs its 12-image calibration before production scoring.
3. **Communication causality:** prompt version is a real intervention, but batches ran at different times and provider conditions may differ. Report matched effect sizes as experimental evidence with this limitation, rather than claiming that particular messages caused particular repairs.

No Luna preprocessing is needed for the metrics. The two Luna subagents were used only to audit the schema and judge protocol; all production calculations should remain deterministic Python except metric 2.
