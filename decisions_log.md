
## 2026-06-22 — Day 1: NP/Z stimuli

- Adopted SAP Benchmark NP/Z (Huang et al. 2024) as the base set (commit
  `15e61066d510`); adapted, not authored.
- Distance manipulation = prenominal adjectives only (0/1/2) on the
  temporarily-ambiguous NP; postnominal PP/RC modifiers rejected (would inject a
  second attachment ambiguity).
- Plausibility "norming" = gpt2-large drift screen (held-out scorer), not human
  norming; flagged items reviewed. Primary materials caveat for the writeup.
- Item 14 control spillover normalized (removed stray "political").
- NP/S replication set deferred to Day 2.
- Confirmatory test = ambiguity × distance interaction (see analysis_plan.md).

## 2026-07-08 — Day 2: surprisal pipeline

- Model: gpt2-large, loaded via minicons IncrementalLMScorer, device=cpu.
- Token alignment: critical_word located via whole-word regex match against
  sentence, mapped to subword token indices via HF fast-tokenizer offset
  mapping; surprisal summed across critical-word tokens.
- Hand-validated on 5 items (see check_alignment.py output): all single-token
  matches, no splits, no misses.
- Full run: 144/144 rows scored, 0 skipped (critical word not found).
- Output: phase1_stimuli/day2_surprisal.csv

## 2026-07-08 — Day 2: surprisal pipeline

- Model: gpt2-large, loaded via minicons IncrementalLMScorer, device=cpu.
- Token alignment: critical_word located via whole-word regex match against
  sentence, mapped to subword token indices via HF fast-tokenizer offset
  mapping; surprisal summed across critical-word tokens.
- Hand-validated on 5 items: all single-token matches, no splits, no misses.
- Full run: 144/144 rows scored, 0 skipped.
- Descriptive means: GP effect (ambiguous - control) = 6.51/5.93/5.90 bits at
  d1/d2/d3. Effect present and large at every distance; direction of the
  distance trend is a mild *decrease*, not the increase H1 predicts. Purely
  descriptive at this stage — no significance test yet (Day 4). Flagging so
  it isn't lost before the mixed model is fit.
- Output: phase1_stimuli/day2_surprisal.csv

## 2026-07-08 — Day 3: multi-model run

- Models: gpt2-large (from Day 2, re-output for comparison), EleutherAI/pythia-1.4b.
- Pythia-1.4b download + scoring: ~3m total (116s load, 100s score on CPU).
- Output: phase1_stimuli/day3_surprisal_gpt2-large.csv and
  phase1_stimuli/day3_surprisal_EleutherAI_pythia-1.4b.csv
- Comparison: both models show large GP effect (5.87–6.51 bits at d1, 5.25–5.93 at
  d2, 5.48–5.90 at d3). Distance trend is consistent across models: effect
  shrinks from d1 to d2, stays flat or slightly rebounds at d3. Neither matches
  H1 prediction of scaled effect. Pythia control condition noisier across distance
  than gpt2-large.
- Note on Llama-3 8B: gated access still pending, 16GB RAM insufficient for fp32,
  deferred to Colab GPU (can run later this week in parallel with Day 4/5).

## 2026-07-08 — Day 4: mixed-effects model and headline result

- Model: surprisal ~ C(ambiguity) * distance_numeric + (1 | item)
- Data: 288 observations (144 sentences × 2 models), 24 items
- Converged: Yes
- **Key result: ambiguity × distance interaction is NOT significant (p=0.315)**
  The garden-path effect is stable across distances (d1: 6.19 bits avg,
  d2: 5.59 bits, d3: 5.70 bits); distance does NOT modulate the effect size.
- Main effects: ambiguity (p<0.001), distance (p=0.008), but no interaction.
- Headline: GP effect is large and robust, not distance-dependent. Consistent
  with H2/H3 (structure-driven, not linear-distance-driven).
- Figure: phase1_stimuli/day4_headline_figure.png (GP effect vs. distance, per model)

## 2026-07-08 — Day 5: robustness checks

- Hold-out replication on 50% of items (n=12 held out): 
  Train set: ambiguity effect −6.05 (p<0.001), interaction p=0.323
  Test set: ambiguity effect −6.59 (p<0.001), interaction p=0.716
- Effect is stable across train/test split (difference 0.54 bits).
- Interaction remains non-significant in all splits (p=0.32–0.72).
- Direction check: both models show ambiguous > control (16.74/10.77 and 16.88/11.21).
- Conclusion: effect is robust, replicable, and consistent across held-out items.
  Ready for writeup.

## 2026-07-11 — Housekeeping: file renames

Folder renamed sapbenchmark → price-of-reanalysis (matching the paper title).
Project scripts and outputs renamed from day-based to descriptive names; all
internal path references updated. External SAP Benchmark files untouched.
Mapping (old → new):

- day1_stimuli.py → build_stimuli.py
- day2_pipeline.py → score_surprisal.py
- day3_multimodel.py → score_surprisal_multimodel.py
- day4_model.py → fit_mixed_model.py
- day5_robustness.py → robustness_checks.py
- day6_reanalysis.py → corrected_statistics.py
- phase1_stimuli/day2_surprisal.csv → phase1_stimuli/surprisal_gpt2-large_initial.csv
- phase1_stimuli/day3_surprisal_gpt2-large.csv → phase1_stimuli/surprisal_gpt2-large.csv
- phase1_stimuli/day3_surprisal_EleutherAI_pythia-1.4b.csv → phase1_stimuli/surprisal_EleutherAI_pythia-1.4b.csv
- phase1_stimuli/day4_headline_figure.png → phase1_stimuli/garden_path_by_distance.png

Filenames in the dated entries above are historical; use this mapping to
translate.
