# Preregistration — NP/Z garden-path dissociation (Phase 1)

*Frozen 2026-06-22, before any test-model surprisal is computed.*

1. **Materials.** SAP Benchmark NP/Z items (Huang et al., 2024), repo commit
   `15e61066d510`, file sha256 `967a0e79ac6337bd`.
   24 lexicalizations. Item 14's control spillover normalized (stray 'political'
   removed). Distance manipulation: 0/1/2 plausibility-screened prenominal
   adjectives inserted into the temporarily-ambiguous NP; full modifier table and
   gpt2-large plausibility-drift report released with the stimuli.

2. **Conditions.** 2 (ambiguity: ambiguous, control) × 3 (distance: d1, d2, d3),
   within item; 6 cells × 24 items = 144 sentences.

3. **Region of interest.** Disambiguating matrix verb + 2 spillover words;
   spillover-only region analyzed separately.

4. **Models.** Llama-3 8B base (pinned revision), secondary fast contrast.
   Probabilities read, not sampled.

5. **Confirmatory test.** ambiguity × distance interaction on critical-region
   surprisal in the mixed model specified in analysis_plan.md. Random-effects
   structure: by-item intercepts; slopes only if convergent.

6. **Predictions.** H1 → positive ambiguity×distance (GP grows with distance);
   H2 → null interaction (GP flat). Direction validated against published human
   norms; no new human data collected.

7. **Exclusions.** Any item flagged by the plausibility gate is reported and, if
   retained, sensitivity-checked with the item removed.
