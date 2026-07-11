# Phase-1 analysis plan — NP/Z garden-path dissociation

*Locked 2026-06-22. Source: SAP Benchmark (Huang et al., 2024, JML), repo
commit `15e61066d510`, file `Items for all subsets.xlsx` (sha256
`967a0e79ac6337bd`). Constructions: NP/Z (primary).*

## Design
Ambiguity (ambiguous vs. control) × linear distance (d1/d2/d3), within item.
Distance lengthens the temporarily-ambiguous NP with 0/1/2 prenominal adjectives,
holding the disambiguating matrix verb and the NP's structural role (matrix
subject) fixed. Both ambiguity levels carry the same modifier, so the modifier's
own surprisal cancels in the garden-path contrast.

## Dependent measure
Token-level surprisal (base-2) from the LM, summed over the critical region:
the disambiguating word + 2 spillover words (Huang et al.'s ROI). Critical word
located by string, aligned to its subword span with minicons. A spillover-only
region is scored separately.

GP effect = surprisal(ambiguous) − surprisal(control) at the critical region.

## Model — the hypothesis test
Linear mixed-effects model (pymer4 / lme4) of critical-region surprisal:

    surprisal ~ ambiguity * distance + (1 | item)

Fixed effects: ambiguity, distance (treated as ordered/numeric 1–3), and the
**ambiguity × distance interaction — the confirmatory test**. Random intercepts
for item; by-item random slopes for ambiguity and/or distance added only if the
model converges (no maximal-model fights). Distance coded both as a factor and as
a linear contrast; the linear ambiguity×distance term is the primary estimate.

- H1 (linear): GP effect grows with distance → ambiguity×distance ≠ 0, GP rising.
- H2 (hierarchical): GP effect flat across distance → ambiguity×distance ≈ 0.

## Models scored
Llama-3 8B (base, pinned revision) primary; a fast contrast (Pythia-1.4B or
GPT-2-large) secondary. Read probabilities; do not sample. Scoring model ≠ the
gpt2-large plausibility scorer used for stimulus screening.

## Robustness (Day 5)
Replicate on a second random item sample; re-align tokens and re-check; confirm
GP-effect direction matches published human reading-time norms (van Schijndel &
Linzen, 2021; Arehalli et al., 2022).

## Out of scope (Phase 2)
Structural probes, attention analysis, activation patching, Pythia/OLMo
checkpoint developmental study, and the agreement structural-attraction contrast.
