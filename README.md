# The Price of Reanalysis

Does the cost of syntactic reanalysis in neural language models grow with the
distance over which a misanalysis is maintained?

This project tests whether the garden-path (NP/Z) surprisal effect in neural
LMs scales with the length of the temporarily ambiguous noun phrase. We
manipulate distance by adding 0, 1, or 2 prenominal adjectives to the
ambiguous NP (e.g., *"While the man hunted the (brown) (majestic) deer ran
into the woods"*) and measure surprisal at the disambiguating verb.

**Headline result:** the garden-path effect is large and robust (~6 bits at
the disambiguating region in both models tested), but it is **not modulated by
distance**. The ambiguity × distance interaction is non-significant in
per-model and pooled mixed-effects fits, and stable across held-out item
splits — consistent with a structure-driven account of reanalysis cost rather
than a linear-distance one.

Stimuli are adapted from the NP/Z subset of the **SAP Benchmark**
([Huang et al. 2024](https://github.com/caplabnyu/sapbenchmark)); see
[Attribution](#attribution) below.

## Repository layout

Project files live at the repo root and in `phase1_stimuli/`. Three files are
carried over from the original SAP Benchmark repo: `Items for all
subsets.xlsx` (the item source `build_stimuli.py` reads and checksums),
`readme_SAP.txt` (the upstream documentation), and `LICENSE` (which applies to
those upstream materials). The rest of the benchmark — the human reading-time
data and analyses — is not included here; get it from the
[upstream repo](https://github.com/caplabnyu/sapbenchmark) at the commit
pinned in `phase1_stimuli/provenance.json`.

### Pipeline (in run order)

| Script | What it does |
|---|---|
| `build_stimuli.py` | Builds the NP/Z stimulus set from the SAP Benchmark items: validates the adjective insertion slot, applies the distance manipulation (d1/d2/d3), runs an optional held-out-LM plausibility screen, and writes `phase1_stimuli/npz_stimuli.csv` plus provenance and design docs. |
| `check_alignment.py` | Hand-validation utility: prints token-by-token offsets for the first items so the critical-word-to-subword alignment can be eyeballed. |
| `score_surprisal.py` | Initial single-model surprisal run (GPT-2-large via `minicons`); sums surprisal over the critical word's subword tokens using fast-tokenizer offset mapping. → `phase1_stimuli/surprisal_gpt2-large_initial.csv` |
| `score_surprisal_multimodel.py` | Production scoring run over GPT-2-large and Pythia-1.4b. → `phase1_stimuli/surprisal_<model>.csv` |
| `fit_mixed_model.py` | First-pass mixed-effects model (`surprisal ~ ambiguity × distance + (1 \| item)`) and the headline figure. → `phase1_stimuli/garden_path_by_distance.png` |
| `robustness_checks.py` | Hold-out replication on 50% of items, effect-direction checks, and full-sample reference fit. |
| `corrected_statistics.py` | Final statistics for the paper: per-model fits as primary (avoiding pseudo-replication from pooling models), pooled fit with a model term as secondary, random-slope attempts with documented fallback, 95% CIs checked against a SESOI (±0.3 bits/step) for the equivalence claim, and a split-half check. |

`decisions_log.md` is a dated log of every design and analysis decision,
including the file-rename mapping from the original day-numbered scripts.
`phase1_stimuli/analysis_plan.md` and `phase1_stimuli/preregistration.md`
record the locked design.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run scripts from the repo root, e.g.:

```bash
python build_stimuli.py
python score_surprisal_multimodel.py
python corrected_statistics.py
```

Everything runs on CPU; the multi-model scoring run takes a few minutes.

## Design summary

- **Construction:** NP/Z garden paths (24 items × 2 conditions × 3 distances = 144 sentences)
- **Manipulation:** ambiguity (ambiguous vs. comma-disambiguated control) × distance (0/1/2 prenominal adjectives on the ambiguous NP)
- **Measure:** summed surprisal (bits, base 2) over the critical (disambiguating) word
- **Models:** GPT-2-large, Pythia-1.4b
- **Plausibility screen:** held-out-LM drift check (deliberately not a test model, to avoid circularity); a screening proxy, not human norming

## Attribution

Stimuli and supporting materials are adapted from the SAP Benchmark:

> Huang, K.J., Arehalli, S., Kugemoto, M., Muxica, C., Prasad, G., Linzen, T.,
> & Dillon, B. (2024). A large-scale investigation of syntactic processing
> reveals misalignments between humans and neural language models.

The exact upstream commit and source-file checksum are recorded in
`phase1_stimuli/provenance.json`. The original repo's documentation is
preserved in `readme_SAP.txt`, and the inherited `LICENSE` file applies to the
upstream materials.
