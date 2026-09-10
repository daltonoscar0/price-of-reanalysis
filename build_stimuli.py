#!/usr/bin/env python3
"""
build_stimuli.py  --  Phase-1 NP/Z stimulus build, end to end.

Run from the project root (the cloned SAP Benchmark repo). Does everything after
"locate the items":

  0. provenance   : record repo commit + source-file checksum   -> phase1_stimuli/provenance.json
  1. validate     : confirm the det+N insertion slot on every item
  2. item 14 fix  : trim the stray 'political' from its control spillover
  3. plausibility : held-out-LM screen for plausibility drift d1->d3 (optional)
  4. generate     : the ambiguity x distance long CSV            -> phase1_stimuli/npz_stimuli.csv
  5. (--also-nps) : same for the NP/S replication set           -> phase1_stimuli/nps_stimuli.csv
  6. docs         : analysis_plan.md + preregistration.md prefilled with the locked design
  7. log          : append a dated entry to decisions_log.md

The plausibility step (3) loads a SCORER MODEL (default gpt2-large) and is a
*screening proxy*, not human norming: it flags items whose modified NP becomes a
markedly worse object of the subordinate verb, or worse subject of the matrix verb,
as adjectives are added. It is deliberately NOT Llama-3 (validating stimuli on the
test model is circular). Skip with --skip-plausibility; no model is loaded then.

No test-model surprisal is computed here. Tested with python 3.12, pandas 3.0.2,
openpyxl 3.1.5; the gate additionally needs torch + transformers.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
SHEET = "NPSNPZMVRR"
DETERMINERS = {
    "the", "a", "an", "his", "her", "its", "their",
    "our", "my", "your", "this", "that", "these", "those",
}

# Prenominal-adjective modifier table, keyed by item number, REUSED for NP/Z and
# NP/S (both constructions share head nouns in the same item order). d2 = one
# adjective, d3 = two (natural English order). Chosen to stay plausible as both
# object of the subordinate verb and subject of the matrix verb; the plausibility
# gate validates them and flags any that drift. Edit freely, then re-run.
MODIFIERS = {
    1:  {"d2": ["confidential"], "d3": ["confidential", "legal"]},   # file
    2:  {"d2": ["new"],          "d3": ["new", "federal"]},          # bill
    3:  {"d2": ["important"],    "d3": ["important", "confidential"]},# mail
    4:  {"d2": ["small"],        "d3": ["small", "white"]},          # chicken
    5:  {"d2": ["complex"],      "d3": ["complex", "surgical"]},     # operation
    6:  {"d2": ["large"],        "d3": ["large", "federal"]},        # grant
    7:  {"d2": ["local"],        "d3": ["local", "delivery"]},       # service
    8:  {"d2": ["old"],          "d3": ["old", "rusty"]},            # truck
    9:  {"d2": ["new"],          "d3": ["catchy", "new"]},           # song
    10: {"d2": ["large"],        "d3": ["large", "annual"]},         # bonus
    11: {"d2": ["new"],          "d3": ["demanding", "new"]},        # job
    12: {"d2": ["intensive"],    "d3": ["intensive", "mandatory"]},  # training
    13: {"d2": ["official"],     "d3": ["official", "legal"]},       # document
    14: {"d2": ["lucrative"],    "d3": ["lucrative", "new"]},        # contract
    15: {"d2": ["heavy"],        "d3": ["heavy", "specialized"]},    # equipment
    16: {"d2": ["young"],        "d3": ["young", "timid"]},          # lamb
    17: {"d2": ["difficult"],    "d3": ["difficult", "advanced"]},   # position
    18: {"d2": ["detailed"],     "d3": ["detailed", "new"]},         # contract
    19: {"d2": ["new"],          "d3": ["new", "experimental"]},     # treatment
    20: {"d2": ["large"],        "d3": ["large", "noisy"]},          # machine
    21: {"d2": ["new"],          "d3": ["elegant", "new"]},          # ballet
    22: {"d2": ["extra"],        "d3": ["extra", "prize"]},          # money
    23: {"d2": ["new"],          "d3": ["small", "new"]},            # restaurant
    24: {"d2": ["large"],        "d3": ["large", "industrial"]},     # oven
}

CONSTRUCTIONS = {"NPZ": "Transitive/Intransitive (NP/Z)",
                 "NPS": "Direct Object/Sentential Complement (NP/S)"}


# ----------------------------------------------------------------------------
# 0. provenance
# ----------------------------------------------------------------------------
def provenance(xlsx: Path, out_dir: Path):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "UNKNOWN-not-a-git-checkout"
    sha = hashlib.sha256(xlsx.read_bytes()).hexdigest()
    rec = {"date": str(date.today()), "repo_commit": commit,
           "source_file": xlsx.name, "source_sha256": sha,
           "modifier_table_items": sorted(MODIFIERS)}
    (out_dir / "provenance.json").write_text(json.dumps(rec, indent=2))
    print(f"[0] provenance: commit {commit[:12]}  xlsx sha256 {sha[:12]}")
    return rec


# ----------------------------------------------------------------------------
# load + validate
# ----------------------------------------------------------------------------
def load_items(xlsx: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=SHEET, engine="openpyxl")
    df.columns = ["item", "condition", "disambPositionUnamb", "control",
                  "disambPositionAmb", "ambiguous", "Question",
                  "Option1", "Option0", "Answer", "AmbiguityTargeted"]
    df = df[df["condition"].astype(str).str.startswith(prefix)].copy()
    df["disambPositionAmb"] = df["disambPositionAmb"].astype(int)
    df["disambPositionUnamb"] = df["disambPositionUnamb"].astype(int)
    return df[["item", "control", "disambPositionUnamb",
               "ambiguous", "disambPositionAmb"]].reset_index(drop=True)


def validate_pattern(df: pd.DataFrame, prefix: str):
    bad = []
    for _, r in df.iterrows():
        for sent, pos in ((r["ambiguous"], r["disambPositionAmb"]),
                          (r["control"], r["disambPositionUnamb"])):
            w = sent.split()
            det = w[pos - 3].lower().strip(",.")
            if det not in DETERMINERS:
                bad.append((int(r["item"]), w[pos - 3]))
    if bad:
        print(f"[1] {prefix}: {len(bad)} items FAIL det+N check -> hand-edit: {bad}")
    else:
        print(f"[1] {prefix}: all {len(df)} items pass the det+N insertion check")
    return bad


# ----------------------------------------------------------------------------
# 2. item-14 spillover fix
# ----------------------------------------------------------------------------
def fix_item14(df: pd.DataFrame):
    """Huang item 14's control carries a stray 'political' in the spillover that
    the ambiguous version lacks. Drop it so the ROI matches across the pair."""
    m = df["item"] == 14
    if not m.any():
        return df, False
    ctrl = df.loc[m, "control"].iloc[0]
    fixed = ctrl.replace("another political controversy", "another controversy")
    if fixed != ctrl:
        df.loc[m, "control"] = fixed
        print("[2] item 14: removed stray 'political' from control spillover")
        return df, True
    print("[2] item 14: nothing to fix (already matched)")
    return df, False


# ----------------------------------------------------------------------------
# core insertion
# ----------------------------------------------------------------------------
def insert(sentence: str, disamb_pos: int, mods):
    """Insert `mods` after the determiner of the ambiguous NP (immediately before
    the head noun, which sits immediately before the disambiguator at `disamb_pos`,
    1-indexed). Returns (sentence, new_disamb_pos_1idx, disambiguator)."""
    w = sentence.split()
    di = disamb_pos - 1
    disambiguator = w[di]
    det_idx = di - 2
    w = w[:det_idx + 1] + list(mods) + w[det_idx + 1:]
    new_di = di + len(mods)
    assert w[new_di] == disambiguator
    return " ".join(w), new_di + 1, disambiguator


def build_long(df: pd.DataFrame, construction: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        item = int(r["item"])
        if item not in MODIFIERS:
            continue
        spec = {"d1": [], "d2": MODIFIERS[item]["d2"], "d3": MODIFIERS[item]["d3"]}
        for dist, mods in spec.items():
            for cond, scol, pcol in (("ambiguous", "ambiguous", "disambPositionAmb"),
                                     ("control", "control", "disambPositionUnamb")):
                sent, idx, crit = insert(r[scol], int(r[pcol]), mods)
                rows.append({"item": item, "construction": construction,
                             "condition": cond, "distance": dist,
                             "critical_region_index": idx, "critical_word": crit,
                             "sentence": sent})
    return (pd.DataFrame(rows)
            .sort_values(["item", "condition", "distance"]).reset_index(drop=True))


# ----------------------------------------------------------------------------
# 3. plausibility gate (held-out LM, optional)
# ----------------------------------------------------------------------------
def _span_nll(model, tok, prefix: str, target: str):
    """Mean per-token NLL (nats) of `target` continued from `prefix`."""
    import torch
    pre = tok(prefix, return_tensors="pt")
    full = tok(prefix + " " + target, return_tensors="pt")
    n_pre = pre["input_ids"].shape[1]
    ids = full["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits
    logp = torch.log_softmax(logits[0, :-1], dim=-1)
    tgt = ids[0, 1:]
    tok_nll = -logp[range(tgt.shape[0]), tgt]
    span = tok_nll[n_pre - 1:]                      # tokens belonging to `target`
    return float(span.mean()) if span.numel() else float("nan")


def plausibility_gate(df: pd.DataFrame, prefix: str, scorer: str,
                      out_dir: Path, threshold: float):
    try:
        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[3] SKIP plausibility: torch/transformers not installed "
              "(`pip install torch transformers`)")
        return None
    print(f"[3] plausibility screen on {scorer} (first run downloads weights)...")
    tok = AutoTokenizer.from_pretrained(scorer)
    model = AutoModelForCausalLM.from_pretrained(scorer).eval()

    rep = []
    for _, r in df.iterrows():
        item = int(r["item"])
        if item not in MODIFIERS:
            continue
        rec = {"item": item}
        for dist in ("d1", "d2", "d3"):
            mods = [] if dist == "d1" else MODIFIERS[item][dist]
            # object role: P(head | subordinate-clause + det + mods)
            aw = r["ambiguous"].split(); adi = r["disambPositionAmb"] - 1
            adet = adi - 2
            amods = aw[:adet + 1] + list(mods) + aw[adet + 1:]
            head_i = adet + 1 + len(mods)
            rec[f"obj_{dist}"] = _span_nll(
                model, tok, " ".join(amods[:head_i]), amods[head_i])
            # subject role: P(matrix verb | unambiguous frame + modified subject)
            cw = r["control"].split(); cdi = r["disambPositionUnamb"] - 1
            cdet = cdi - 2
            cmods = cw[:cdet + 1] + list(mods) + cw[cdet + 1:]
            mv_i = cdi + len(mods)
            rec[f"subj_{dist}"] = _span_nll(
                model, tok, " ".join(cmods[:mv_i]), cmods[mv_i])
        rec["obj_drift"] = rec["obj_d3"] - rec["obj_d1"]
        rec["subj_drift"] = rec["subj_d3"] - rec["subj_d1"]
        rec["flag"] = (abs(rec["obj_drift"]) > threshold or
                       abs(rec["subj_drift"]) > threshold)
        rep.append(rec)

    rdf = pd.DataFrame(rep).round(3)
    rdf.to_csv(out_dir / f"plausibility_{prefix.lower()}.csv", index=False)
    flagged = rdf[rdf["flag"]]["item"].tolist()
    print(f"    drift threshold {threshold} nats; "
          f"{len(flagged)} item(s) flagged for review: {flagged or 'none'}")
    return rdf


# ----------------------------------------------------------------------------
# 6. docs
# ----------------------------------------------------------------------------
def emit_docs(out_dir: Path, prov: dict, also_nps: bool):
    sets = "NP/Z (primary)" + (" and NP/S (replication)" if also_nps else "")
    (out_dir / "analysis_plan.md").write_text(f"""# Phase-1 analysis plan — NP/Z garden-path dissociation

*Locked {prov['date']}. Source: SAP Benchmark (Huang et al., 2024, JML), repo
commit `{prov['repo_commit'][:12]}`, file `{prov['source_file']}` (sha256
`{prov['source_sha256'][:16]}`). Constructions: {sets}.*

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

## Model, the hypothesis test
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
""")

    (out_dir / "preregistration.md").write_text(f"""# Preregistration — NP/Z garden-path dissociation (Phase 1)

*Frozen {prov['date']}, before any test-model surprisal is computed.*

1. **Materials.** SAP Benchmark NP/Z items (Huang et al., 2024), repo commit
   `{prov['repo_commit'][:12]}`, file sha256 `{prov['source_sha256'][:16]}`.
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
""")
    print("[6] wrote analysis_plan.md + preregistration.md")


# ----------------------------------------------------------------------------
# 7. decisions log
# ----------------------------------------------------------------------------
def update_log(repo_root: Path, prov: dict, item14_fixed: bool, also_nps: bool):
    log = repo_root / "decisions_log.md"
    entry = f"""
## {prov['date']}, Day 1: NP/Z stimuli

- Adopted SAP Benchmark NP/Z (Huang et al. 2024) as the base set (commit
  `{prov['repo_commit'][:12]}`); adapted, not authored.
- Distance manipulation = prenominal adjectives only (0/1/2) on the
  temporarily-ambiguous NP; postnominal PP/RC modifiers rejected (would inject a
  second attachment ambiguity).
- Plausibility "norming" = gpt2-large drift screen (held-out scorer), not human
  norming; flagged items reviewed. Primary materials caveat for the writeup.
- Item 14 control spillover {'normalized (removed stray "political")' if item14_fixed else 'left as is'}.
- NP/S replication set {'also generated' if also_nps else 'deferred to Day 2'}.
- Confirmatory test = ambiguity × distance interaction (see analysis_plan.md).
"""
    with log.open("a") as f:
        f.write(entry)
    print(f"[7] appended Day-1 entry to {log.name}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", default="Items for all subsets.xlsx")
    ap.add_argument("--out-dir", default="phase1_stimuli")
    ap.add_argument("--skip-plausibility", action="store_true")
    ap.add_argument("--scorer", default="gpt2-large")
    ap.add_argument("--drift-threshold", type=float, default=2.0,
                    help="flag items with |d3-d1| NLL drift above this (nats)")
    ap.add_argument("--also-nps", action="store_true",
                    help="also template the NP/S replication set")
    args = ap.parse_args()

    xlsx = Path(args.xlsx)
    if not xlsx.exists():
        sys.exit(f"cannot find {xlsx!r}; run from inside the sapbenchmark repo "
                 f"or pass --xlsx")
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    prov = provenance(xlsx, out)
    targets = [("NPZ", "npz_stimuli.csv")]
    if args.also_nps:
        targets.append(("NPS", "nps_stimuli.csv"))

    item14_fixed = False
    for prefix, fname in targets:
        df = load_items(xlsx, prefix)
        validate_pattern(df, prefix)
        if prefix == "NPZ":
            df, item14_fixed = fix_item14(df)
        if not args.skip_plausibility:
            plausibility_gate(df, prefix, args.scorer, out, args.drift_threshold)
        long = build_long(df, CONSTRUCTIONS[prefix].split()[0] if False else prefix)
        long.to_csv(out / fname, index=False)
        print(f"[4] {prefix}: wrote {len(long)} rows "
              f"({long['item'].nunique()} items × 2 × 3) -> {out/fname}")

    emit_docs(out, prov, args.also_nps)
    update_log(Path("."), prov, item14_fixed, args.also_nps)

    print("\nDone. Review phase1_stimuli/ then:")
    print("  git add phase1_stimuli decisions_log.md && "
          "git commit -m 'Day 1: NP/Z stimuli + analysis plan + prereg'")


if __name__ == "__main__":
    main()
