"""
corrected_statistics.py -- corrected statistics for the revised paper.

Fixes relative to fit_mixed_model.py:
  1. Per-model fits are PRIMARY (no pseudo-replication from pooling the
     same 144 sentences scored by two models without a model term).
  2. Pooled fit is SECONDARY and includes a fixed effect of model.
  3. Attempts a by-item random slope for ambiguity; falls back to
     intercept-only and says so.
  4. Reports 95% CIs on the interaction and checks them against a
     smallest effect size of interest (SESOI) for the equivalence claim.
  5. Verifies the all-rows single-token alignment claim from the
     recorded n_critical_tokens column.
  6. Reruns the split-half check under the per-model specification.

Run from ~/price-of-reanalysis with the venv active:
    python corrected_statistics.py
Then paste each printed [PLACEHOLDER] value into main_v2.tex.

Requires: pandas, statsmodels, numpy (already installed).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# SESOI: smallest interaction (bits per distance step) treated as
# theoretically meaningful. Default = 10% of the ~6-bit ambiguity effect
# spread over the 2-step range => 0.3 bits/step. Adjust and justify in
# the paper if you pick differently.
SESOI = 0.3

FILES = {
    "gpt2-large": "phase1_stimuli/surprisal_gpt2-large.csv",
    "pythia-1.4b": "phase1_stimuli/surprisal_EleutherAI_pythia-1.4b.csv",
}

dist_map = {"d1": 1, "d2": 2, "d3": 3}


def load(path, model_label):
    df = pd.read_csv(path)
    df = df.rename(columns={"condition": "ambiguity",
                            "critical_region_surprisal": "surprisal"})
    df["distance_numeric"] = df["distance"].map(dist_map)
    df["model"] = model_label
    return df


def fit_one(df, label):
    """Per-model fit; try by-item random slope for ambiguity, fall back."""
    print(f"\n{'='*70}\nPER-MODEL FIT: {label}\n{'='*70}")
    slope_ok = False
    try:
        m = smf.mixedlm("surprisal ~ C(ambiguity) * distance_numeric",
                        df, groups=df["item"],
                        re_formula="~C(ambiguity)")
        r = m.fit(reml=True)
        conv = getattr(r, "converged", True)
        # statsmodels can 'converge' to a singular fit; check the RE cov
        singular = np.any(np.isclose(np.diag(r.cov_re), 0, atol=1e-8))
        if conv and not singular:
            slope_ok = True
            print("Random-effects structure: intercept + by-item ambiguity slope")
        else:
            raise RuntimeError("slope model singular or non-converged")
    except Exception as e:
        print(f"Slope model failed ({e}); falling back to intercept-only.")
        m = smf.mixedlm("surprisal ~ C(ambiguity) * distance_numeric",
                        df, groups=df["item"], re_formula="~1")
        r = m.fit(reml=True)
        print("Random-effects structure: intercept-only")

    fe, se, p = r.fe_params, r.bse, r.pvalues
    ci = r.conf_int()
    amb = "C(ambiguity)[T.control]"
    inter = "C(ambiguity)[T.control]:distance_numeric"
    dist = "distance_numeric"

    tag = label.upper().replace("-", "").replace(".", "")
    print(f"[BETA-AMB-{tag}]  = {fe[amb]:.3f}")
    print(f"[SE-AMB-{tag}]    = {se[amb]:.3f}")
    print(f"[P-AMB-{tag}]     = {p[amb]:.4g}")
    print(f"[BETA-DIST-{tag}] = {fe[dist]:.3f}   (p = {p[dist]:.4g})")
    print(f"[BETA-INT-{tag}]  = {fe[inter]:.3f}")
    print(f"[CI-INT-{tag}]    = [{ci.loc[inter, 0]:.3f}, {ci.loc[inter, 1]:.3f}]")
    print(f"[P-INT-{tag}]     = {p[inter]:.4g}")

    lo, hi = ci.loc[inter, 0], ci.loc[inter, 1]
    within = (lo > -SESOI) and (hi < SESOI)
    print(f"Equivalence vs SESOI ±{SESOI}: CI "
          f"{'EXCLUDES' if within else 'DOES NOT exclude'} effects beyond SESOI "
          f"-> paper wording: {'supports equivalence' if within else 'qualified invariance'}")
    return r, slope_ok


def main():
    frames = {k: load(v, k) for k, v in FILES.items()}

    # ---- tokenization claim (Methods 5.2 / Results robustness) ----
    print(f"\n{'='*70}\nTOKENIZATION CHECK\n{'='*70}")
    for k, df in frames.items():
        mx = df["n_critical_tokens"].max()
        mn = df["n_critical_tokens"].min()
        print(f"{k}: n_critical_tokens min={mn} max={mx} "
              f"-> single-token claim {'HOLDS' if mx == 1 else 'FAILS: fix Methods text'}")

    # ---- primary: per-model fits ----
    for k, df in frames.items():
        fit_one(df, k)

    # ---- secondary: pooled with fixed effect of model ----
    print(f"\n{'='*70}\nPOOLED FIT (secondary; fixed effect of model)\n{'='*70}")
    pooled = pd.concat(frames.values(), ignore_index=True)
    m = smf.mixedlm("surprisal ~ C(ambiguity) * distance_numeric + C(model)",
                    pooled, groups=pooled["item"], re_formula="~1")
    r = m.fit(reml=True)
    inter = "C(ambiguity)[T.control]:distance_numeric"
    ci = r.conf_int()
    print(f"[BETA-INT-POOLED] = {r.fe_params[inter]:.3f}")
    print(f"[CI-INT-POOLED]   = [{ci.loc[inter,0]:.3f}, {ci.loc[inter,1]:.3f}]")
    print(f"[P-INT-POOLED]    = {r.pvalues[inter]:.4g}")

    # ---- split-half under the corrected per-model spec ----
    print(f"\n{'='*70}\nSPLIT-HALF (per-model spec, intercept-only for stability)\n{'='*70}")
    rng = np.random.default_rng(42)
    items = np.array(sorted(pooled["item"].unique()))
    rng.shuffle(items)
    halves = {"half A": items[: len(items) // 2],
              "half B": items[len(items) // 2:]}
    for hname, hitems in halves.items():
        for k, df in frames.items():
            sub = df[df["item"].isin(hitems)]
            mm = smf.mixedlm("surprisal ~ C(ambiguity) * distance_numeric",
                             sub, groups=sub["item"], re_formula="~1")
            rr = mm.fit(reml=False)
            amb = "C(ambiguity)[T.control]"
            it = "C(ambiguity)[T.control]:distance_numeric"
            print(f"{hname:7s} {k:12s} ambiguity {rr.fe_params[amb]:+.2f} "
                  f"(p={rr.pvalues[amb]:.3g})  interaction p={rr.pvalues[it]:.3f}")

    print("\nDone. Paste the bracketed values into main_v2.tex, decide the")
    print("SESOI wording, and log the analysis change in decisions_log.md.")


if __name__ == "__main__":
    main()
