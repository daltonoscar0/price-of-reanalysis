"""
day7_centered_refit.py — centered recoding for rev-4.
Primary per-model fits: REML, random intercept + by-item ambiguity slope,
dist_c = distance - 2, ambig_c = +/-0.5 so the coefficient is the GP
effect (ambiguous minus control) at mean distance.
Split-half: same halves as robustness_checks.py (train_test_split,
random_state=42), but fit PER MODEL, fixing the pooling in that script.
"""
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

df = pd.concat([
    pd.read_csv("phase1_stimuli/surprisal_gpt2-large.csv"),
    pd.read_csv("phase1_stimuli/surprisal_EleutherAI_pythia-1.4b.csv"),
], ignore_index=True).rename(columns={
    "condition": "ambiguity", "critical_region_surprisal": "surprisal"})

df["dist_c"] = df["distance"].map({"d1": 1, "d2": 2, "d3": 3}) - 2
df["ambig_c"] = df["ambiguity"].map({"control": -0.5, "ambiguous": 0.5})

def fit(sub, slope=True):
    rf = "~ambig_c" if slope else "~1"
    r = smf.mixedlm("surprisal ~ ambig_c * dist_c", sub,
                    groups=sub["item"], re_formula=rf).fit(reml=True)
    return r

def fmt_p(p):
    return "p < .001" if p < .001 else f"p = {p:.3f}"

print("=" * 64)
print("PRIMARY per-model fits (centered, REML, by-item ambiguity slope)")
print("=" * 64)
for name, sub in df.groupby("model"):
    r = fit(sub)
    b, se, p = r.params, r.bse, r.pvalues
    ci = r.conf_int()
    print(f"\n{name}  (converged: {r.converged})")
    print(f"  6.1 ambiguity at mean distance: {b['ambig_c']:.2f} bits "
          f"($SE = {se['ambig_c']:.2f}$, ${fmt_p(p['ambig_c'])}$)")
    print(f"  sanity vs 6.2 — interaction: {b['ambig_c:dist_c']:.2f}/step, "
          f"95% CI [{ci.loc['ambig_c:dist_c',0]:.2f}, "
          f"{ci.loc['ambig_c:dist_c',1]:.2f}], {fmt_p(p['ambig_c:dist_c'])}")
    print(f"  sanity — distance main effect: {b['dist_c']:.2f}/step, "
          f"{fmt_p(p['dist_c'])}")

print("\n" + "=" * 64)
print("SPLIT-HALF per model (same halves as robustness_checks.py)")
print("=" * 64)
all_items = df["item"].unique()          # same ordering as robustness_checks
h1, h2 = train_test_split(all_items, test_size=0.5, random_state=42)
for name, sub in df.groupby("model"):
    for label, items in (("train-half", h1), ("test-half", h2)):
        s = sub[sub["item"].isin(items)]
        try:
            r = fit(s)
            note = ""
        except Exception:
            r = fit(s, slope=False)
            note = " [slope dropped: 12-item fit did not converge]"
        print(f"  {name:28} {label}: ambiguity "
              f"{r.params['ambig_c']:.2f} bits ({fmt_p(r.pvalues['ambig_c'])}), "
              f"interaction {fmt_p(r.pvalues['ambig_c:dist_c'])}{note}")
print("\nCompare split-half values against the draft's 7.00/6.44 and "
      "6.39/5.44; see notes in chat if they don't match.")
