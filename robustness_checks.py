import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

# Load combined data
df_gpt2 = pd.read_csv("phase1_stimuli/surprisal_gpt2-large.csv")
df_pythia = pd.read_csv("phase1_stimuli/surprisal_EleutherAI_pythia-1.4b.csv")
df = pd.concat([df_gpt2, df_pythia], ignore_index=True)

df = df.rename(columns={
    "condition": "ambiguity",
    "critical_region_surprisal": "surprisal"
})

dist_map = {"d1": 1, "d2": 2, "d3": 3}
df["distance_numeric"] = df["distance"].map(dist_map)

# ============================================================================
# CHECK 1: Hold-out replication on 50% of items
# ============================================================================
print("="*60)
print("CHECK 1: Hold-out replication (50% of items held out)")
print("="*60)

all_items = df["item"].unique()
train_items, test_items = train_test_split(all_items, test_size=0.5, random_state=42)

df_train = df[df["item"].isin(train_items)]
df_test = df[df["item"].isin(test_items)]

# Fit on training set
model_train = smf.mixedlm("surprisal ~ C(ambiguity) * distance_numeric", 
                          df_train, 
                          groups=df_train["item"], 
                          re_formula="~1")
result_train = model_train.fit(reml=False)

print(f"Training set ({len(train_items)} items, {len(df_train)} obs):")
print(f"  Ambiguity effect: {result_train.fe_params['C(ambiguity)[T.control]']:.3f} (p={result_train.pvalues['C(ambiguity)[T.control]']:.4f})")
print(f"  Interaction: {result_train.fe_params['C(ambiguity)[T.control]:distance_numeric']:.3f} (p={result_train.pvalues['C(ambiguity)[T.control]:distance_numeric']:.4f})")

# Fit on test set
model_test = smf.mixedlm("surprisal ~ C(ambiguity) * distance_numeric", 
                         df_test, 
                         groups=df_test["item"], 
                         re_formula="~1")
result_test = model_test.fit(reml=False)

print(f"\nTest set ({len(test_items)} items, {len(df_test)} obs):")
print(f"  Ambiguity effect: {result_test.fe_params['C(ambiguity)[T.control]']:.3f} (p={result_test.pvalues['C(ambiguity)[T.control]']:.4f})")
print(f"  Interaction: {result_test.fe_params['C(ambiguity)[T.control]:distance_numeric']:.3f} (p={result_test.pvalues['C(ambiguity)[T.control]:distance_numeric']:.4f})")

# ============================================================================
# CHECK 2: Direction matches human norms (ambiguous > control)
# ============================================================================
print("\n" + "="*60)
print("CHECK 2: Effect direction (should be ambiguous > control)")
print("="*60)
for model_name in sorted(df["model"].unique()):
    df_model = df[df["model"] == model_name]
    amb_mean = df_model[df_model["ambiguity"] == "ambiguous"]["surprisal"].mean()
    ctrl_mean = df_model[df_model["ambiguity"] == "control"]["surprisal"].mean()
    direction = "✓ CORRECT" if amb_mean > ctrl_mean else "✗ WRONG"
    print(f"{model_name:30} ambiguous={amb_mean:.2f}, control={ctrl_mean:.2f} {direction}")

# ============================================================================
# CHECK 3: Summary vs. full-sample model (fit_mixed_model.py)
# ============================================================================
print("\n" + "="*60)
print("CHECK 3: Full-sample model (all 24 items) for reference")
print("="*60)

model_all = smf.mixedlm("surprisal ~ C(ambiguity) * distance_numeric", 
                        df, 
                        groups=df["item"], 
                        re_formula="~1")
result_all = model_all.fit(reml=False)

print(f"Full sample ({len(all_items)} items, {len(df)} obs):")
print(f"  Ambiguity effect: {result_all.fe_params['C(ambiguity)[T.control]']:.3f} (p={result_all.pvalues['C(ambiguity)[T.control]']:.4f})")
print(f"  Interaction: {result_all.fe_params['C(ambiguity)[T.control]:distance_numeric']:.3f} (p={result_all.pvalues['C(ambiguity)[T.control]:distance_numeric']:.4f})")

print("\nROBUSTNESS SUMMARY:")
print(f"  Ambiguity effect stable across train/test? {abs(result_train.fe_params['C(ambiguity)[T.control]'] - result_test.fe_params['C(ambiguity)[T.control]']) < 1.0}")
print(f"  Interaction non-significant in all splits? train p={result_train.pvalues['C(ambiguity)[T.control]:distance_numeric']:.3f}, test p={result_test.pvalues['C(ambiguity)[T.control]:distance_numeric']:.3f}, all p={result_all.pvalues['C(ambiguity)[T.control]:distance_numeric']:.3f}")
