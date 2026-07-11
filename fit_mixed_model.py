import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Combine models
df_gpt2 = pd.read_csv("phase1_stimuli/surprisal_gpt2-large.csv")
df_pythia = pd.read_csv("phase1_stimuli/surprisal_EleutherAI_pythia-1.4b.csv")
df = pd.concat([df_gpt2, df_pythia], ignore_index=True)

# Rename for model compatibility
df = df.rename(columns={
    "condition": "ambiguity",
    "critical_region_surprisal": "surprisal"
})

# Convert distance to numeric for interaction (d1=1, d2=2, d3=3)
dist_map = {"d1": 1, "d2": 2, "d3": 3}
df["distance_numeric"] = df["distance"].map(dist_map)

# Fit mixed-effects model using statsmodels
print("Fitting mixed-effects model...")
model = smf.mixedlm("surprisal ~ C(ambiguity) * distance_numeric", 
                     df, 
                     groups=df["item"], 
                     re_formula="~1")
result = model.fit()
print(result.summary())

# Extract the fixed effects (the interaction test)
print("\n" + "="*60)
print("Fixed effects (key test: ambiguity:distance_numeric interaction)")
print("="*60)
print(result.fe_params)

# Compute GP effect (ambiguous - control) per distance, per model
print("\n" + "="*60)
print("GP effect (ambiguous - control) by distance and model")
print("="*60)
for model_name in sorted(df["model"].unique()):
    df_model = df[df["model"] == model_name]
    for dist in ["d1", "d2", "d3"]:
        amb_mean = df_model[(df_model["ambiguity"] == "ambiguous") & (df_model["distance"] == dist)]["surprisal"].mean()
        ctrl_mean = df_model[(df_model["ambiguity"] == "control") & (df_model["distance"] == dist)]["surprisal"].mean()
        gp_effect = amb_mean - ctrl_mean
        print(f"{model_name:30} {dist}: {gp_effect:.2f} bits")

# Headline figure: GP effect vs. distance, per model
fig, ax = plt.subplots(figsize=(8, 5))

for model_name in sorted(df["model"].unique()):
    df_model = df[df["model"] == model_name]
    distances = ["d1", "d2", "d3"]
    gp_effects = []
    for dist in distances:
        amb_mean = df_model[(df_model["ambiguity"] == "ambiguous") & (df_model["distance"] == dist)]["surprisal"].mean()
        ctrl_mean = df_model[(df_model["ambiguity"] == "control") & (df_model["distance"] == dist)]["surprisal"].mean()
        gp_effects.append(amb_mean - ctrl_mean)
    
    ax.plot(distances, gp_effects, marker="o", label=model_name, linewidth=2, markersize=8)

ax.set_xlabel("Distance (d1, d2, d3)", fontsize=12)
ax.set_ylabel("Garden-path effect (bits)", fontsize=12)
ax.set_title("Garden-path effect by distance and model", fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("phase1_stimuli/garden_path_by_distance.png", dpi=150)
print("\nFigure saved to phase1_stimuli/garden_path_by_distance.png")
plt.close()
