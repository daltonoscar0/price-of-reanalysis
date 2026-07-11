import re
import time
import pandas as pd
from tqdm import tqdm
from minicons import scorer
from transformers import AutoTokenizer

MODEL_NAME = "gpt2-large"
CSV_IN = "phase1_stimuli/npz_stimuli.csv"
CSV_OUT = "phase1_stimuli/surprisal_gpt2-large_initial.csv"

print(f"Loading {MODEL_NAME}...")
t0 = time.time()
model = scorer.IncrementalLMScorer(MODEL_NAME, device="cpu")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Loaded in {time.time()-t0:.1f}s")

df = pd.read_csv(CSV_IN)
rows = []
skipped = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring sentences"):
    sentence = row["sentence"]
    crit_word = row["critical_word"]

    m = re.search(r'\b' + re.escape(crit_word) + r'\b', sentence)
    if not m:
        skipped.append((row["item"], row["condition"], row["distance"], crit_word))
        continue
    crit_start, crit_end = m.start(), m.end()

    encoding = tok(sentence, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]

    token_surprisals = model.token_score([sentence], surprisal=True, base_two=True)[0]

    crit_token_idxs = [
        i for i, (s, e) in enumerate(offsets)
        if s < crit_end and e > crit_start
    ]

    crit_surprisal = sum(token_surprisals[i][1] for i in crit_token_idxs)

    rows.append({
        "item": row["item"],
        "construction": row["construction"],
        "condition": row["condition"],
        "distance": row["distance"],
        "critical_word": crit_word,
        "sentence": sentence,
        "critical_region_surprisal": crit_surprisal,
        "n_critical_tokens": len(crit_token_idxs),
    })

out = pd.DataFrame(rows)
out.to_csv(CSV_OUT, index=False)
print(f"Wrote {len(out)} rows to {CSV_OUT}")
if skipped:
    print(f"WARNING: {len(skipped)} rows skipped, critical word not found:")
    for s in skipped:
        print("  ", s)
print(out.head())
