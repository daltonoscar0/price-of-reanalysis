import re
import time
import pandas as pd
from tqdm import tqdm
from minicons import scorer
from transformers import AutoTokenizer

models = ["gpt2-large", "EleutherAI/pythia-1.4b"]
CSV_IN = "phase1_stimuli/npz_stimuli.csv"

df = pd.read_csv(CSV_IN)

for model_name in models:
    print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
    t0 = time.time()
    model = scorer.IncrementalLMScorer(model_name, device="cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded in {time.time()-t0:.1f}s")

    rows = []
    skipped = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
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
            "model": model_name,
        })

    out = pd.DataFrame(rows)
    csv_out = f"phase1_stimuli/surprisal_{model_name.replace('/', '_')}.csv"
    out.to_csv(csv_out, index=False)
    print(f"Wrote {len(out)} rows to {csv_out}")
    if skipped:
        print(f"WARNING: {len(skipped)} skipped")
    print(out.groupby(['condition','distance'])['critical_region_surprisal'].agg(['mean','std']))
