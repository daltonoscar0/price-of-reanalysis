import re
import pandas as pd
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2-large")
df = pd.read_csv("phase1_stimuli/npz_stimuli.csv").head(5)

for _, row in df.iterrows():
    sentence, crit_word = row["sentence"], row["critical_word"]
    encoding = tok(sentence, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tok.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    m = re.search(r'\b' + re.escape(crit_word) + r'\b', sentence)
    crit_start, crit_end = (m.start(), m.end()) if m else (None, None)

    print(f"\nSENTENCE: {sentence}")
    print(f"CRITICAL WORD: {crit_word!r}  char span [{crit_start}:{crit_end}]")
    for t, (s, e) in zip(tokens, offsets):
        marker = "  <-- CRITICAL" if (crit_start is not None and s < crit_end and e > crit_start) else ""
        print(f"  {t!r:15} chars[{s}:{e}] = {sentence[s:e]!r}{marker}")
