"""One-time: extract ProsusAI/finbert [CLS] last-hidden-state embeddings for
every unique headline and cache to artifacts/finbert_cls_embeddings.parquet.

The existing sentiment cache throws away ~95% of the model's semantic signal
(only keeps argmax + score). This keeps the full 768-dim context vector so a
downstream ridge can read richer information than the 3-class label provides.

~34k headlines. MPS batch=32 runs in a few minutes on Apple Silicon.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
OUT = ART / "finbert_cls_embeddings.parquet"

MODEL_NAME = "ProsusAI/finbert"
BATCH = 32
MAX_LEN = 128

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device = {device}")

cache = pd.read_parquet(DATA / "headlines_finbert_sentiment.parquet")
headlines = cache["headline"].drop_duplicates().tolist()
print(f"unique headlines: {len(headlines)}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

all_embs = np.zeros((len(headlines), 768), dtype=np.float32)
with torch.no_grad():
    for i in range(0, len(headlines), BATCH):
        batch = headlines[i:i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        all_embs[i:i + len(batch)] = cls
        if (i // BATCH) % 50 == 0:
            print(f"  embedded {i + len(batch)} / {len(headlines)}")

df = pd.DataFrame(all_embs, columns=[f"emb_{j}" for j in range(768)])
df.insert(0, "headline", headlines)
df.to_parquet(OUT)
print(f"saved {OUT}  shape={df.shape}")
