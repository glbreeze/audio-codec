from pathlib import Path
import numpy as np
import pandas as pd
import re, unicodedata

import torch
import torch.utils.data as tud
from torchaudio.models.decoder import download_pretrained_files


# ========= define the tokens  ========= 
files = download_pretrained_files("librispeech-4-gram")
with open(files.tokens, 'r') as f:
    tokens = [line.strip() for line in f]

stoi = {token: idx for idx, token in enumerate(tokens)}
itos = {idx: token for idx, token in enumerate(tokens)}

def text_to_ids(text):
    text = text.lower().strip()
    return [stoi[c] for c in text if c in stoi]

def ids_to_text(ids):
    return "".join(itos.get(i, "") for i in ids)


# ========= preprocess the target transcript =========
def normalize_text(text: str) -> str:
    # 1) Unicode normalize + lowercase
    t = unicodedata.normalize("NFKD", text).lower()

    # 2) Normalize quotes → apostrophe
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = t.replace("“", " ").replace("”", " ").replace('"', " ")

    # 3) Treat hyphen-like separators as spaces
    t = re.sub(r"[-–—−_/\\]", " ", t)

    # 4) Keep only [a-z ' ] ; drop digits and other punctuation
    t = re.sub(r"[^a-z' ]", " ", t)

    # 5) Collapse multiple spaces
    t = re.sub(r"\s+", " ", t).strip()

    # 6) Replace space with '|' (LibriSpeech token for space)
    t = t.replace(" ", "|")

    return t


def build_manifest(root_dir: Path, split: str) -> Path:
    
    root_dir = Path(root_dir)
    data_dir = root_dir / split
    rows = []
    for f in sorted(data_dir.glob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            text = str(z["text"])
            rows.append({
                "utt_id": str(z["utt_id"]), 
                "file": str(f),
                "frames_code": int(z["code"].shape[-1]),     # [K, T]
                "frames_latent": int(z["latent"].shape[-1]), # [8, T] / [16, T]
                "text": text,  # optional convenience
            })
    manifest = root_dir / f"manifest_{split}.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


# ---------- Loading for training ----------

class CodecDataset(tud.Dataset):
    def __init__(self, manifest_csv: str | Path, input_type: str = "discrete"):
        self.df = pd.read_csv(manifest_csv)
        assert input_type in ("discrete", "continuous")
        self.input_type = input_type
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = Path(row["file"])
        with np.load(path, allow_pickle=True) as z:
            utt_id = str(z["utt_id"])
            text = str(z["text"])
            text_norm = normalize_text(text)
            if self.input_type == "discrete":
                x = torch.from_numpy(z["code"].astype(np.int64, copy=False)).T         # [M, T]->[T, M]
            else:
                x = torch.from_numpy(z["latent"].astype(np.float32, copy=False)).T     # [D, T]->[T, D]
        return {"utt_id": utt_id, "x": x, "text": text_norm}


def build_collate(kind: str):
    assert kind in ("discrete", "continuous")

    def collate(batch):
        # batch: list of dicts {"utt_id", "x", "text"}
        uids, xs, xlens, ys, ylens = [], [], [], [], []
        for item in batch:
            uid, x, text = item["utt_id"], item["x"], item["text"]
            uids.append(uid)
            xs.append(x)
            xlens.append(x.size(0))
            y = torch.tensor(text_to_ids(text), dtype=torch.long)
            ys.append(y)
            ylens.append(len(y))

        T = max(xlens)
        if kind == "discrete":
            M = xs[0].size(1)
            xpad = torch.zeros(len(xs), T, M, dtype=torch.long)
            for i, x in enumerate(xs):
                xpad[i, : x.size(0)] = x
        else:
            D = xs[0].size(1)
            xpad = torch.zeros(len(xs), T, D, dtype=torch.float32)
            for i, x in enumerate(xs):
                xpad[i, : x.size(0)] = x.float()

        ycat  = torch.cat(ys) if len(ys) else torch.empty(0, dtype=torch.long)
        xlens = torch.tensor(xlens, dtype=torch.long)
        ylens = torch.tensor(ylens, dtype=torch.long)

        return {"utt_id": uids, "x": xpad, "xlens": xlens, "ycat": ycat, "ylens": ylens}

    return collate