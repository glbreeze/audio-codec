from pathlib import Path
import numpy as np
import pandas as pd
import re, unicodedata

import torch
import torch.utils.data as tud


stoi = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 
        'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 
        'x': 24, 'y': 25, 'z': 26, ' ': 27, "'": 28, '0': 29, '1': 30, '2': 31, '3': 32, '4': 33, '5': 34, 
        '6': 35, '7': 36, '8': 37, '9': 38}
itos = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l',
        13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w',
        24: 'x', 25: 'y', 26: 'z', 27: ' ', 28: "'", 29: '0', 30: '1', 31: '2', 32: '3', 33: '4', 34: '5',
        35: '6', 36: '7', 37: '8', 38: '9'}

def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).lower() # 1) Unicode normalize + lowercase

    # 2) Map curly/backtick quotes to ASCII apostrophe (keep apostrophes in words)
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = t.replace("“", " ").replace("”", " ").replace('"', " ")

    # 3) Treat hyphen-like separators as spaces
    t = re.sub(r"[-–—−_/\\]", " ", t)

    # 4) Keep only [a-z 0-9 ' ] ; drop everything else
    t = re.sub(r"[^a-z0-9' ]", " ", t)

    # 5) Collapse spaces
    t = re.sub(r"\s+", " ", t).strip()
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
                "text_len": len(text),
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

    def text_to_ids(text):
        return [stoi[c] for c in text if c in stoi]

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