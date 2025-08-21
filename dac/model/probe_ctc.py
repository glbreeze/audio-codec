# probe_ctc_on_latents.py
# Minimal BLSTM-CTC probe for codec latents -> text (chars).
# - Supports DISCRETE codes (e.g., RVQ indices) or CONTINUOUS embeddings.
# - Computes train/dev WER and an MI-style contrastive score.
#
# Data layout (example):
#   data_root/
#     latents/
#       1089-134686-0000.npy  # shape [T,M] for DISCRETE (M codebooks) or [T,D] for CONTINUOUS
#       ...
#     refs.csv                # columns: utt_id,text
#
# Usage:
#   python probe_ctc_on_latents.py \
#     --data_root /path/to/data_root \
#     --mode discrete --codebooks 8 --vocab 1024 \
#     --epochs 5 --batch_size 32
#
#   python probe_ctc_on_latents.py \
#     --data_root /path/to/data_root \
#     --mode continuous --in_dim 256 \
#     --epochs 5 --batch_size 32
#
# Notes:
# - Char set is a–z, space, apostrophe, digits; lowercased.
# - BLSTM + CTC (blank=0). Length-normalized CTC for MI proxy (optional).
# - For DISCRETE: each codebook has its own nn.Embedding; we SUM over codebooks per frame.

import argparse, math, random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Utils: text <-> ids
# --------------------------
def build_charset():
    # LibriSpeech-friendly
    chars = list("abcdefghijklmnopqrstuvwxyz '0123456789")
    stoi = {c:i+1 for i,c in enumerate(chars)}  # 0 reserved for CTC blank
    itos = {i:c for c,i in stoi.items()}
    return stoi, itos

def text_to_ids(text, stoi):
    text = text.lower().strip()
    return [stoi[c] for c in text if c in stoi]

def ids_to_text(ids, itos):
    return "".join(itos.get(i, "") for i in ids)

def ctc_greedy_decode(logits, blank=0):
    # logits: [B,T,C] log-probs
    pred = logits.argmax(-1)  # [B,T]
    hyps = []
    for seq in pred.cpu().tolist():
        hyp = []
        prev = None
        for p in seq:
            if p != blank and p != prev:
                hyp.append(p)
            prev = p
        hyps.append(hyp)
    return hyps

def wer(ref, hyp):
    # tiny WER for quick probe (char/word insensitive; treat tokens as chars sequence)
    r = "".join(ref).split()
    h = "".join(hyp).split()
    # fallback to char-level if no spaces (common)
    if not r or not h:
        r, h = list(ref), list(hyp)
    # Levenshtein
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): dp[i][0] = i
    for j in range(len(h)+1): dp[0][j] = j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1] / max(1,len(r))

# --------------------------
# Dataset
# --------------------------
class LatentTextSet(torch.utils.data.Dataset):
    def __init__(self, root, split_ratio=0.9):
        root = Path(root)
        self.lat_dir = root / "latents"
        refs = pd.read_csv(root / "refs.csv")
        self.items = [(u, t) for u, t in refs[["utt_id","text"]].itertuples(index=False)]
        # simple split
        n = int(len(self.items)*split_ratio)
        self.tr_items = self.items[:n]
        self.dv_items = self.items[n:]

    def _load(self, uid):
        arr = np.load(self.lat_dir / f"{uid}.npy")  # [T,M] or [T,D]
        return torch.from_numpy(arr).float()

# --------------------------
# Model
# --------------------------
class EmbeddingStack(nn.Module):
    """Embedding stack for DISCRETE RVQ codes shape [B,T,M] with vocab V per codebook."""
    def __init__(self, codebooks, vocab, d_emb):
        super().__init__()
        self.M = codebooks
        if isinstance(vocab, int):
            self.embs = nn.ModuleList([nn.Embedding(vocab, d_emb) for _ in range(codebooks)])
        else:
            self.embs = nn.ModuleList([nn.Embedding(vocab[0], d_emb)] + 
                                      [nn.Embedding(vocab[1], d_emb) for _ in range(1, codebooks)])
    def forward(self, x):  # x: [B,T,M] long
        embs = [self.embs[m](x[:,:,m]) for m in range(self.M)]  # list of [B,T,d]
        return torch.stack(embs, dim=0).sum(0)  # [B,T,d] (sum across codebooks)

class LatentCTCProbe(nn.Module):
    def __init__(self, mode, d_in=None, codebooks=None, vocab=None, d_emb=128, hidden=256, n_layers=2, n_chars=1+len(build_charset()[0])):
        super().__init__()
        assert mode in {"discrete","continuous"}
        self.mode = mode
        if mode == "discrete":
            assert codebooks and vocab
            self.frontend = EmbeddingStack(codebooks, vocab, d_emb)
            d_front = d_emb
        else:
            assert d_in is not None
            self.frontend = nn.Linear(d_in, d_emb)
            d_front = d_emb
            
        self.rnn = nn.LSTM(d_front, hidden, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden*2, n_chars)  # include blank at idx 0
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, x_lens):
        # x: [B,T,M] (long) for discrete OR [B,T,D] float for continuous
        if self.mode == "discrete" and x.dtype != torch.long:
            x = x.long()
        h = self.frontend(x)                # [B,T,d]
        
        h, _ = self.rnn(h)                  # [B,T,2H]
        logits = self.log_softmax(self.out(h))  # [B,T,C]
        # pack to T,B,C for CTC
        logits = logits.transpose(0,1)      # [T,B,C]
        return logits

# --------------------------
# Collate
# --------------------------
def collate_batch(batch, mode, stoi):
    # batch: list of (uid, x_tensor [T,M] or [T,D], text)
    xs, xlens, ys, ylens, uids = [], [], [], [], []
    for uid, x, text in batch:
        uids.append(uid)
        xs.append(x)
        xlens.append(x.size(0))
        y = torch.tensor(text_to_ids(text, stoi), dtype=torch.long)
        ys.append(y)
        ylens.append(len(y))
    # pad xs
    T = max(xlens)
    if mode == "discrete":
        M = xs[0].size(1)
        xpad = torch.zeros(len(xs), T, M, dtype=torch.long)
        for i,x in enumerate(xs): xpad[i,:x.size(0)] = x.long()
    else:
        D = xs[0].size(1)
        xpad = torch.zeros(len(xs), T, D, dtype=torch.float32)
        for i,x in enumerate(xs): xpad[i,:x.size(0)] = x.float()
    # concat ys
    ycat = torch.cat(ys) if len(ys) else torch.empty(0, dtype=torch.long)
    return uids, xpad, torch.tensor(xlens, dtype=torch.long), ycat, torch.tensor(ylens, dtype=torch.long)

# --------------------------
# Training / Eval
# --------------------------
def run_epoch(model, loader, mode, stoi, itos, optimizer=None, device="cpu", neg_k=2):
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    model.train(optimizer is not None)
    tot_loss = 0.0
    all_refs, all_hyps = [], []
    mi_pos, mi_neg = 0.0, 0.0
    for batch in loader:
        uids, xpad, xlens, ycat, ylens = collate_batch(batch, mode, stoi)
        xpad, xlens, ycat, ylens = xpad.to(device), xlens.to(device), ycat.to(device), ylens.to(device)
        logits = model(xpad, xlens)               # [T,B,C]
        # POS: matched transcripts
        loss_pos = ctc_loss(logits, ycat, xlens, ylens)
        # NEG: randomly permute transcripts across utterances (approx E_j!=i)
        # build a permuted target by chunking ycat by ylens
        with torch.no_grad():
            chunks = torch.split(ycat, ylens.cpu().tolist())
            idx = list(range(len(chunks)))
            random.shuffle(idx)
            neg = torch.cat([chunks[j] for j in idx]).to(device)
        loss_neg = ctc_loss(logits, neg, xlens, ylens)

        loss = loss_pos if optimizer is None else loss_pos
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        tot_loss += float(loss_pos.detach().cpu())

        # Greedy decode for WER
        with torch.no_grad():
            lp = logits.transpose(0,1)  # [B,T,C] log-probs
            hyps_ids = ctc_greedy_decode(lp)
            # Rebuild references to text
            refs_txt, hyps_txt = [], []
            start = 0
            chunks = torch.split(ycat.cpu(), ylens.cpu().tolist())
            for r_ids, h_ids in zip(chunks, hyps_ids):
                refs_txt.append(ids_to_text(r_ids.tolist(), itos))
                hyps_txt.append(ids_to_text(h_ids, itos))
            all_refs.extend(refs_txt)
            all_hyps.extend(hyps_txt)

        # MI accumulators (use -CTC as log q)
        mi_pos += -float(loss_pos.detach().cpu())
        mi_neg += -float(loss_neg.detach().cpu())

    # WER over the epoch
    wers = [wer(r, h) for r, h in zip(all_refs, all_hyps)] if all_refs else [1.0]
    # MI estimate ~ E[log q(y|x) - log q(y'|x)]
    mi_hat = (mi_pos - mi_neg) / max(1, len(loader))
    return np.mean(wers), tot_loss/ max(1,len(loader)), mi_hat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--mode", choices=["discrete","continuous"], required=True)
    ap.add_argument("--codebooks", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=None, help="per-codebook size for DISCRETE")
    ap.add_argument("--in_dim", type=int, default=None, help="input dim for CONTINUOUS")
    ap.add_argument("--d_emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    stoi, itos = build_charset()
    ds = LatentTextSet(args.data_root)

    # Small helper to feed dataset samples as (uid, x, text)
    def make_sampler(items):
        for uid, text in items:
            x = ds._load(uid)  # [T,M] or [T,D]
            yield (uid, x, text)

    # DataLoaders
    def collate_list(batch):  # leave to collate later (we need stoi inside)
        return batch
    tr_loader = torch.utils.data.DataLoader(list(make_sampler(ds.tr_items)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_list)
    dv_loader = torch.utils.data.DataLoader(list(make_sampler(ds.dv_items)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_list)

    # Infer dims for model
    ex_uid, ex_text = ds.tr_items[0]
    ex = ds._load(ex_uid)
    if args.mode == "discrete":
        assert ex.dim()==2, "Expect DISCRETE codes as [T,M] integers in .npy"
        codebooks = args.codebooks or ex.size(1)
        vocab = args.vocab or 1024
        model = LatentCTCProbe("discrete", codebooks=codebooks, vocab=vocab, d_emb=args.d_emb,
                               hidden=args.hidden, n_layers=args.layers)
    else:
        assert ex.dim()==2, "Expect CONTINUOUS embeddings as [T,D] floats in .npy"
        in_dim = args.in_dim or ex.size(1)
        model = LatentCTCProbe("continuous", d_in=in_dim, d_emb=args.d_emb,
                               hidden=args.hidden, n_layers=args.layers)

    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        tr_wer, tr_loss, tr_mi = run_epoch(model, tr_loader, args.mode, stoi, itos, optimizer=opt, device=args.device)
        dv_wer, dv_loss, dv_mi = run_epoch(model, dv_loader, args.mode, stoi, itos, optimizer=None, device=args.device)
        print(f"[ep {ep:02d}] train WER={tr_wer:.3f} loss={tr_loss:.3f} MI={tr_mi:.3f} | dev WER={dv_wer:.3f} loss={dv_loss:.3f} MI={dv_mi:.3f}")

if __name__ == "__main__":
    main()
