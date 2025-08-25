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

from process_data.asr_data import stoi, itos, build_collate, CodecDataset, build_manifest
from dac.model.probe_ctc import LatentCTCProbe

# --------------------------
# Utils: text <-> ids
# --------------------------


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

def compute_wer(ref, hyp):
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
# Training / Eval
# --------------------------
def run_epoch(model, loader, optimizer=None, device="cpu", neg_permute=True, codebooks=None, input_type=None):
    """
    model: your BLSTM-CTC probe. forward(x, xlens) -> log_probs [T,B,C]
    loader: yields {"utt_id", "x","xlens","ycat","ylens"}
    optimizer: if provided, we train; else we eval.
    device: "cuda" or "cpu"
    neg_permute: if True, compute contrastive MI proxy with permuted transcripts
    returns: (avg_WER, avg_pos_CTC_loss, mi_hat)
    """
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    training = optimizer is not None
    model.train(training)

    total_pos_loss = 0.0
    total_batches = 0
    all_ref_texts, all_hyp_texts = [], []
    mi_pos_sum, mi_neg_sum = 0.0, 0.0

    for batch in loader:
        uids  = batch["utt_id"]
        xlens = batch["xlens"]; xpad  = batch["x"]       # [B,T,M] long OR [B,T,D] float
        ylens = batch["ylens"]; ycat  = batch["ycat"]    # concatenated targets (1-D)
        
        if input_type == 'discrete' and codebooks is not None and codebooks>0:
            xpad = xpad[:, :, :codebooks]
    
        xpad  = xpad.to(device); xlens = xlens.to(device)
        ycat  = ycat.to(device); ylens = ylens.to(device)

        logits_TBC = model(xpad, xlens)  # [T,B,C]

        if (ylens > xlens).any(): #  CTC needs input_len >= target_len per item
            print("Skipping batch with ylens > xlens")
            continue

        # Positive (matched) CTC loss
        loss_pos = ctc(logits_TBC, ycat, xlens, ylens)

        # Optional negative (mismatched) for MI proxy
        if neg_permute:
            with torch.no_grad():
                chunks = torch.split(ycat, ylens.cpu().tolist())
                perm = torch.randperm(len(chunks)) # random permutation across items
                yneg = torch.cat([chunks[i] for i in perm]).to(device)
            loss_neg = ctc(logits_TBC, yneg, xlens, ylens)
        else:
            loss_neg = torch.tensor(0.0, device=device)

        # Backprop (only on positive matched loss)
        if training:
            optimizer.zero_grad()
            loss_pos.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Decode greedily for WER
        with torch.no_grad():
            hyps_ids = ctc_greedy_decode(logits_TBC.transpose(0,1), blank=0)  # [T, B, C] -> list of int ids
            # rebuild reference text per sample from ycat+ylens
            ref_chunks = torch.split(ycat.detach().cpu(), ylens.detach().cpu().tolist())
            for ref_ids, hyp_ids in zip(ref_chunks, hyps_ids):
                ref_txt = ids_to_text(ref_ids.tolist(), itos)
                hyp_txt = ids_to_text(hyp_ids, itos)
                all_ref_texts.append(ref_txt)
                all_hyp_texts.append(hyp_txt)

        total_pos_loss += float(loss_pos.detach().cpu())
        if neg_permute:
            mi_pos_sum += -float(loss_pos.detach().cpu())
            mi_neg_sum += -float(loss_neg.detach().cpu())
        total_batches += 1

    # Aggregate
    avg_ctc_pos = total_pos_loss / max(1, total_batches)
    wers = [compute_wer(r, h) for r, h in zip(all_ref_texts, all_hyp_texts)]
    avg_wer = float(np.mean(wers))

    # MI proxy: E[log q(y|x) - log q(y'|x)]
    mi_hat = (mi_pos_sum - mi_neg_sum) / max(1, total_batches) if total_batches else 0.0

    return avg_wer, avg_ctc_pos, mi_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--input_type", choices=["discrete","continuous"], required=True)
    ap.add_argument("--n_codebooks", type=int, default=0)
    ap.add_argument("--in_dim", type=int, default=None, help="input dim for CONTINUOUS")
    ap.add_argument("--d_emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    
    args.device="cuda" if torch.cuda.is_available() else "cpu"
    if "baseline" in args.data_root:
        args.model_type='dac'
    else:
        args.model_type='discodac'

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    
    # ======= prepare dataset =======
    tr_manifest = Path(args.data_root) / "manifest_train.csv"
    ev_manifest = Path(args.data_root) / "manifest_val.csv"
    
    if not tr_manifest.exists():
        build_manifest(Path(args.data_root), split='train')
    if not ev_manifest.exists():
        build_manifest(Path(args.data_root), split='val')
        
    tr_data = CodecDataset(tr_manifest, input_type=args.input_type)
    ev_data = CodecDataset(ev_manifest, input_type=args.input_type)
    collate = build_collate(args.input_type)
    
    tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    ev_loader = torch.utils.data.DataLoader(ev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Infer dims for model
    example_item = tr_data[0]
    uid0, x0 = example_item["utt_id"], example_item["x"]
    if args.input_type == "discrete":
        codebooks = x0.shape[1] if args.n_codebooks<=0 else args.n_codebooks
        vocab=1024 if args.model_type=='dac' else [512, 1024]
        model = LatentCTCProbe("discrete", codebooks=codebooks, vocab=vocab, d_emb=args.d_emb,
                               hidden=args.hidden, n_layers=args.layers)
    else:
        in_dim = x0.size(1)
        model = LatentCTCProbe("continuous", d_in=in_dim, d_emb=args.d_emb, hidden=args.hidden, n_layers=args.layers)

    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        tr_wer, tr_loss, tr_mi = run_epoch(model, tr_loader, optimizer=opt, device=args.device, codebooks=codebooks, input_type=args.input_type)
        dv_wer, dv_loss, dv_mi = run_epoch(model, ev_loader, optimizer=None, device=args.device, codebooks=codebooks, input_type=args.input_type)

        print(f"[ep {ep:02d}] train WER={tr_wer:.3f} loss={tr_loss:.3f} MI={tr_mi:.3f} | dev WER={dv_wer:.3f} loss={dv_loss:.3f} MI={dv_mi:.3f}")

if __name__ == "__main__":
    main()
