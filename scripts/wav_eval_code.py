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

import argparse, math, random, os
from pathlib import Path
import wandb
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from process_data.asr_data import stoi, itos, build_collate, CodecDataset, build_manifest, text_to_ids, ids_to_text
from dac.probe_asr.probe_ctc import LatentCTCProbe
from dac.probe_asr.decoders import ctc_greedy_decode, beam_ctc_decode


def namespace_to_dict(ns):
    if isinstance(ns, (SimpleNamespace, argparse.Namespace)):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns


def edit_distance(ref_tokens, hyp_tokens):
    """Compute Levenshtein distance between two token sequences."""
    dp = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    
    # initialize
    for i in range(len(ref_tokens) + 1):
        dp[i][0] = i
    for j in range(len(hyp_tokens) + 1):
        dp[0][j] = j
    
    # DP fill
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            cost = 0 if ref_tokens[i-1] == hyp_tokens[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,       # deletion
                dp[i][j-1] + 1,       # insertion
                dp[i-1][j-1] + cost   # substitution
            )
    return dp[-1][-1]


def compute_wer_cer(ref, hyp):
    
    if isinstance(ref, list) and isinstance(hyp, list):
        wers, cers = [], []
        for r, h in zip(ref, hyp):
            w, c = compute_wer_cer(r, h)
            wers.append(w); cers.append(c)
        return float(np.mean(wers)), float(np.mean(cers))
    
    # ---- CER ----
    ref_chars = list(ref.strip())
    hyp_chars = list(hyp.strip())
    cer_dist = edit_distance(ref_chars, hyp_chars)
    cer = cer_dist / max(1, len(ref_chars))
    
    # ---- WER ----
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    wer_dist = edit_distance(ref_words, hyp_words)
    wer = wer_dist / max(1, len(ref_words))
    
    return wer, cer

# --------------------------
# Training / Eval
# --------------------------
def run_epoch(model, loader, optimizer=None, device="cpu", neg_permute=True, codebooks=None, input_type=None, dec_scheme='greedy'):
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
    wers, cers = [], []
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
        
        total_pos_loss += float(loss_pos.detach().cpu())
        if neg_permute:
            mi_pos_sum += -float(loss_pos.detach().cpu())
            mi_neg_sum += -float(loss_neg.detach().cpu())
        total_batches += 1

        # Decode greedily for WER
        ref_batch, hyp_batch = [], []
        with torch.no_grad():
            if dec_scheme == 'greedy':
                pred, pred_lens = ctc_greedy_decode(logits_TBC.transpose(0,1), xlens, blank=0)  # [T, B, C] -> list of int ids
            elif dec_scheme == 'search' or dec_scheme == 'beam':
                pred, pred_lens, transcripts = beam_ctc_decode(logits_TBC.transpose(0, 1), xlens, lm_weight=2.0, word_score=0.0, beam_size=50)
                
        ref_chunks = torch.split(ycat.cpu(), ylens.cpu().tolist())      # token_ids
        hyp_chunks = torch.split(pred.cpu(), pred_lens.cpu().tolist())  # token_ids
        for ref_ids, hyp_ids in zip(ref_chunks, hyp_chunks):
            ref_txt = ids_to_text(ref_ids.tolist()).replace('|', ' ')
            hyp_txt = ids_to_text(hyp_ids.tolist()).replace('|', ' ')
            ref_batch.append(ref_txt)
            hyp_batch.append(hyp_txt)
        wer, cer = compute_wer_cer(ref_batch, hyp_batch)
        wers.append(wer)
        cers.append(cer)

    # Aggregate
    avg_wer = float(np.mean(wers))
    avg_cer = float(np.mean(cers))
    avg_ctc_pos = total_pos_loss / max(1, total_batches)

    # MI proxy: E[log q(y|x) - log q(y'|x)]
    mi_hat = (mi_pos_sum - mi_neg_sum) / max(1, total_batches) if total_batches else 0.0

    return avg_wer, avg_cer, avg_ctc_pos, mi_hat


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
    ap.add_argument("--exp_name", type=str, default='baseline')
    args = ap.parse_args()
    
    os.environ["WANDB_MODE"] = "online"
    wandb.init(project='probe_asr', config=namespace_to_dict(args), name=args.exp_name)
    
    args.device="cuda" if torch.cuda.is_available() else "cpu"
    if "baseline" in args.data_root:
        args.model_type='dac'
    else:
        args.model_type='discodac'

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    
    # ======= prepare dataset =======
    print('---------load data--------')
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

   # ======= prepare model =======
    print('---------define model--------')
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
        tr_wer, tr_cer, tr_loss, tr_mi = run_epoch(model, tr_loader, optimizer=opt, device=args.device, codebooks=codebooks, input_type=args.input_type, dec_scheme='greedy')
        wandb.log({
            'train/wer':tr_wer, 
            'train/cer':tr_cer,
            'train/loss':tr_loss,
            'train/mi':tr_mi,
            }, step=ep)
        dv_wer, dv_cer, dv_loss, dv_mi = run_epoch(model, ev_loader, optimizer=None, device=args.device, codebooks=codebooks, input_type=args.input_type, dec_scheme='beam')
        wandb.log({
            'val/wer':dv_wer, 
            'val/cer':dv_cer,
            'val/loss':dv_loss,
            'val/mi':dv_mi,
            }, step=ep)

        print(f"[ep {ep:02d}] train WER={tr_wer:.3f} CER={tr_cer:.3f} loss={tr_loss:.3f} MI={tr_mi:.3f} | dev WER={dv_wer:.3f} CER={dv_cer:.3f} loss={dv_loss:.3f} MI={dv_mi:.3f}")

if __name__ == "__main__":
    main()
