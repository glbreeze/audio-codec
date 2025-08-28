
import argparse, os, sys, json, math, glob, yaml, re, tempfile, shutil, warnings, subprocess, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

import librosa
import soundfile as sf
from audiotools import AudioSignal
from audiotools.data import transforms

import whisper
from pesq import pesq
from pystoi import stoi
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
from speechbrain.pretrained import EncoderClassifier

import dac
from utils import save_item_npz, si_snr

# ================= utilities ================= 
def load_audio(path, sr=16000, postprocess=["VolumeNorm", "RescaleAudio"]):
    sig = AudioSignal(path)
    sig = sig.to_mono()
    if sig.sample_rate != sr:
        sig = sig.resample(sample_rate=sr)
    
    # ------ get transform  ------ 
    to_tfm = lambda l: [getattr(transforms, x)() for x in l]
    transform = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform_args = transform.instantiate(signal=sig)
    sig = transform(sig, **transform_args)
    
    return sig


def transcribe_whisper(model, wav_1d, language="en"):
    # wav_1d: numpy [T] @16k
    if isinstance(wav_1d, torch.Tensor):
        wav_1d = wav_1d.detach().cpu().float().numpy()
    result = model.transcribe(wav_1d, fp16=torch.cuda.is_available(), language=language)
    return result["text"].strip()


def get_speaker_embedding(encoder, wav_1d):
    with torch.no_grad():
        emb = encoder.encode_batch(wav_1d.unsqueeze(0))  # [1, D]
        emb = F.normalize(emb.squeeze(0), dim=-1)  # [1, D]
    return emb.squeeze(0).detach().cpu().numpy()  # [D]


def compute_eer(target_scores, nontarget_scores):
    scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([np.ones_like(target_scores), np.zeros_like(nontarget_scores)])
    order = np.argsort(scores)
    scores, labels = scores[order], labels[order]
    P = labels.sum(); N = len(labels) - P
    fa, fr, eer = N, 0, 1.0
    for i in range(len(scores)):
        if labels[i] == 1: fr += 1
        else: fa -= 1
        far = fa / max(N, 1); frr = fr / max(P, 1)
        eer = min(eer, max(far, frr))
    return float(eer)


def load_refs_csv(path):
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if "utt_id" not in cols:
        raise ValueError("refs.csv must contain column: utt_id")
    return df.set_index("utt_id")


# =========================
# Model loading
# =========================
def load_yaml_cfg(cfg_yml: Path):
    with open(cfg_yml, "r") as f:
        return yaml.safe_load(f)


def build_codec_from_yaml(cfg: dict, ckpt_path: Path, device: str = "cpu", model_type: str ='DAC'):
    model_cfg = {k.split('.')[1]: v for k, v in cfg.items() if k.startswith(model_type)}
    
    model_folder = ckpt_path.parts[-2]
    match = re.search(r"cb(\d+)", model_folder)
    codebooks = int(match.group(1))
    if model_type == 'DiscoDAC':
        match = re.search(r"film(\d+)", model_folder)
        film_layers = match.group(1)
    
    if model_type.lower() == 'discodac':
        model = dac.model.DiscoDAC(
            sample_rate=model_cfg.get("sample_rate", 16000),
            encoder_dim=model_cfg.get("encoder_dim", 64),
            encoder_rates=model_cfg.get("encoder_rates", []),
            decoder_dim=model_cfg.get("decoder_dim", 1536),
            decoder_rates=model_cfg.get("decoder_rates", []),
            codebook_size=model_cfg.get("codebook_size", 1024),
            codebook_dim=model_cfg.get("codebook_dim", 8),
            n_codebooks=codebooks,
            film_layer_idx=film_layers,
        )
    elif model_type.lower() == 'dac': 
        model = dac.model.DAC(
            sample_rate=model_cfg.get("sample_rate", 16000),
            encoder_dim=model_cfg.get("encoder_dim", 64),
            encoder_rates=model_cfg.get("encoder_rates", []),
            decoder_dim=model_cfg.get("decoder_dim", 1536),
            decoder_rates=model_cfg.get("decoder_rates", []),
            codebook_size=model_cfg.get("codebook_size", 1024),
            codebook_dim=model_cfg.get("codebook_dim", 8),
            n_codebooks=codebooks,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported types: 'DAC', 'DiscoDAC'")
        
    state = torch.load(os.path.join(ckpt_path, model_type.lower(), 'weights.pth'), map_location=device)
    # state_dict = state.get("model", state)  # support both raw and wrapped
    state_dict = state['state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org_dir", required=True, type=str)
    ap.add_argument("--pattern", type=str, default="**/*.flac")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--cb", type=int, default=3, help="Number of codebooks to use during decoding (if supported).")

    # NEW: model loading / generation options
    ap.add_argument("--cfg_yml", type=str, default=None, help="YAML config to build the codec (e.g., DiscoDAC).")
    ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for the pretrained codec.")
    
    ap.add_argument("--whisper_model", type=str, default="medium.en", help="e.g., tiny.en, base.en, small.en, medium.en")
    
    ap.add_argument("--subset", type=float, default=1.0)
    ap.add_argument("--eval_flag", action='store_true', default=False)

    args = ap.parse_args()
    if 'train' in args.org_dir:
        args.split='train'; args.out_csv='train_results.csv'
    else:
        args.split='val'; args.out_csv='val_results.csv'
    args.model_type = 'DAC' if 'baseline' in args.ckpt else 'DiscoDAC'

    org_dir = Path(args.org_dir)
    assert org_dir.exists(), "org_dir must exist."
    org_files = {Path(p).stem: Path(p) for p in glob.glob(str(org_dir / args.pattern), recursive=True)}
    
    # ================== load audio meta info. utt_id -> spk_id, text ================== 
    utt_ids = sorted(org_files.keys()) 
    rows = []
    id2text = {}
    for utt_id in utt_ids:
        spk_id = int(utt_id.split('-')[0])
        chp_id = int(utt_id.split('-')[1])
        trans_path = org_files[utt_id].parent / f'{spk_id}-{chp_id}.trans.txt'
        
        if utt_id not in id2text:
            with open(trans_path, "r", encoding="utf-8") as f:
                _id2text = {id_: text for id_, text in (line.strip().split(" ", 1) for line in f)}
                id2text.update(_id2text)
        
        text = id2text[utt_id]
        
        rows.append({'utt_id': utt_id, 'spk_id': spk_id, 'text': text})
    refs_df = pd.DataFrame(rows).drop_duplicates(subset=["utt_id"]).set_index("utt_id")
        
    # ================== load ASR and Speaker Model  ================== 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper ASR: {args.whisper_model} on {device}")
    asr_model = whisper.load_model(args.whisper_model, device=device)

    print("Loading speaker embedding model: speechbrain/spkrec-ecapa-voxceleb")
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir="pretrained_ecapa"
    )
    
    # ================== load pretrained codec model  ================== 
    cfg = load_yaml_cfg(Path(args.cfg_yml))
    model = build_codec_from_yaml(cfg, Path(args.ckpt), device=device, model_type=args.model_type)

    rows = []
    org_texts = {}  # whisper model output based on org wav
    dec_texts = {}  # whisper model output based on dec wav
    spk_emb_ref = {}
    spk_emb_dec = {}
    spk_map = {}  # utt_id -> speaker (if available)
    
    # Normalization for WER/CER
    normalize = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    
    if args.subset < 1:
        idx_path = f'./subset_idx_{int(args.subset*100)}.txt'
        if not os.path.exists(idx_path):
            utt_ids_all = sorted(org_files.keys())
            random.seed(42)
            subset_ids = random.sample(utt_ids_all, int(len(utt_ids_all)*args.subset))
            
            with open(idx_path, "w") as f:
                for uid in subset_ids:
                    f.write(uid + "\n")
            utt_ids = subset_ids
        else:
            with open(idx_path, "r") as f:
                subset_ids = [line.strip() for line in f]
            utt_ids = [uid for uid in sorted(org_files.keys()) if uid in subset_ids]
    else:
        utt_ids = sorted(org_files.keys())

    for idx, uid in enumerate(utt_ids):
        file_path = org_files[uid]
        org_signal = load_audio(file_path, sr=args.sr)
        gt_txt = str(refs_df.loc[uid, "text"])
        
        if idx % 1000 == 0:
            print(f'---processing {idx}th sample')
        
        # ======== codec forward =========
        with torch.no_grad():
            org_signal.audio_data = org_signal.audio_data.to(device)
            out = model(org_signal.audio_data, org_signal.sample_rate, n_quantizers=args.cb)  # audio_data: [B, C=1, T]
            dec_signal = AudioSignal(out["audio"], org_signal.sample_rate)
            
        if args.model_type.lower() == 'dac': 
            codes = out["codes"]      # [B, K, T]
            latents = out["latents"]  # [B, 8, T]
        elif args.model_type.lower() == 'discodac':
            codes = torch.cat([out["codes_sem"].unsqueeze(1), out["codes_acs"]], dim=1)  # [B, K, T]
            latents = torch.cat([out['latents_sem'], out['latents_acs'][:, :model.codebook_dim,:]], dim=1) # [B, 16, T]
        
        # === store latent code ===
        save_item_npz(
            out_dir=Path(args.ckpt).parent/'asr_data', split=args.split, utt_id=uid, text=gt_txt,
            code=codes[0].cpu().numpy(),
            latent=latents[0].cpu().numpy(),
            )
        
        if args.eval_flag:
            # ======== get wer and cer  ======== 
            if uid not in org_texts:
                org_texts[uid] = transcribe_whisper(asr_model, org_signal.audio_data.squeeze())
            if uid not in dec_texts:
                dec_texts[uid] = transcribe_whisper(asr_model, dec_signal.audio_data.squeeze())

            spk_id = refs_df.at[uid, "spk_id"] if "spk_id" in refs_df.columns else None
            wer_ref = jiwer_wer(normalize(gt_txt), normalize(org_texts[uid]))
            cer_ref = jiwer_cer(normalize(gt_txt), normalize(org_texts[uid]))
            wer_dec = jiwer_wer(normalize(gt_txt), normalize(dec_texts[uid]))
            cer_dec = jiwer_cer(normalize(gt_txt), normalize(dec_texts[uid]))
            delta_wer = wer_dec - wer_ref
            delta_cer = cer_dec - cer_ref
            
            #  ======== PESQ/STOI  ======== 
            pesq_score = pesq(args.sr, ref=org_signal.audio_data.squeeze().cpu().numpy(), deg=dec_signal.audio_data.squeeze().cpu().numpy(), mode='wb')
            stoi_score = stoi(org_signal.audio_data.squeeze().cpu().numpy(), dec_signal.audio_data.squeeze().cpu().numpy(), args.sr, extended=False)
            snr_score = si_snr(dec_signal.audio_data.squeeze(1).cpu(), org_signal.audio_data.squeeze(1).cpu(), reduction=True)
            
            #  ========  Speaker embeddings  ======== 
            spk_id = refs_df.loc[uid, "spk_id"] if (uid in refs_df.index) else None
            emb_ref = get_speaker_embedding(spk_model, org_signal.audio_data.squeeze())
            emb_dec = get_speaker_embedding(spk_model, dec_signal.audio_data.squeeze())
            spk_cos = float(np.dot(emb_ref, emb_dec)) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_dec) + 1e-8)
            
            spk_emb_ref[uid] = emb_ref
            spk_emb_dec[uid] = emb_dec
            spk_map[uid] = spk_id

            rows.append({
                "utt_id": uid,
                "speaker": spk_id,
                "WER_ref": wer_ref,
                "CER_ref": cer_ref,
                "WER_decoded": wer_dec,
                "CER_decoded": cer_dec,
                "Delta_WER": delta_wer,
                "Delta_CER": delta_cer,
                "PESQ_wb": pesq_score,
                "STOI": stoi_score,
                "SNR": snr_score,
                "SpkCos": spk_cos,
                "gt_text": (refs_df.loc[uid, "text"] if (uid in refs_df.index) else None),
                "text_ref": org_texts[uid], 
                "text_deg": dec_texts[uid],
            })

    if args.eval_flag:
        df = pd.DataFrame(rows)
        df.to_csv(Path(args.ckpt).parent / args.out_csv, index=False)
        print(f"\nPer-utterance metrics saved to: {args.out_csv}")

        # ========== Aggregate summary  ========== 
        def mean_ignore_nan(x):
            x = [v for v in x if not (isinstance(v, float) and math.isnan(v))]
            return float(np.mean(x)) if len(x) else np.nan

        summary = {
            "N": len(df),
            "WER_ref@GT": mean_ignore_nan(df["WER_ref"].tolist()),
            "WER_decoded@GT": mean_ignore_nan(df["WER_decoded"].tolist()),
            "Delta_WER": mean_ignore_nan(df["Delta_WER"].tolist()),
            "CER_ref@GT": mean_ignore_nan(df["CER_ref"].tolist()),
            "CER_decoded@GT": mean_ignore_nan(df["CER_decoded"].tolist()),
            "Delta_CER": mean_ignore_nan(df["Delta_CER"].tolist()),
            "PESQ_wb": mean_ignore_nan(df["PESQ_wb"].tolist()),
            "STOI": mean_ignore_nan(df["STOI"].tolist()),
            "SpkCos": mean_ignore_nan(df["SpkCos"].tolist()),
            "SNR": mean_ignore_nan(df["SNR"].tolist())
        }

        #  ==========  EER  ========== 
        uids = [u for u, s in spk_map.items() if s is not None]
        
        same_scores = np.array([np.dot(spk_emb_ref[u], spk_emb_dec[u]) for u in uids], dtype=float)
        
        rng = np.random.default_rng(0)
        diff_scores = []
        for u in uids:
            other_uids = [v for v in uids if spk_map[v] != spk_map[u]]
            if not other_uids:
                continue
            neg = rng.choice(other_uids)
            diff_scores.append(np.dot(spk_emb_ref[u], spk_emb_dec[neg]))
            
        diff_scores = np.asarray(diff_scores, dtype=float)
        
        if len(diff_scores) >= 10: 
            summary["eer"] = compute_eer(same_scores, diff_scores)

        print("\n=== Summary ===")
        for k, v in summary.items():
            print(f"{k:>18s}: {v:.4f}" if isinstance(v, (float, np.floating)) and not math.isnan(v) else f"{k:>18s}: {v}")

        summary_path = (Path(args.ckpt).parent / args.out_csv).with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

