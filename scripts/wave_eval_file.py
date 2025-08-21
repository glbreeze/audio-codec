
import argparse, os, sys, json, math, glob, yaml, re, argbind
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

import librosa
import soundfile as sf
from audiotools import ml
from audiotools.core import util
from audiotools import AudioSignal
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset

from utils import si_snr, save_item_npz

import whisper
from pesq import pesq
from pystoi import stoi
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
from speechbrain.pretrained import EncoderClassifier
import dac

Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)

DAC = argbind.bind(dac.model.DAC)
DiscoDAC = argbind.bind(dac.model.DiscoDAC)

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


def pad_or_trim_to_match(ref, deg):
    n = min(len(ref), len(deg))
    if n == 0:
        return ref[:0], deg[:0]
    return ref[:n], deg[:n]

# ================= get data using Audiotools ================= 
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in ["BaseTransform", "Compose", "Choose"]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform



@argbind.bind("train", "val", "test")
def build_dataset(sample_rate: int, folders: dict = None):
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        dataset.length = len(dataset.loaders[0].audio_indices)
        datasets.append(dataset)
    
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset

# ================= utility models ================= 

def transcribe_batch(model, batch_audio_16k):  # torch.Tensor [B, T] or list of 1-D np arrays
    texts = []
    for i in range(len(batch_audio_16k)):
        x = batch_audio_16k[i]
        x = x.detach().cpu().numpy() if hasattr(x, "detach") else x
        out = model.transcribe(x.astype("float32"), language="en", fp16=True)
        texts.append(out["text"].strip())
    return texts


def get_speaker_embedding(encoder, wav_1d, device="cpu"):
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
    match = re.search(r"libri(\d+)", model_folder)
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


def load(
    args,
    accel: ml.Accelerator,
    save_path: str,
    load_weights: bool = False,
    tag: str = "latest",
    split: str = 'val',
):
    # ======= load model     
    if 'baseline' in save_path:
        generator = DAC()
        model_type = 'dac'
    else:
        generator = DiscoDAC()
        model_type = 'discodac'
    
    model_path = Path('.')/save_path/tag/model_type/'weights.pth'
    print(f"Resuming from model from {model_path}")
    model_dict = torch.load(model_path, "cpu")
    generator.load_state_dict(model_dict["state_dict"], strict=True)
        
    print(generator)
    generator = accel.prepare_model(generator)
     
    # ====== load dataset 
    sample_rate = accel.unwrap(generator).sample_rate
    if split == 'val':
        with argbind.scope(args, "val"):
            data = build_dataset(sample_rate)
    elif split == 'train':
        with argbind.scope(args, "train"):
            data = build_dataset(sample_rate)

    return generator, data, model_type
    

@argbind.bind(without_prefix=True)
def main(
    args,
    accel: ml.Accelerator,
    save_path: str = "ckpt",
    tag: str = "latest",
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 4,
    split: str = 'val',
    ):
    
    
    # ================== load model and data loader  ================== 
    generator, data, model_type = load(args, accel, save_path, load_weights=True, tag=tag, split=split)
    
    dataloader = accel.prepare_dataloader(
        data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )
    generator.eval()

    # ================== load ASR and Speaker Model  ================== 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper ASR: medium.en on {device}")
    asr_model = whisper.load_model("medium.en", device=device)
    normalize = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

    print("Loading speaker embedding model: speechbrain/spkrec-ecapa-voxceleb")
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir="pretrained_ecapa"
    )
    
    # ================== evaluate the model  ================== 

    rows = {}         # utt_id -> metric dict
    spk_emb_ref = {}  # utt_id -> [D]
    spk_emb_dec = {}  # utt_id -> [D]
    spk_map = {}  # utt_id -> speaker
    txt_map = {}  # utt_id -> text
    
    for batch in tqdm(dataloader, desc="Validating"):
        batch = util.prepare_batch(batch, accel.device)
        signal = data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
        with torch.no_grad():
            out = generator(signal.audio_data, signal.sample_rate)
            recons = AudioSignal(out["audio"], signal.sample_rate)
        
        if model_type == 'dac':
            codes = out["codes"]      # [B, K, T]
            latents = out["latents"]  # [B, 8, T]
        elif model_type == 'discodac':
            codes = torch.cat([out["codes_sem"].unsqueeze(1), out["codes_acs"]], dim=1)  # [B, K, T]
            latents = torch.cat([ out['latents_sem'], out['latents_acs'][:, :generator.codebook_dim,:]], dim=1) # [B, 16, T]
        
        snr_scores = si_snr(recons.audio_data.squeeze(1), signal.audio_data.squeeze(1), reduction=False)
        

        ref_texts = transcribe_batch(asr_model, signal.audio_data.squeeze(1))
        dec_texts = transcribe_batch(asr_model, recons.audio_data.squeeze(1))
        
        _spk_emb_ref = spk_model.encode_batch(signal.audio_data.squeeze(1))
        _spk_emb_dec = spk_model.encode_batch(recons.audio_data.squeeze(1))
        _spk_emb_ref = F.normalize(_spk_emb_ref, dim=-1).cpu().numpy()
        _spk_emb_dec = F.normalize(_spk_emb_dec, dim=-1).cpu().numpy()
    
        # ======== load audio meta info  ======= 
        
        for idx, path in enumerate(batch["path"]):
            file_name = Path(path).name
            spk_id, chp_id = file_name.split('-')[:2]
            utt_id = file_name.split('.')[0]
     
            if utt_id not in txt_map:
                trans_path = Path(path).parent / f'{spk_id}-{chp_id}.trans.txt'
                with open(trans_path, "r", encoding="utf-8") as f:
                    id2text = {id_: text for id_, text in (line.strip().split(" ", 1) for line in f)}
                    txt_map.update(id2text)
            gt_txt = txt_map[utt_id]

            wer_ref = jiwer_wer(normalize(gt_txt), normalize(ref_texts[idx]))
            cer_ref = jiwer_cer(normalize(gt_txt), normalize(ref_texts[idx]))
            wer_dec = jiwer_wer(normalize(gt_txt), normalize(dec_texts[idx]))
            cer_dec = jiwer_cer(normalize(gt_txt), normalize(dec_texts[idx]))
            
            ref_np = signal.audio_data[idx].squeeze().detach().cpu().numpy()
            dec_np = recons.audio_data[idx].squeeze().detach().cpu().numpy()
            pesq_score = pesq(signal.sample_rate, ref_np, dec_np, mode="wb")
            stoi_score = stoi(ref_np, dec_np, signal.sample_rate, extended=False)
            
            spk_emb_ref[utt_id] = _spk_emb_ref[idx].squeeze()
            spk_emb_dec[utt_id] = _spk_emb_dec[idx].squeeze()
            spk_map[utt_id] = spk_id
            spk_cos = float(np.dot(spk_emb_ref[utt_id], spk_emb_dec[utt_id])) 
            
            # === store latent code ===
            save_item_npz(
                out_dir=Path(save_path)/'ld', split=split, utt_id=utt_id, text=gt_txt,
                code=codes[idx].cpu().numpy(),
                latent=latents[idx].cpu().numpy(),
                )
            
            # === update the overall dict  === 
            rows[utt_id] = {
                "speaker": spk_id, 
                "WER_ref": wer_ref,
                "CER_ref": cer_ref,
                "WER_decoded": wer_dec,
                "CER_decoded": cer_dec,
                "Delta_WER": wer_dec - wer_ref,
                "Delta_CER": cer_dec - cer_ref,
                "pesq": pesq_score,
                "stoi": stoi_score,
                "snr": snr_scores[idx].item(),
                "SpkCos": spk_cos,
                "gt_text": gt_txt,
                "text_ref": ref_texts[idx], 
                "text_deg": dec_texts[idx],
            }

    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="utt_id")
    df.to_csv(Path(save_path) / 'result.csv', index=False)
    print(f"\nPer-utterance metrics saved to: {save_path}")

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
        "PESQ_wb": mean_ignore_nan(df["pesq"].tolist()),
        "STOI": mean_ignore_nan(df["stoi"].tolist()),
        "SNR": mean_ignore_nan(df["snr"].tolist()),
        "SpkCos": mean_ignore_nan(df["SpkCos"].tolist()),
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

    summary_path = (Path(save_path) / 'result.csv').with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        with Accelerator() as accel:
            main(args, accel)
  

