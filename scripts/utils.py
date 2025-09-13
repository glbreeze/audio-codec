import torch
import librosa
import soundfile as sf
import tempfile, os, subprocess, warnings, json
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace

def si_snr(est, ref, epsilon=1e-8, reduction=True):
    # est [B, T]
    
    est = est - est.mean(dim=1, keepdim=True)
    ref = ref - ref.mean(dim=1, keepdim=True)
    
    ref_pow = (ref * ref).mean(dim=1, keepdim=True) # [B, 1]
    mix_pow = (est * ref).mean(dim=1, keepdim=True) # [B, 1]
    scale = mix_pow / (ref_pow + epsilon)
    
    scaled_ref = scale * ref # [B, D]
    error = est - scaled_ref # [B, D]
    
    si_snr = 10 * torch.log10((scaled_ref * scaled_ref).mean(dim=1) + epsilon) - \
             10 * torch.log10((error * error).mean(dim=1) + epsilon)
    
    if reduction:
        return si_snr.mean().item()
    else:
        return si_snr.cpu().numpy()


def pad_or_trim_to_match(ref, deg):
    n = min(len(ref), len(deg))
    if n == 0:
        return ref[:0], deg[:0]
    return ref[:n], deg[:n]


# ---------- ViSQOL (optional) ----------
def compute_visqol(ref_np, deg_np, sr, mode="speech", model_path=None):
    """
    Returns MOS-LQO or np.nan on failure.
    Uses Python bindings if available; otherwise tries CLI 'visqol' if present.
    """
    # Ensure same length
    ref_np, deg_np = pad_or_trim_to_match(ref_np, deg_np)
    if len(ref_np) == 0:
        return float("nan")

    target_sr = 16000 if mode == "speech" else 48000
    if sr != target_sr:
        ref_np = librosa.resample(ref_np, orig_sr=sr, target_sr=target_sr)
        deg_np = librosa.resample(deg_np, orig_sr=sr, target_sr=target_sr)

    # Try python bindings
    try:
        from visqol import visqol_lib_py, visqol_config_pb2
        cfg = visqol_config_pb2.VisqolConfig()
        # handle field name differences across versions
        if hasattr(cfg, "use_speech_mode"):
            cfg.use_speech_mode = (mode == "speech")
        if hasattr(cfg, "audio_sample_rate"):
            cfg.audio_sample_rate = target_sr
        elif hasattr(cfg, "audio") and hasattr(cfg.audio, "sample_rate"):
            cfg.audio.sample_rate = target_sr
        if model_path is not None:
            # model/svr file path field varies; set if present
            for k in ["svr_model_path", "similarity_to_quality_model"]:
                if hasattr(cfg, k):
                    setattr(cfg, k, model_path)

        visq = visqol_lib_py.Visqol()
        visq.Create(cfg)

        # ViSQOL Python prefers file paths; write temp wavs.
        with tempfile.TemporaryDirectory() as td:
            ref_wav = os.path.join(td, "ref.wav")
            deg_wav = os.path.join(td, "deg.wav")
            sf.write(ref_wav, ref_np, target_sr)
            sf.write(deg_wav, deg_np, target_sr)
            res = visq.Measure(ref_wav, deg_wav)
            # different bindings expose .moslqo or .moslqo_score
            score = getattr(res, "moslqo", getattr(res, "moslqo_score", None))
            return float(score) if score is not None else float("nan")
    except Exception as e:
        warnings.warn(f"ViSQOL python binding not available/failed: {e}")

    # Fallback: CLI
    try:
        with tempfile.TemporaryDirectory() as td:
            ref_wav = os.path.join(td, "ref.wav")
            deg_wav = os.path.join(td, "deg.wav")
            out_json = os.path.join(td, "out.json")
            sf.write(ref_wav, ref_np, target_sr)
            sf.write(deg_wav, deg_np, target_sr)
            cmd = ["visqol", "--reference_file", ref_wav, "--degraded_file", deg_wav]
            if mode == "speech":
                cmd += ["--use_speech_mode", "true"]
            if model_path is not None:
                cmd += ["--similarity_to_quality_model", model_path]
            cmd += ["--output_json", out_json]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(out_json, "r") as f:
                j = json.load(f)
            # most CLI builds output {"aggregate": {"moslqo": ...}, ...}
            if "aggregate" in j and "moslqo" in j["aggregate"]:
                return float(j["aggregate"]["moslqo"])
    except Exception as e:
        warnings.warn(f"ViSQOL CLI not available/failed: {e}")

    return float("nan")



def save_item_npz(
    out_dir: Path, split: str, utt_id: str, text: str,
    code: np.ndarray,    # [M, T] int
    latent: np.ndarray,  # [D, T] float32
    ):
    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{utt_id}.npz"

    kw = {"utt_id": utt_id, "text": text}
    
    if code.dtype.kind != "i":
        raise ValueError("code must be integer array")
    if latent.dtype != np.float32:
        latent = latent.astype(np.float32, copy=False)
    kw["code"] = code.astype(np.int32, copy=False)
    kw["latent"] = latent

    np.savez_compressed(path, **kw)
    return path