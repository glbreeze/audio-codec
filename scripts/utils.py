import torch
import argparse
from types import SimpleNamespace

def si_snr(est, ref, epsilon=1e-8):
    
    est = est - est.mean(dim=1, keepdim=True)
    ref = ref - ref.mean(dim=1, keepdim=True)
    
    ref_pow = (ref * ref).mean(dim=1, keepdim=True) # [B, 1]
    mix_pow = (est * ref).mean(dim=1, keepdim=True) # [B, 1]
    scale = mix_pow / (ref_pow + epsilon)
    
    scaled_ref = scale * ref # [B, D]
    error = est - scaled_ref # [B, D]
    
    si_snr = 10 * torch.log10((scaled_ref * scaled_ref).mean(dim=1) + epsilon) - \
             10 * torch.log10((error * error).mean(dim=1) + epsilon)
 
    return si_snr.mean().item()