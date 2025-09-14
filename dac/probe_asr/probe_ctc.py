
import torch
import torch.nn as nn
import torch.nn.functional as F
from process_data.asr_data import tokens


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
        return torch.stack(embs, dim=0).mean(0)  # [B,T,d]

class LatentCTCProbe(nn.Module):
    def __init__(self, mode, d_in=None, codebooks=None, vocab=None, d_emb=128, hidden=256, n_layers=2, n_chars=len(tokens), dropout=0):
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
        
        self.proj = nn.Sequential(
            nn.Linear(d_front, 2*d_front),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
            
        self.rnn = nn.LSTM(
            input_size = 2*d_front, 
            hidden_size = hidden, 
            num_layers=n_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout
        )
        
        self.out = nn.Linear(hidden*2, n_chars)  # include blank at idx 0
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, x_lens):
        # x: [B,T,M] (long) for discrete OR [B,T,D] float for continuous
        if self.mode == "discrete" and x.dtype != torch.long:
            x = x.long()
            
        h = self.frontend(x)   # [B,T,d]
        h = self.proj(h)       # [B,T,2d]              
        
        h = nn.utils.rnn.pack_padded_sequence(h, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(h)                  # [B,T,2H]
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        
        logits = self.log_softmax(self.out(h))  # [B,T,C]
        logits = logits.transpose(0,1)          # [T,B,C] for CTC
        return logits