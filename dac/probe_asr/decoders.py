import numpy as np
import torch
from itertools import groupby
from process_data.asr_data import stoi, itos, tokens
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

def ctc_greedy_decode(outputBatch, inputLenBatch, blank=0):
    """
    Args:
        outputBatch: [B, T, C] log-probabilities from the model
        inputLenBatch: [B] lengths of valid time steps for each sequence
        blank: index of the CTC blank symbol

    Returns:
        predictionBatch: concatenated predictions (torch.int tensor)
        predictionLenBatch: lengths of each prediction (torch.int tensor)
    """

    predCharIxs = torch.argmax(outputBatch, dim=2).cpu().numpy()  #[B, T]
    inpLens =  inputLenBatch.cpu().numpy()

    preds = []
    predLens = []

    for i in range(len(predCharIxs)):
        pred = predCharIxs[i]
        ilen = inpLens[i]
        pred = pred[:ilen]

        # collapse repeats and remove blanks
        pred = np.array([x[0] for x in groupby(pred)])
        pred = pred[pred != blank]

        # store
        preds.extend(pred.tolist())
        predLens.append(len(pred))

    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch


files = download_pretrained_files("librispeech-4-gram")
with open(files.tokens, 'r') as f:
    tokens = [line.strip() for line in f]


def beam_ctc_decode(
    probs, input_lens,
    lm_weight=2.0, word_score=0.0, beam_size=50,
    ):
    """
    CTC Beam Search decoding with optional KenLM language model.

    probs: Tensor [B, T, C] of character probabilities (logits/softmax).
    input_lens: Length of each sequence in the batch (list[int] or tensor).
    """
    
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        nbest=1,
        beam_size=beam_size,
        lm_weight=lm_weight,
        word_score=word_score,
        blank_token=tokens[0],
        sil_token=tokens[1],
    )
    
    beam_results = beam_search_decoder(probs.cpu(), input_lens.cpu())

    beam_indices, beam_lens, beam_transcripts = [], [], []
    for sample in beam_results:   # one sample per batch
        best_hyp = sample[0]      # best hypothesis
        tokens_ids = torch.tensor(best_hyp.tokens, dtype=torch.int)
        beam_indices.append(tokens_ids)
        beam_lens.append(len(tokens_ids))
        beam_transcripts.append(" ".join(best_hyp.words))

    pred_idxs = torch.cat(beam_indices, dim=0)
    pred_lens = torch.tensor(beam_lens)
    return pred_idxs, pred_lens, beam_transcripts


def ctc_greedy_decode_copy(logits, blank=0):
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



