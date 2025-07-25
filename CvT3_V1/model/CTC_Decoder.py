import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import CTCLoss
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image
import cv2
from pyctcdecode import build_ctcdecoder


class CTCDecoder:
    """CTC Decoder with KenLM language model support"""

    def __init__(self, vocab, lm_path=None, alpha=0.5, beta=1.0):
        self.vocab = vocab
        self.blank_id = 0
        self.alpha = alpha
        self.beta = beta
        self.use_lm = False

    def greedy_decode(self, logits):
        """Simple greedy CTC decoding"""
        # logits: [seq_len, vocab_size]
        predictions = torch.argmax(logits, dim=-1)

        # Remove blanks and consecutive duplicates
        decoded = []
        prev = -1
        for pred in predictions:
            if pred != self.blank_id and pred != prev:
                decoded.append(pred.item())
            prev = pred

        return decoded

    def beam_search_decode(self, logits, beam_width=100):
        """Beam search decoding with optional language model"""
        if self.use_lm:
            # Use pyctcdecode with language model
            logits_np = F.softmax(logits, dim=-1).cpu().numpy()
            text = self.decoder.decode(logits_np, beam_width=beam_width)
            return text
        else:
            # Simple beam search without language model
            return self._simple_beam_search(logits, beam_width)

    def _simple_beam_search(self, logits, beam_width):
        """Simple beam search without language model"""
        seq_len, vocab_size = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)

        # Initialize beam with empty sequence
        beams = [([], 0.0)]  # (sequence, log_prob)

        for t in range(seq_len):
            new_beams = []

            for sequence, log_prob in beams:
                for c in range(vocab_size):
                    new_log_prob = log_prob + log_probs[t, c].item()
                    new_sequence = self._update_sequence(sequence, c)
                    new_beams.append((new_sequence, new_log_prob))

            # Keep top beam_width beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # Return best sequence as text
        best_sequence = beams[0][0]
        return ''.join([self.vocab[i] for i in best_sequence if i < len(self.vocab)])

    def _update_sequence(self, sequence, char_id):
        """Update sequence according to CTC rules"""
        if char_id == self.blank_id:
            # Blank token - don't extend sequence
            return sequence
        elif len(sequence) > 0 and sequence[-1] == char_id:
            # Same character as previous - don't extend
            return sequence
        else:
            # New character
            return sequence + [char_id]

# Training utilities
