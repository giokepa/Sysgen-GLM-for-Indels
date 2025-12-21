from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
import torch
import numpy as np

@dataclass
class MotifAwareMLMCollator:
    tokenizer: PreTrainedTokenizerBase
    motif_mask_prob: float = 0.6     # high masking on motifs
    bg_mask_prob: float = 0.15       # normal elsewhere

    def __call__(self, examples):
        input_ids = torch.stack([e["input_ids"] for e in examples])
        labels = input_ids.clone()

        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id

        bsz, seqlen = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for i, ex in enumerate(examples):
            posA = ex.get("posA", None)
            posB = ex.get("posB", None)

            # background
            bg_mask = torch.rand(seqlen) < self.bg_mask_prob

            if posA is not None:
                bg_mask[posA:posA+7] = torch.rand(7) < self.motif_mask_prob
            if posB is not None:
                bg_mask[posB:posB+7] = torch.rand(7) < self.motif_mask_prob

            bg_mask[input_ids[i] == pad_id] = False
            mask[i] = bg_mask

        labels[~mask] = -100
        indices = torch.nonzero(mask, as_tuple=False)
        rand = torch.rand(indices.size(0))
        mask80 = rand < 0.8
        rand10 = (rand >= 0.8) & (rand < 0.9)
        keep10 = rand >= 0.9

        input_ids[indices[mask80, 0], indices[mask80, 1]] = mask_id
        vocab_size = self.tokenizer.vocab_size
        random_tokens = torch.randint(0, vocab_size, (rand10.sum(),))
        input_ids[indices[rand10, 0], indices[rand10, 1]] = random_tokens


        return {"input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones_like(input_ids)}
