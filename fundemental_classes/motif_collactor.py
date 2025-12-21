from transformers import DataCollatorForLanguageModeling
import torch


class MotifAwareCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, motif_prob=0.8, bg_prob=0.10):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        self.motif_prob = motif_prob
        self.bg_prob = bg_prob

    def __call__(self, examples):
        # We need to manually construct the mask based on "posAmotif" info
        # But 'examples' here is a list of dicts from the dataset.
        # We need the Dataset to give us the positions.

        # 1. Standard collate to get batch tensors
        batch = super().__call__(examples)

        # 2. Override the mask!
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # We need to re-parse positions from the raw example if passed,
        # OR better: Update DNADataset to return 'pos_a' and 'pos_b' in the dict.
        # Assuming DNADataset returns 'pos_a' and 'pos_b' tensors/ints:

        # Create a new mask
        new_mask = torch.full_like(input_ids, 0, dtype=torch.bool)

        seq_len = input_ids.shape[1]

        for i, ex in enumerate(examples):
            # Default background masking
            bg_mask = torch.rand(seq_len) < self.bg_prob

            # Motif masking (higher probability)
            # You need to modify DNADataset to return these fields!
            pos_a = ex.get("pos_a", -1)
            pos_b = ex.get("pos_b", -1)

            # Map sequence position to token position (account for CLS offset)
            if pos_a != -1:
                # Assuming motif length 7
                start = pos_a + 1
                end = start + 7
                if end < seq_len:
                    bg_mask[start:end] = torch.rand(7) < self.motif_prob

            if pos_b != -1:
                start = pos_b + 1
                end = start + 7
                if end < seq_len:
                    bg_mask[start:end] = torch.rand(7) < self.motif_prob

            # Protect special tokens
            bg_mask[0] = False  # CLS
            bg_mask[-1] = False  # SEP
            bg_mask[input_ids[i] == self.tokenizer.pad_token_id] = False

            new_mask[i] = bg_mask

        # 3. Apply the 80/10/10 logic to the NEW mask
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            inputs=batch["input_ids"].clone(),
            special_tokens_mask=~new_mask  # hacky way to pass our mask?
            # Actually torch_mask_tokens computes the mask internally.
            # We should just manually do it or rely on the probability matrix method.
        )

        # EASIER WAY: Just set the labels manually based on new_mask
        # reset labels
        labels = input_ids.clone()
        labels[~new_mask] = -100  # Ignore unmasked

        # Apply 80% MASK, 10% Random, 10% Original logic
        probability_matrix = torch.full(labels.shape, 0.15)  # dummy
        masked_indices = new_mask  # Use our custom mask

        # 80% of masked -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of masked -> Random
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest 10% stays original

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch

