import torch
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
import pandas as pd
from data.augmenter import RuleBasedAugmentor
from data.tokenizer import PhonemeTokenizer
from datasets import load_dataset
from typing import Optional
import random
import numpy as np

class BatchSampler(Sampler):
    def __init__(self, df, batch_size: int = 64, n_singlish_per_batch: int = 2, n_negation_per_batch: int = 2, seed: int = 42, drop_last: bool = True):
        self.n_singlish = n_singlish_per_batch
        self.n_negation = n_negation_per_batch
        self.n_regular = batch_size - n_singlish_per_batch - n_negation_per_batch
        assert self.n_regular >= 0, "Kuota singlish + negation melebihi batch size"

        self.rng = random.Random(seed)
        self._n = len(df)  # dipakai sebagai offset index mode-negasi
 
        positions = np.arange(self._n)
        singlish_mask = (df["unnormalized_text"] != "-").to_numpy()
        negation_mask = (df["negation"] != "-").to_numpy() & ~singlish_mask  # hindari overlap
 
        self.singlish_idx = positions[singlish_mask].tolist()
        self.negation_idx = positions[negation_mask].tolist()
        special_mask = singlish_mask | negation_mask
        self.regular_idx = positions[~special_mask].tolist()
 
        assert len(self.singlish_idx) >= self.n_singlish
        assert len(self.negation_idx) >= self.n_negation
        assert len(self.regular_idx) >= self.n_regular
 
        self.n_batches = self._n // batch_size if drop_last else -(-self._n // batch_size)
 
        self._regular_cycle = self._make_cycle(self.regular_idx)
        self._singlish_cycle = self._make_cycle(self.singlish_idx)
        self._negation_cycle = self._make_cycle(self.negation_idx)
    
    def _make_cycle(self, pool: list):
        pool = list(pool)
        while True:
            shuffled = pool[:]
            self.rng.shuffle(shuffled)
            for i in shuffled:
                yield i
    
    def _draw_unique(self, cycle_gen, k: int, exclude: set) -> list:
        picked = []
        while len(picked) < k:
            i = next(cycle_gen)
            if i not in exclude and i not in picked:
                picked.append(i)
        return picked

    def __iter__(self):
        for _ in range(self.n_batches):
            regular = self._draw_unique(self._regular_cycle, self.n_regular, exclude=set())
            singlish = self._draw_unique(self._singlish_cycle, self.n_singlish, exclude=set(regular))
            negation = self._draw_unique(
                self._negation_cycle, self.n_negation, exclude=set(regular) | set(singlish)
            )
            negation_twins = [i + self._n for i in negation]  # dispatch ke mode-negasi di __getitem__
 
            yield regular + singlish + negation + negation_twins
    
    def __len__(self):
        return self.n_batches

def collate_fn(batch, pad_value=0):
    max_len = max(seq.shape[0] for sample in batch for seq in sample["x"])
 
    batch_x = []
    batch_mask = []
 
    for sample in batch:
        padded = []
        masks = []
        for seq, l in zip(sample["x"], sample["x_lengths"]):
            seq = F.pad(seq, (1, 1), value=pad_value)
            seq = F.pad(seq, (0, max_len - len(seq)), value=pad_value)
            padded.append(seq)
 
            m = torch.ones(l + 2, dtype=torch.bool)
            m = F.pad(m, (0, max_len - len(m)), value=False)
            masks.append(m)
 
        batch_x.append(torch.stack(padded))
        batch_mask.append(torch.stack(masks))
 
    return {
        "id": [b["id"] for b in batch],
        "x": torch.stack(batch_x),
        "texts": [[b["x_canon_text"]] + b["x_aug_1"] + b["x_aug_2"] for b in batch],
        "x_lengths": [[l + 2 for l in b["x_lengths"]] for b in batch],
        "mask": torch.stack(batch_mask),
    }

class AugmentDataset(Dataset):
    def __init__(
        self,
        lexicon_path: str,
        dataset_path: Optional[str] = None,
        mode: str = "huggingface",
        dataframe: Optional[pd.DataFrame] = None,
        tokenizer: Optional[PhonemeTokenizer] = None,
    ):
        if dataframe is not None:
            self.dataset = dataframe.reset_index(drop=True).copy()
        else:
            if mode == "huggingface":
                dataset = load_dataset("avalonai/english-singlish-g2p")
                self.dataset = dataset["train"].to_pandas()
            else:
                self.dataset = pd.read_csv(dataset_path)

            self.dataset = self.dataset.reset_index(drop=True)

        if "phoneme_negation" not in self.dataset.columns:
            self.dataset["phoneme_negation"] = "-"

        self.augmenter = RuleBasedAugmentor(lexicon_path=lexicon_path)

        if tokenizer is None:
            self.phoneme_tokenizer = PhonemeTokenizer.from_corpus(self.dataset["phoneme"].tolist())
        else:
            self.phoneme_tokenizer = tokenizer

        self._n = len(self.dataset)

    def __len__(self):
        return self._n

    def __getitem__(self, idx, n_aug_1=3, n_aug_2=2, canon_mode="phoneme"):
        assert canon_mode in ("phoneme", "text"), (
            "Unknown canon repr. return mode. It has to be either phoneme or text."
        )

        if idx < self._n:
            data = self.dataset.iloc[idx]
            text = data["text"]
            phoneme = data["phoneme"]
            unnormalized_text = data["unnormalized_text"]
            data_id = data["id"]
        else:
            base_idx = idx - self._n
            data = self.dataset.iloc[base_idx]
            assert data["negation"] != "-", (
                f"Baris {base_idx} tidak berlabel negasi, tidak valid diakses lewat idx negasi ({idx})"
            )
            text = data["negation"]
            phoneme = data["phoneme_negation"]
            assert phoneme != "-", (
                f"phoneme_negation masih '-' untuk baris {base_idx}, cek preprocessing dulu"
            )
            unnormalized_text = "-"
            data_id = f"{data['id']}-NEG"

        x_aug_1, x_aug_2 = [], []

        for _ in range(n_aug_1):
            x_aug_1.append(self.augmenter.augment_easy(text))

        if n_aug_2 > 0:
            remaining = n_aug_2
            if unnormalized_text != "-":
                x_aug_2.append(unnormalized_text)
                remaining -= 1
            for _ in range(remaining):
                x_aug_2.append(self.augmenter.augment_hard_surface(text))

        if canon_mode == "phoneme":
            x_canon = self.phoneme_tokenizer.encode_single(phoneme)
        else:
            x_canon = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

        x_v = x_aug_1 + x_aug_2
        x_v = [torch.tensor(list(t.encode("utf-8")), dtype=torch.long) for t in x_v]
        x_lengths = [len(x_i) for x_i in [x_canon] + x_v]

        return {
            "id": data_id,
            "x_canon": phoneme,
            "x_canon_text": text,
            "x_aug_1": x_aug_1,
            "x_aug_2": x_aug_2,
            "x": [x_canon] + x_v,
            "x_lengths": x_lengths,
        }
