import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from data.augmenter import RuleBasedAugmentor
from data.tokenizer import PhonemeTokenizer
from datasets import load_dataset

def collate_fn(batch, pad_value=0):
    max_len = max(
        seq.shape[0]
        for sample in batch
        for seq in sample["x"]
    )

    batch_x = []
    batch_mask = []
    
    for sample in batch:
        padded = []
        masks  = []

        for seq, l in zip(sample["x"], sample["x_lengths"]):
            seq = F.pad(
                seq,
                (0, max_len - len(seq)),
                value = pad_value
            )

            seq = F.pad(
                seq,
                (1, 1),
                value = pad_value
            )

            padded.append(seq)

            m = torch.arange(max_len) < l
            m = F.pad(m, (1, 1), value=False)

            masks.append(m)
        
        batch_x.append(torch.stack(padded))
        batch_mask.append(torch.stack(masks))
    
    return {
        "id": [b["id"] for b in batch],
        "x": torch.stack(batch_x),
        "x_lengths": [
            [l + 2 for l in b["x_lengths"]] for b in batch
        ],
        "mask": torch.stack(batch_mask)
    }

class AugmentDataset(Dataset):
    def __init__(self, lexicon_path: str, dataset_path: str, mode = "huggingface"):
        if mode == "huggingface":
            dataset = load_dataset("avalonai/english-singlish-g2p")
            self.dataset = dataset["train"].to_pandas()
        else:
            self.dataset = pd.read_csv(dataset_path)
        self.augmenter = RuleBasedAugmentor(lexicon_path=lexicon_path)
        self.phoneme_tokenizer = PhonemeTokenizer.from_corpus(self.dataset['phoneme'].tolist())

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx, n_aug_1 = 3, n_aug_2 = 2, canon_mode = "phoneme"):
        assert canon_mode in ("phoneme", "text"), "Unknown canon repr. return mode. It has to be either phoneme or text."
        data = self.dataset.iloc[idx]
        
        data_id = data["id"]
        text = data["text"]
        unnormalized_text = data["unnormalized_text"]
        phoneme = data["phoneme"]

        x_aug_1, x_aug_2 = [], []

        for _ in range(n_aug_1):
            x_aug_1.append(self.augmenter.augment_easy(text))
        
        if n_aug_2 > 0:
            if unnormalized_text != "-":
                x_aug_2.append(unnormalized_text)
                n_aug_2 -= 1
            
            for _ in range(n_aug_2):
                x_aug_2.append(self.augmenter.augment_hard_surface(text))

        # x_canon = phoneme if canon_mode == "phoneme" else text
        if canon_mode == "phoneme":
            x_canon = self.phoneme_tokenizer.encode_single(phoneme)
        else:
            x_canon = torch.tensor(list(text.encode('utf-8')))

        x_v = x_aug_1 + x_aug_2
        x_v = [torch.tensor(list(t.encode('utf-8'))) for t in x_v]
        x_lengths = [len(x_i) for x_i in [x_canon] + x_v]

        return {
            "id": data_id,
            "x_canon": phoneme if canon_mode == "phoneme" else text,
            "x_aug_1": x_aug_1,
            "x_aug_2": x_aug_2,
            "x": [x_canon] + x_v,
            "x_lengths": x_lengths
        }