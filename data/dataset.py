import torch
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
from data.augmenter import RuleBasedAugmentor
from data.tokenizer import PhonemeTokenizer
from datasets import load_dataset, Dataset as HFDataset
from typing import Optional
import random
import numpy as np


_REQUIRED_COLS = ["id", "text", "phoneme", "unnormalized_text", "negation"]

# Baris dengan `text` >= ambang ini akan dibuang, KECUALI id-nya berawalan
# _EXEMPT_ID_PREFIX (mis. korpus Singlish "NUS-*" yang ingin selalu dipertahankan
# berapa pun panjangnya).
_MAX_TEXT_CHARS = 512
_EXEMPT_ID_PREFIX = "NUS"


def _length_filter_batch(ids, texts, max_text_chars: int, exempt_id_prefix: str):
    keep = []
    for i, t in zip(ids, texts):
        if exempt_id_prefix and str(i).startswith(exempt_id_prefix):
            keep.append(True)
        else:
            keep.append(len(str(t)) < max_text_chars)
    return keep


def _load_and_clean(
    mode: str,
    dataset_path: Optional[str],
    max_text_chars: int = _MAX_TEXT_CHARS,
    exempt_id_prefix: str = _EXEMPT_ID_PREFIX,
) -> HFDataset:
    if mode == "huggingface":
        raw = load_dataset("avalonai/english-singlish-g2p")["train"]
    else:
        raw = load_dataset("csv", data_files=dataset_path)["train"]

    existing_required_cols = [c for c in _REQUIRED_COLS if c in raw.column_names]

    if existing_required_cols:
        raw = raw.filter(
            lambda ex: all(ex[c] is not None for c in existing_required_cols),
            desc="Membuang baris dengan kolom wajib null",
        )

    sentinel_cols = [c for c in ["unnormalized_text", "negation"] if c in raw.column_names]
    if sentinel_cols:
        def _fill_sentinel(example):
            for col in sentinel_cols:
                v = example[col]
                if v is None or str(v).strip() == "":
                    example[col] = "-"
            return example
        raw = raw.map(_fill_sentinel, desc="Normalisasi sentinel '-'")

    raw = raw.filter(
        lambda ex: len(str(ex["text"]).split()) > 1,
        desc="Membuang kalimat <=1 kata",
    )

    if max_text_chars is not None and "id" in raw.column_names and "text" in raw.column_names:
        raw = raw.filter(
            lambda batch: _length_filter_batch(
                batch["id"], batch["text"], max_text_chars, exempt_id_prefix
            ),
            batched=True,
            desc=(
                f"Membuang teks >={max_text_chars} char "
                f"(kecuali id berawalan '{exempt_id_prefix}')"
            ),
        )

    return raw


class BatchSampler(Sampler):
    def __init__(self, dataset: HFDataset, batch_size: int = 64, n_singlish_per_batch: int = 2,
                 n_negation_per_batch: int = 2, seed: int = 42, drop_last: bool = True):
        self.n_singlish = n_singlish_per_batch
        self.n_negation = n_negation_per_batch
        self.n_regular = batch_size - n_singlish_per_batch - n_negation_per_batch
        assert self.n_regular >= 0, "Kuota singlish + negation melebihi batch size"

        self.rng = random.Random(seed)
        self._n = len(dataset)  # dipakai sebagai offset index mode-negasi

        positions = np.arange(self._n)
        unnorm = np.array(dataset["unnormalized_text"], dtype=object)
        neg = np.array(dataset["negation"], dtype=object)
        singlish_mask = unnorm != "-"
        negation_mask = (neg != "-") & ~singlish_mask  # hindari overlap

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


def collate_fn(batch, pad_value=0, max_seq_len: int = 4096):
    max_seq_len = max_seq_len - 2
    natural_max = max(seq.shape[0] for sample in batch for seq in sample["x"])
    max_len = min(natural_max, max_seq_len) + 2

    batch_x = []
    batch_mask = []

    for sample in batch:
        padded = []
        masks = []
        for seq, l in zip(sample["x"], sample["x_lengths"]):
            l_capped = min(l, max_seq_len)
            seq = seq[:l_capped]
            seq = F.pad(seq, (1, 1), value=pad_value)
            seq = F.pad(seq, (0, max_len - len(seq)), value=pad_value)
            padded.append(seq)

            m = torch.ones(l_capped + 2, dtype=torch.bool)
            m = F.pad(m, (0, max_len - len(m)), value=False)
            masks.append(m)

        batch_x.append(torch.stack(padded))
        batch_mask.append(torch.stack(masks))

    return {
        "id": [b["id"] for b in batch],
        "x": torch.stack(batch_x),
        # texts[i] sejajar dengan view ke-i pada x. Indeks 0 (anchor) dan indeks 1
        # (grafem kanonik) sama-sama memakai teks kanonik: view 0 adalah realisasi
        # fonemnya, view 1 adalah teksnya sendiri. Teacher semantik membaca
        # texts[0], yaitu teks kanonik.
        "texts": [
            [b["x_canon_text"], b["x_canon_text"]] + b["x_aug_1"] + b["x_aug_2"]
            for b in batch
        ],
        "x_lengths": [[l + 2 for l in b["x_lengths"]] for b in batch],
        "mask": torch.stack(batch_mask),
    }


class AugmentDataset(Dataset):
    def __init__(
        self,
        lexicon_path: str,
        dataset_path: Optional[str] = None,
        mode: str = "huggingface",
        hf_dataset: Optional[HFDataset] = None,
        tokenizer: Optional[PhonemeTokenizer] = None,
        max_text_chars: Optional[int] = _MAX_TEXT_CHARS,
        exempt_id_prefix: str = _EXEMPT_ID_PREFIX,
        canon_mode: str = "phoneme",
        n_aug_1: int = 3,
        n_aug_2: int = 2,
    ):
        assert canon_mode in ("phoneme", "text"), (
            f"canon_mode harus 'phoneme' atau 'text', dapat {canon_mode!r}"
        )
        # Disimpan sebagai atribut karena DataLoader hanya memanggil dataset[idx];
        # argumen default pada __getitem__ tidak bisa diatur dari luar.
        self.canon_mode = canon_mode
        self.n_aug_1 = n_aug_1
        self.n_aug_2 = n_aug_2
        if hf_dataset is not None:
            self.dataset = hf_dataset
        else:
            self.dataset = _load_and_clean(
                mode, dataset_path, max_text_chars=max_text_chars, exempt_id_prefix=exempt_id_prefix
            )

        if "phoneme_negation" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "phoneme_negation", ["-"] * len(self.dataset)
            )

        self.augmenter = RuleBasedAugmentor(lexicon_path=lexicon_path)

        if tokenizer is None:
            self.phoneme_tokenizer = PhonemeTokenizer.from_corpus(self.dataset["phoneme"])
        else:
            self.phoneme_tokenizer = tokenizer

        self._n = len(self.dataset)

    def __len__(self):
        return self._n

    def __getitem__(self, idx, n_aug_1=None, n_aug_2=None, canon_mode=None):
        n_aug_1 = self.n_aug_1 if n_aug_1 is None else n_aug_1
        n_aug_2 = self.n_aug_2 if n_aug_2 is None else n_aug_2
        canon_mode = self.canon_mode if canon_mode is None else canon_mode
        try:
            return self._getitem_impl(idx, n_aug_1, n_aug_2, canon_mode)
        except Exception as e:
            fallback_idx = (idx + 1) % self._n
            print(f"[AugmentDataset] idx={idx} korup ({e!r}), fallback ke idx={fallback_idx}")
            return self._getitem_impl(fallback_idx, n_aug_1, n_aug_2, canon_mode)

    def _getitem_impl(self, idx, n_aug_1=3, n_aug_2=2, canon_mode="phoneme"):
        assert canon_mode in ("phoneme", "text"), (
            "Unknown canon repr. return mode. It has to be either phoneme or text."
        )

        if idx < self._n:
            data = self.dataset[idx]  # dict kecil, transient -- bukan seluruh tabel
            text = data["text"]
            phoneme = data["phoneme"]
            unnormalized_text = data["unnormalized_text"]
            data_id = data["id"]
        else:
            base_idx = idx - self._n
            data = self.dataset[base_idx]
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

        # V' = V + 1 view. Indeks 0 = anchor, indeks 1 = grafem kanonik yang TIDAK
        # pernah jadi anchor, indeks 2..V = augmentasi.
        #
        # Anchor DITAMBAHKAN, bukan menggantikan: seluruh view augmentasi tetap
        # utuh dan grafem kanonik selalu hadir sebagai view tersendiri, sehingga
        # pemetaan teks-kanonik -> representasi-kanonik selalu tersupervisi.
        # Itu penting karena saat inferensi model justru menerima teks bersih.
        #
        # Pada canon_mode="text", indeks 0 dan 1 memuat tensor IDENTIK. Redundansi
        # itu disengaja: bentuk tensor dan jumlah forward pass jadi sama persis di
        # kedua modalitas anchor, sehingga perbandingan modalitas tidak terkonfound
        # oleh perbedaan banyaknya view augmentasi.
        x_graph_canon = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        if canon_mode == "phoneme":
            x_anchor = self.phoneme_tokenizer.encode_single(phoneme)
        else:
            x_anchor = x_graph_canon.clone()

        x_v = x_aug_1 + x_aug_2
        x_v = [torch.tensor(list(t.encode("utf-8")), dtype=torch.long) for t in x_v]
        x_all = [x_anchor, x_graph_canon] + x_v
        x_lengths = [len(x_i) for x_i in x_all]

        return {
            "id": data_id,
            "x_canon": phoneme,
            "x_canon_text": text,
            "x_aug_1": x_aug_1,
            "x_aug_2": x_aug_2,
            "x": x_all,
            "x_lengths": x_lengths,
            "canon_mode": canon_mode,
        }
