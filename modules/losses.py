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
        hf_dataset: Optional[HFDataset] = None,
        tokenizer: Optional[PhonemeTokenizer] = None,
        max_text_chars: Optional[int] = _MAX_TEXT_CHARS,
        exempt_id_prefix: str = _EXEMPT_ID_PREFIX,
    ):
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

    def __getitem__(self, idx, n_aug_1=3, n_aug_2=2, canon_mode="phoneme"):
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
i0002672@xlogin1:~$ cat text-encoder-trial/modules/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

from modules.utils import move_batch_to_device

def canon_len_loss(out: dict) -> torch.Tensor:
    l_v_preds = out['l_v_preds']
    l_v_gts = out['l_v_gts'].detach()
    loss = F.mse_loss(torch.log1p(l_v_preds), torch.log1p(l_v_gts))
    return loss


def syntactical_loss(out: dict) -> torch.Tensor:
    z_v = out["z_v"]
    z_canon = out["z_v_canon"]
    masks = out["masks_v_canon"]

    B, V = z_canon.shape[0], z_canon.shape[1]
    L = min(z_v.size(2), z_canon.size(2), masks.size(2))

    mask = masks[:, 0, :L].unsqueeze(-1).float()

    z_v_trunc = z_v[:, :, :L, :] * mask.unsqueeze(1)
    mu = z_v_trunc.sum(dim=1) / V

    denom = (mask.sum() * z_canon.size(-1) * V).clamp(min=1.0)

    loss = 0.0
    for v in range(V):
        z_canon_v = z_canon[:, v, :L, :] * mask
        loss = loss + F.mse_loss(z_canon_v, mu, reduction='sum')

    return loss / denom

def masked_mean(z: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    summed = (z * mask_f).sum(dim=-2)
    count = mask_f.sum(dim=-2).clamp(min=eps)
    return summed / count

class SemanticTeacherSimilarity(nn.Module):
    def __init__(self, teacher_model: str = "tum-nlp/NegMPNet"):
        super().__init__()
        self.teacher_semantic_model = SentenceTransformer(teacher_model).eval()

    @torch.no_grad()
    def _get_raw(self, batch: dict) -> torch.Tensor:
        canonical_texts = [t[0] for t in batch['texts']]
        emb = self.teacher_semantic_model.encode(canonical_texts, convert_to_tensor=True)
        return emb

    def forward(
        self,
        batch: dict,
        n_core: int = 64,
        negation_offset: int = 62,
        n_negation: int = 2,
    ) -> torch.Tensor:
        emb = F.normalize(self._get_raw(batch), dim=-1, eps=1e-8)

        emb_A = emb[:n_core]
        emb_B = torch.cat(
            [emb[:negation_offset], emb[n_core:n_core + n_negation]],
            dim=0,
        )
        return emb_A @ emb_B.t()

    def get_raw_embeddings(self, batch: dict) -> torch.Tensor:
        """Dipakai magnitude_loss -- embedding teacher mentah, (B_total, d_teacher)."""
        return self._get_raw(batch)


class CachedSemanticTeacherSimilarity(nn.Module):
    def __init__(self, base_teacher: nn.Module, cache_size: int = 500_000):
        super().__init__()
        self.base_teacher = base_teacher
        for p in self.base_teacher.parameters():
            p.requires_grad_(False)
        self.base_teacher.eval()
        self._cache: dict = {}
        self._cache_size = cache_size

    def train(self, mode: bool = True):
        self.base_teacher.eval()
        return self

    @torch.no_grad()
    def _get_raw(self, batch) -> torch.Tensor:
        """
        Cache sekarang menyimpan embedding MENTAH (belum dinormalisasi) --
        supaya forward() (cosine-sim, normalisasi di sini) dan
        get_raw_embeddings() (magnitude_loss, butuh skala asli) sama-sama
        dilayani dari SATU encode() saja, tidak encode dua kali per teks.
        """
        ids = batch["id"]
        texts = [t[0] for t in batch["texts"]]
        device = (
            next(self.base_teacher.teacher_semantic_model.parameters(), torch.tensor(0.0)).device
            if hasattr(self.base_teacher.teacher_semantic_model, "parameters")
            else torch.device("cpu")
        )

        missing_idx = [i for i, _id in enumerate(ids) if _id not in self._cache]
        local_lookup = {}
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_embs = self.base_teacher.teacher_semantic_model.encode(missing_texts, convert_to_tensor=True)
            for local_i, global_i in enumerate(missing_idx):
                emb_i = new_embs[local_i].detach().cpu()
                local_lookup[ids[global_i]] = emb_i
                if len(self._cache) < self._cache_size:
                    self._cache[ids[global_i]] = emb_i

        emb = torch.stack([
            self._cache.get(_id, local_lookup.get(_id)) for _id in ids
        ]).to(device)
        return emb

    @torch.no_grad()
    def forward(self, batch, n_core=64, negation_offset=62, n_negation=2):
        emb = F.normalize(self._get_raw(batch), dim=-1, eps=1e-8)
        emb_A = emb[:n_core]
        emb_B = torch.cat([emb[:negation_offset], emb[n_core:n_core + n_negation]], dim=0)
        return emb_A @ emb_B.t()

    @torch.no_grad()
    def get_raw_embeddings(self, batch) -> torch.Tensor:
        """Dipakai magnitude_loss -- embedding teacher mentah, (B_total, d_teacher)."""
        return self._get_raw(batch)

    def cache_stats(self) -> dict:
        return {"cached_sentences": len(self._cache)}


def semantic_loss(
    batch: dict,
    out: dict,
    cossim_fn,
    eps: float = 1e-6,
    n_core: int = 64,
    negation_offset: int = 62,
    n_negation: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_c = out['z_v_canon']
    m_c = out['masks_v_canon']

    pooled = masked_mean(z_c, m_c, eps=eps)
    pooled_norm = F.normalize(pooled, dim=-1, eps=eps)

    V, d = pooled_norm.shape[1], pooled_norm.shape[2]

    pooled_A = pooled_norm[:n_core]
    pooled_B = torch.cat(
        [pooled_norm[:negation_offset], pooled_norm[n_core:n_core + n_negation]],
        dim=0,
    )

    flat_A = pooled_A.reshape(n_core * V, d)
    flat_B = pooled_B.reshape(n_core * V, d)

    sim_all = flat_A @ flat_B.t()
    sim_all = sim_all.view(n_core, V, n_core, V).permute(0, 2, 1, 3)

    teacher_sim = cossim_fn(batch, n_core=n_core, negation_offset=negation_offset, n_negation=n_negation)
    assert teacher_sim.shape == (n_core, n_core), (
        f"teacher_sim harus ({n_core},{n_core}), didapat {tuple(teacher_sim.shape)}"
    )
    teacher_sim = teacher_sim.to(sim_all.device)

    D = sim_all - teacher_sim.view(n_core, n_core, 1, 1)
    delta = D.pow(2).sum(dim=(-2, -1))

    neg_block_mask = torch.zeros(n_core, n_core, dtype=torch.bool, device=delta.device)
    if negation_offset < n_core:
        neg_block_mask[negation_offset:n_core, negation_offset:n_core] = True

    n_gen_entries = (~neg_block_mask).sum().clamp(min=1)
    n_neg_entries = neg_block_mask.sum().clamp(min=1)

    l_sem_general = delta[~neg_block_mask].sum() / n_gen_entries
    l_sem_negation = delta[neg_block_mask].sum() / n_neg_entries

    l_sem_negation_contrast = torch.zeros((), device=delta.device, dtype=delta.dtype)
    if negation_offset < n_core and n_core > 1:
        sim_canon = sim_all[:, :, 0, 0]
        premise_rows = torch.arange(negation_offset, n_core, device=delta.device)
        denom = float(n_core - 1)

        teacher_diag = teacher_sim[premise_rows, premise_rows]
        teacher_baseline = (teacher_sim[premise_rows].sum(dim=1) - teacher_diag) / denom
        teacher_contrast = (teacher_diag - teacher_baseline).detach()

        student_diag = sim_canon[premise_rows, premise_rows]
        student_baseline = (sim_canon[premise_rows].sum(dim=1) - student_diag) / denom
        student_contrast = student_diag - student_baseline

        l_sem_negation_contrast = F.mse_loss(student_contrast, teacher_contrast)

    return l_sem_general, l_sem_negation, l_sem_negation_contrast


class TeacherMagnitudeProjection(nn.Module):
    """
    Proyeksi linear student (d_model) -> ruang embedding teacher (d_teacher).
    Modul ini SCAFFOLDING MURNI untuk magnitude_loss:
      - Parameternya WAJIB didaftarkan ke optimizer yang sama dengan model
        utama (lihat contoh wiring di bawah). Kalau tidak, dia diam di
        inisialisasi acak dan magnitude_loss justru MERUSAK training --
        mendorong z_v_canon mengejar output acak yang tidak berarti.
      - Modul ini BUKAN bagian dari TLeJEPA dan tidak dipakai saat inference.
        Setelah training selesai, buang saja state_dict-nya (jangan ikut
        di-load ke model final).
    """
    def __init__(self, d_model: int, d_teacher: int = 768):
        super().__init__()
        self.proj = nn.Linear(d_model, d_teacher)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def magnitude_loss(
    batch: dict,
    out: dict,
    cossim_fn,
    proj_head: "TeacherMagnitudeProjection",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Regresi MSE langsung per-kalimat (bukan matriks N x N seperti
    semantic_loss) antara embedding teacher MENTAH (belum dinormalisasi --
    magnitude-nya masih ada) dan pooled decoder output KANONIK (v=0) student,
    setelah diproyeksikan ke dimensi teacher lewat proj_head.

    Bedanya dari semantic_loss: semantic_loss pakai cosine similarity (murni
    sudut, F.normalize dulu di kedua sisi) -- di sini MSE mentah otomatis ikut
    menghukum selisih PANJANG vektor, bukan cuma selisih arah.
    """
    if not hasattr(cossim_fn, "get_raw_embeddings"):
        raise AttributeError(
            "cossim_fn butuh method get_raw_embeddings(batch) -> (B_total, d_teacher) "
            "untuk magnitude_loss. Pastikan pakai SemanticTeacherSimilarity / "
            "CachedSemanticTeacherSimilarity versi terbaru di file ini."
        )

    teacher_raw = cossim_fn.get_raw_embeddings(batch).detach()

    z_c = out['z_v_canon']
    m_c = out['masks_v_canon']

    pooled = masked_mean(z_c[:, 0, :, :], m_c[:, 0, :], eps=eps)
    pooled_proj = proj_head(pooled)

    teacher_centered = teacher_raw - teacher_raw.mean(dim=0, keepdim=True)
    student_centered = pooled_proj - pooled_proj.mean(dim=0, keepdim=True)

    return F.mse_loss(student_centered, teacher_centered.to(student_centered.device))


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_slices: int = 256, t_max: float = 3.0):
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_slices = num_slices
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        A = torch.randn(pooled.size(-1), self.num_slices, device=pooled.device)
        A = A / A.norm(p=2, dim=0)
        x_t = (pooled @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * pooled.size(-2)
        return statistic.mean()

def sigreg_loss(out: dict, sigreg_fn: "SIGReg", key: str = "z_v", mask_key: str = "masks") -> torch.Tensor:
    """
    Digeneralisasi supaya bisa dipanggil ke ruang embedding MANA PUN --
    dipakai dua kali di compute_losses: sekali utk 'z_v' (encoder), sekali
    lagi utk 'z_v_canon' (decoder). Ini SESUAI Algorithm 2 LeJEPA: SIGReg
    harus menargetkan embedding yang SAMA dengan yang dipakai loss prediktif
    (di sini: syntactic_loss/semantic_loss/magnitude_loss, yang semuanya
    beroperasi di z_v_canon) -- bukan cuma z_v seperti sebelumnya.
    """
    z, masks = out[key], out[mask_key]
    pooled = masked_mean(z, masks)
    return sigreg_fn(pooled.transpose(0, 1))

def compute_losses(model, batch, criterion, n_core, n_negation, device, use_amp: bool = True, amp_dtype=torch.bfloat16, canon_type: str = "phoneme"):
    batch = move_batch_to_device(batch, device)

    autocast_enabled = use_amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        out = model.train_forward(batch, type=canon_type)

        l_syn = syntactical_loss(out)
        l_sem_general, l_sem_negation, l_sem_negation_contrast = semantic_loss(
            batch, out, criterion.cossim_fn,
            n_core=n_core, negation_offset=n_core - n_negation, n_negation=n_negation,
        )
        l_mag = magnitude_loss(batch, out, criterion.cossim_fn, criterion.proj_head)

        l_sig_prior = sigreg_loss(out, criterion.sigreg_fn, key="z_v", mask_key="masks")
        l_sig_canon = sigreg_loss(out, criterion.sigreg_fn, key="z_v_canon", mask_key="masks_v_canon")
        l_sig = l_sig_prior + criterion.sigreg_canon_weight * l_sig_canon
        l_len = canon_len_loss(out)

        l_sem_negation_combined = l_sem_negation + l_sem_negation_contrast

        canonical_embedding_obj = (
            criterion.zeta_1 * l_syn
            + criterion.zeta_2 * l_sem_general
            + criterion.zeta_2_neg * l_sem_negation_combined
            + criterion.zeta_mag * l_mag
        )
        total = (1 - criterion.lambda_) * canonical_embedding_obj \
            + criterion.lambda_ * l_sig \
            + criterion.canon_len_weight * l_len

    losses = {
        "total": total,
        "syntactic": l_syn.detach(),
        "semantic": (l_sem_general + l_sem_negation_combined).detach(),
        "semantic_general": l_sem_general.detach(),
        "semantic_negation": l_sem_negation.detach(),
        "semantic_negation_contrast": l_sem_negation_contrast.detach(),
        "magnitude": l_mag.detach(),
        "sigreg": l_sig.detach(),
        "sigreg_prior": l_sig_prior.detach(),
        "sigreg_canon": l_sig_canon.detach(),
        "canon_len": l_len.detach(),
    }
    return total, losses, out, batch

@dataclass
class TLeJEPACriterion:
    sigreg_fn: SIGReg
    cossim_fn: nn.Module
    proj_head: "TeacherMagnitudeProjection"
    lambda_: float = 0.5
    zeta_1: float = 1.0
    zeta_2: float = 1.0
    zeta_2_neg: float = 1.0
    zeta_mag: float = 1.0
    canon_len_weight: float = 1.0
    sigreg_canon_weight: float = 1.0

    def to(self, device):
        self.sigreg_fn = self.sigreg_fn.to(device)
        self.cossim_fn = self.cossim_fn.to(device)
        self.proj_head = self.proj_head.to(device)
        return self
