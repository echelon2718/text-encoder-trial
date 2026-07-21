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
