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

def syntactical_loss(out):
    z_v = out["z_v"]
    z_canon = out["z_v_canon"]
    masks = out["masks_v_canon"]

    B, V = z_canon.shape[0], z_canon.shape[1]
    L = min(z_v.size(2), z_canon.size(2), masks.size(2))

    z_target = z_v[:, 0, :L, :].detach()
    mask = masks[:, 0, :L].unsqueeze(-1).float()   # identik utk semua v (teacher forcing)
    z_target_masked = z_target * mask

    denom = (mask.sum() * z_canon.size(-1) * V).clamp(min=1.0)

    loss = 0.0
    for v in range(V):
        z_canon_v = z_canon[:, v, :L, :] * mask
        loss = loss + F.mse_loss(z_canon_v, z_target_masked, reduction='sum')

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

    def forward(
        self,
        batch: dict,
        n_core: int = 64,
        negation_offset: int = 62,
        n_negation: int = 2,
    ) -> torch.Tensor:
        canonical_texts = [t[0] for t in batch['texts']]
        with torch.no_grad():
            emb = self.teacher_semantic_model.encode(canonical_texts, convert_to_tensor=True)
        emb = F.normalize(emb, dim=-1, eps=1e-8)

        emb_A = emb[:n_core]
        emb_B = torch.cat(
            [emb[:negation_offset], emb[n_core:n_core + n_negation]],
            dim=0,
        )
        return emb_A @ emb_B.t()  # (n_core, n_core)


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
    def forward(self, batch, n_core=64, negation_offset=62, n_negation=2):
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
            new_embs = F.normalize(new_embs, dim=-1, eps=1e-8)
            for local_i, global_i in enumerate(missing_idx):
                emb_i = new_embs[local_i].detach().cpu()
                local_lookup[ids[global_i]] = emb_i          # selalu tersedia utk batch ini
                if len(self._cache) < self._cache_size:
                    self._cache[ids[global_i]] = emb_i         # persist hanya kalau masih ada slot

        emb = torch.stack([
            self._cache.get(_id, local_lookup.get(_id)) for _id in ids
        ]).to(device)

        emb_A = emb[:n_core]
        emb_B = torch.cat([emb[:negation_offset], emb[n_core:n_core + n_negation]], dim=0)
        return emb_A @ emb_B.t()

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
    z_c = out['z_v_canon']       # (B_total, V, L, d), B_total = n_core + n_negation (mis. 66)
    m_c = out['masks_v_canon']

    pooled = masked_mean(z_c, m_c, eps=eps)             # (B_total, V, d)
    pooled_norm = F.normalize(pooled, dim=-1, eps=eps)

    V, d = pooled_norm.shape[1], pooled_norm.shape[2]

    pooled_A = pooled_norm[:n_core]                                            # (n_core, V, d)
    pooled_B = torch.cat(
        [pooled_norm[:negation_offset], pooled_norm[n_core:n_core + n_negation]],
        dim=0,
    )                                                                            # (n_core, V, d)

    flat_A = pooled_A.reshape(n_core * V, d)
    flat_B = pooled_B.reshape(n_core * V, d)

    sim_all = flat_A @ flat_B.t()                          # (n_core*V, n_core*V)
    sim_all = sim_all.view(n_core, V, n_core, V).permute(0, 2, 1, 3)  # (n_core, n_core, V, V)

    teacher_sim = cossim_fn(batch, n_core=n_core, negation_offset=negation_offset, n_negation=n_negation)
    assert teacher_sim.shape == (n_core, n_core), (
        f"teacher_sim harus ({n_core},{n_core}), didapat {tuple(teacher_sim.shape)}"
    )
    teacher_sim = teacher_sim.to(sim_all.device)

    D = sim_all - teacher_sim.view(n_core, n_core, 1, 1)
    delta = D.pow(2).sum(dim=(-2, -1))          # (n_core, n_core) -- error per pasangan teks (i, j)

    # --- BLOK negasi murni: HANYA baris premis (negation_offset:n_core)
    # x kolom negation-twin (negation_offset:n_core). Baris LAIN yang
    # kebetulan dibandingkan ke kolom negasi (mis. kalimat regular vs
    # negation-twin milik premis lain) BUKAN sinyal negasi -- itu cuma
    # pasangan tak berkaitan biasa, jadi masuk grup general.
    neg_block_mask = torch.zeros(n_core, n_core, dtype=torch.bool, device=delta.device)
    if negation_offset < n_core:
        neg_block_mask[negation_offset:n_core, negation_offset:n_core] = True

    n_gen_entries = (~neg_block_mask).sum().clamp(min=1)
    n_neg_entries = neg_block_mask.sum().clamp(min=1)

    l_sem_general = delta[~neg_block_mask].sum() / n_gen_entries
    l_sem_negation = delta[neg_block_mask].sum() / n_neg_entries

    # --- kontras: Delta(p_i) = sim(p_i, n_i) - rata2 SISA n_core-1 kolom
    # pada baris premis yang sama (leave-one-out atas baris i itu sendiri)
    l_sem_negation_contrast = torch.zeros((), device=delta.device, dtype=delta.dtype)
    if negation_offset < n_core and n_core > 1:
        sim_canon = sim_all[:, :, 0, 0]              # (n_core, n_core), slice kanonik student
        premise_rows = torch.arange(negation_offset, n_core, device=delta.device)   # (n_negation,)
        denom = float(n_core - 1)

        teacher_diag = teacher_sim[premise_rows, premise_rows]                       # (n_negation,)
        teacher_baseline = (teacher_sim[premise_rows].sum(dim=1) - teacher_diag) / denom
        teacher_contrast = (teacher_diag - teacher_baseline).detach()                 # (n_negation,)

        student_diag = sim_canon[premise_rows, premise_rows]                          # (n_negation,)
        student_baseline = (sim_canon[premise_rows].sum(dim=1) - student_diag) / denom
        student_contrast = student_diag - student_baseline

        l_sem_negation_contrast = F.mse_loss(student_contrast, teacher_contrast)

    return l_sem_general, l_sem_negation, l_sem_negation_contrast

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

def sigreg_loss(out: dict, sigreg_fn: "SIGReg") -> torch.Tensor:
    z_v, masks = out['z_v'], out['masks']          # (B, V, L*, d), (B, V, L*)
    pooled = masked_mean(z_v, masks)                # (B, V, d) -- SATU vektor per kalimat per view
    return sigreg_fn(pooled.transpose(0, 1))        # -> (V, B, d): N=B jadi populasi

def compute_losses(model, batch, criterion, n_core, n_negation, device, use_amp: bool = True, amp_dtype=torch.bfloat16, canon_type: str = "phoneme"):
    batch = move_batch_to_device(batch, device)

    autocast_enabled = use_amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        out = model.train_forward(batch, type=canon_type)

        l_syn = syntactical_loss(out)
        l_sem_general, l_sem_negation, l_sem_negation_contrast = semantic_loss(
            batch, out, criterion.cossim_fn,
            n_core=n_core, negation_offset=n_core - n_negation, n_negation=n_negation,
        )  # notes, n_core is batch size
        l_sig = sigreg_loss(out, criterion.sigreg_fn)
        l_len = canon_len_loss(out)

        # l_sem_negation (absolut) dan l_sem_negation_contrast (relatif thd
        # baseline anchor) digabung dulu sebelum dikali zeta_2_neg -- supaya
        # cuma SATU knob baru yang perlu Anda tuning, bukan tiga.
        l_sem_negation_combined = l_sem_negation + l_sem_negation_contrast

        canonical_embedding_obj = (
            criterion.zeta_1 * l_syn
            + criterion.zeta_2 * l_sem_general
            + criterion.zeta_2_neg * l_sem_negation_combined
        )
        total = (1 - criterion.lambda_) * canonical_embedding_obj \
            + criterion.lambda_ * l_sig \
            + criterion.canon_len_weight * l_len

    losses = {
        "total": total,
        "syntactic": l_syn.detach(),
        # kompat: kode logging lama (mis. training.py) masih baca losses["semantic"]
        "semantic": (l_sem_general + l_sem_negation_combined).detach(),
        "semantic_general": l_sem_general.detach(),
        "semantic_negation": l_sem_negation.detach(),
        "semantic_negation_contrast": l_sem_negation_contrast.detach(),
        "sigreg": l_sig.detach(),
        "canon_len": l_len.detach(),
    }
    return total, losses, out, batch

@dataclass
class TLeJEPACriterion:
    sigreg_fn: SIGReg
    cossim_fn: nn.Module
    lambda_: float = 0.5
    zeta_1: float = 1.0
    zeta_2: float = 1.0
    zeta_2_neg: float = 1.0  # bobot terpisah utk (l_sem_negation + l_sem_negation_contrast)
    canon_len_weight: float = 1.0

    def to(self, device):
        self.sigreg_fn = self.sigreg_fn.to(device)
        self.cossim_fn = self.cossim_fn.to(device)
        return self
