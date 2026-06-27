import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
 
def sinusoidal_PE(length: int, d_model: int, device=None) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe
 
 
def length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
 
 
def masked_mean(z_local: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(z_local.dtype)        # (B, C, 1)
    summed = (z_local * mask_f).sum(dim=1)                # (B, d)
    count = mask_f.sum(dim=1).clamp(min=1.0)              # (B, 1), cegah div-by-zero
    return summed / count
 
 
class SwaraEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
 
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
 
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
 
    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~mask if mask is not None else None
 
        attn_out, _ = self.self_attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = self.norm1(h + self.dropout1(attn_out))  # eq. (9)
 
        ffn_out = self.ffn(h)
        h = self.norm2(h + self.dropout2(ffn_out))  # eq. (11)
        return h
 
 
class SwaraEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SwaraEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
 
    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, mask=mask)
        return h  # z_local^(v), eq. (12)
 
 
class CanonicalLengthPredictor(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)   # W_len,1 in R^{d x d}
        self.fc2 = nn.Linear(d_model, 1)          # W_len,2 in R^{d x 1}
 
    def forward(self, z_global: torch.Tensor) -> torch.Tensor:
        u = F.relu(self.fc1(z_global))              # eq (14)
        l_hat = F.softplus(self.fc2(u)).squeeze(-1)  # eq (15)
        return l_hat
 
 
class CanonicalDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads_can: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads_can == 0
 
        self.self_attn = nn.MultiheadAttention(d_model, n_heads_can, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
 
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads_can, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
 
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
 
    def forward(
        self,
        q_can: torch.Tensor,
        z_local: torch.Tensor,
        query_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_kpm = ~query_mask if query_mask is not None else None
        attn_out, _ = self.self_attn(q_can, q_can, q_can, key_padding_mask=self_kpm, need_weights=False)
        q_can = self.norm1(q_can + self.dropout1(attn_out))  # eq (21)
 
        cross_kpm = ~source_mask if source_mask is not None else None
        cross_out, _ = self.cross_attn(q_can, z_local, z_local, key_padding_mask=cross_kpm, need_weights=False)
        q_can = self.norm2(q_can + self.dropout2(cross_out))  # eq (27)
 
        ffn_out = self.ffn(q_can)
        q_can = self.norm3(q_can + self.dropout3(ffn_out))  # eq (28)
        return q_can
 
 
class CanonicalDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads_can: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            CanonicalDecoderLayer(d_model, n_heads_can, d_ff, dropout)
            for _ in range(n_layers)
        ])
 
    def forward(
        self,
        z_local: torch.Tensor,
        l_star: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = z_local.shape[0]
        device = z_local.device
        l_star = l_star.to(device=device, dtype=torch.long)
        l_star_max = int(l_star.max().item())
 
        pe = sinusoidal_PE(l_star_max, self.d_model, device=device)  # (L*_max, d)
        q_can = pe.unsqueeze(0).expand(B, -1, -1).clone()  # eq (17)
 
        query_mask = length_to_mask(l_star, l_star_max)  # (B, L*_max) bool
 
        for layer in self.layers:
            q_can = layer(q_can, z_local, query_mask=query_mask, source_mask=source_mask)
 
        return q_can, query_mask
 
 
class SwaraJEPA(nn.Module): 
    def __init__(
        self,
        n_vocab_text: int,
        n_vocab_phoneme: int,
        d_model: int,
        n_attn_heads: int = 8,
        enc_layers: int = 6,
        can_dec_layers: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.text_embedding = nn.Embedding(num_embeddings=n_vocab_text, embedding_dim=d_model)
        self.phoneme_embedding = nn.Embedding(num_embeddings=n_vocab_phoneme, embedding_dim=d_model)
        self.encoder = SwaraEncoder(d_model=d_model, n_heads=n_attn_heads, d_ff=4 * d_model, n_layers=enc_layers, dropout=0.0)
        self.canonical_decoder = CanonicalDecoder(
            d_model=d_model, n_heads_can=n_attn_heads, d_ff=4 * d_model, n_layers=can_dec_layers, dropout=0.0
        )
        self.length_predictor = CanonicalLengthPredictor(d_model=d_model)
 
    def _embed(self, ids: torch.Tensor, use_phoneme: bool) -> torch.Tensor:
        """Embedding + PE, eq (1)-(3). Dipisah jadi helper karena dipakai berkali-kali."""
        emb = self.phoneme_embedding(ids) if use_phoneme else self.text_embedding(ids)
        pe = sinusoidal_PE(ids.shape[1], self.d_model, device=ids.device)  # (C, d)
        return emb + pe.unsqueeze(0)  # eq (3): h_0 = E[x] + PE, broadcast ke (B, C, d)
 
    def forward(self, x: dict, type: str = "phoneme", mode: str = "inference") -> dict:
        assert type in ("phoneme", "text"), "type harus 'phoneme' atau 'text'"
        assert mode in ("training", "inference"), "mode harus 'training' atau 'inference'"
        use_phoneme = (type == "phoneme")
 
        if mode == "training":
            # x["x"]:    (B, V, C)  -- index 0 = anchor, index 1..V-1 = augmented views
            # x["mask"]: (B, V, C)  -- Mask^(v), True = token asli
 
            # --- v = 1 (anchor) ---
            mask_1 = x["mask"][:, 0, :]
            h0_1 = self._embed(x["x"][:, 0, :], use_phoneme=use_phoneme)
            z_local_1 = self.encoder(h=h0_1, mask=mask_1)
            z_global_1 = masked_mean(z_local_1, mask_1)  # FIX: masked, bukan .mean(1) naif
 
            # --- v = 2..V (augmented, selalu byte/text -- tidak pernah phoneme) ---
            n_views_aug = x["x"].shape[1] - 1
            z_local_v, z_global_v, source_mask_v = [], [], []
 
            for i in range(n_views_aug):
                mask_v = x["mask"][:, i + 1, :]
                h0_v = self._embed(x["x"][:, i + 1, :], use_phoneme=False)
                z_local_vi = self.encoder(h=h0_v, mask=mask_v)
                z_local_v.append(z_local_vi)
                z_global_v.append(masked_mean(z_local_vi, mask_v))  # FIX: masked
                source_mask_v.append(mask_v)
 
            z_local_v = torch.stack(z_local_v, dim=1)    # FIX dim=1 -> (B, V-1, C, d), batch-first
            z_global_v = torch.stack(z_global_v, dim=1)  # FIX dim=1 -> (B, V-1, d)
 
            # Length Predictor: hanya untuk supervisi Lcanonlen (eq 58).
            # L_hat TIDAK dipakai sebagai L* aktual di sini -- saat training,
            # L*^(v) selalu teacher-forced ke C^(1)_b (eq 16), sama untuk semua v.
            l_hat_v = self.length_predictor(z_global_v)  # (B, V-1)
 
            l_star = mask_1.sum(dim=-1).long()           # C^(1)_b: panjang asli anchor per sampel
            l_star_max = int(l_star.max().item())
 
            # v = 1: identitas (eq 30) -- JANGAN dilewatkan canonical_decoder
            zhat_canon_1 = z_local_1[:, :l_star_max, :]
            canon_mask = mask_1[:, :l_star_max]           # == length_to_mask(l_star, l_star_max), by construction
 
            # v = 2..V: lewat Canonical Decoder, l_star SAMA untuk semua view
            # -> otomatis menghasilkan lebar L*_max yang konsisten antar-view.
            zhat_canon_v = []
            for i in range(n_views_aug):
                zhat_canon_vi, _query_mask_vi = self.canonical_decoder(
                    z_local_v[:, i, :, :], l_star, source_mask=source_mask_v[i]
                )
                zhat_canon_v.append(zhat_canon_vi)
 
            zhat_canon = torch.stack([zhat_canon_1] + zhat_canon_v, dim=1)  # (B, V, L*_max, d)
 
            return {
                "z_local_1": z_local_1,        # (B, C, d)
                "z_global_1": z_global_1,      # (B, d)
                "z_local_v": z_local_v,         # (B, V-1, C, d)
                "z_global_v": z_global_v,       # (B, V-1, d)
                "l_hat_v": l_hat_v,             # (B, V-1)            -> utk Lcanonlen (eq 58)
                "l_star": l_star,               # (B,)                -> target C^(1)_b
                "zhat_canon": zhat_canon,       # (B, V, L*_max, d)    -> utk Lcanon (eq 57)
                "canon_mask": canon_mask,       # (B, L*_max), SAMA untuk semua v (lih. catatan Bagian 4.6)
            }
 
        else:  # inference -- satu teks input, tanpa anchor acuan (Algoritma 6)
            mask = x["mask"][:, 0, :]
            h0 = self._embed(x["x"][:, 0, :], use_phoneme=use_phoneme)
            z_local = self.encoder(h=h0, mask=mask)
            z_global = masked_mean(z_local, mask)
 
            l_hat = self.length_predictor(z_global)         # (B,)
            l_star = l_hat.round().long().clamp(min=1)       # eq (16): inference -> round(L_hat)
 
            # Canonical Decoder SELALU dijalankan saat inference (tidak ada
            # identity shortcut -- kita tidak pernah tahu input ini "anchor"
            # asli atau bukan).
            zhat_canon, canon_mask = self.canonical_decoder(z_local, l_star, source_mask=mask)
 
            return {
                "z_local": z_local,
                "z_global": z_global,
                "l_hat": l_hat,
                "zhat_canon": zhat_canon,
                "canon_mask": canon_mask,
            }