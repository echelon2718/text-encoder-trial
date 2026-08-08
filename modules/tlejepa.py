import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.utils import length_to_mask


def deepnorm_coeffs(n_enc: int, n_dec: int) -> dict:
    p = math.pow(math.pow(n_enc, 4) * n_dec, 1.0 / 16.0)
    return {
        "enc_alpha": 0.81 * p,
        "enc_beta": 0.87 / p,
        "dec_alpha": math.pow(3.0 * n_dec, 0.25),
        "dec_beta": math.pow(12.0 * n_dec, -0.25),
    }


def _deepnorm_init_attention(attn: nn.MultiheadAttention, beta: float):
    d = attn.embed_dim
    w = attn.in_proj_weight
    nn.init.xavier_uniform_(w[:d])
    nn.init.xavier_uniform_(w[d:2 * d])
    nn.init.xavier_uniform_(w[2 * d:], gain=beta)
    nn.init.xavier_uniform_(attn.out_proj.weight, gain=beta)
    if attn.in_proj_bias is not None:
        nn.init.zeros_(attn.in_proj_bias)
    if attn.out_proj.bias is not None:
        nn.init.zeros_(attn.out_proj.bias)


def _deepnorm_init_ffn(ffn: nn.Sequential, beta: float):
    for m in ffn:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=beta)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def sinusoidal_PE(
    length: int,
    d_model: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    position = torch.arange(
        length, device=device, dtype=torch.float32
    ).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(
            0, d_model, 2, device=device, dtype=torch.float32
        ) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(
        length, d_model, device=device, dtype=torch.float32
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(
        position * div_term[: pe[:, 1::2].shape[1]]
    )
    return pe.to(dtype=dtype)


def uniform_copy_to_length(
    source_embeddings: torch.Tensor,
    source_mask: torch.Tensor,
    target_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gu et al. (2018) Section 3.2 style uniform source copying.

    For each target position t (1-based), copy source position

        Round(T_src * t / T_tgt)

    then convert it to a 0-based tensor index.

    This is discrete copying / nearest-neighbor resampling, NOT linear
    interpolation. No stop-gradient is applied.

    source_embeddings: [B, L_src, D], BEFORE source positional encoding.
    source_mask:       [B, L_src], bool, standard right-padding mask.
    target_lengths:    [B].
    """
    if source_embeddings.dim() != 3:
        raise ValueError(
            "source_embeddings must have shape [B, L_src, D], "
            f"got {tuple(source_embeddings.shape)}"
        )

    if source_mask.dim() != 2:
        raise ValueError(
            "source_mask must have shape [B, L_src], "
            f"got {tuple(source_mask.shape)}"
        )

    B, L_src, D = source_embeddings.shape
    if source_mask.shape != (B, L_src):
        raise ValueError(
            "source_mask shape must match source_embeddings[:2]: "
            f"expected {(B, L_src)}, got {tuple(source_mask.shape)}"
        )

    device = source_embeddings.device
    source_mask = source_mask.to(device=device, dtype=torch.bool)
    target_lengths = target_lengths.to(device=device, dtype=torch.long)

    if target_lengths.shape != (B,):
        raise ValueError(
            f"target_lengths must have shape [B], got {tuple(target_lengths.shape)}"
        )
    if torch.any(target_lengths < 1):
        raise ValueError("All target lengths must be >= 1.")

    source_lengths = source_mask.sum(dim=1).long()
    if torch.any(source_lengths < 1):
        raise ValueError("Every sample must contain at least one valid source token.")

    # This implementation assumes ordinary right padding:
    # [True, True, ..., True, False, False, ...].
    positions = torch.arange(L_src, device=device).unsqueeze(0)
    expected_mask = positions < source_lengths.unsqueeze(1)
    if not torch.equal(source_mask, expected_mask):
        raise ValueError(
            "uniform_copy_to_length expects right-padded source masks "
            "(all valid tokens before all padding tokens)."
        )

    L_tgt_max = int(target_lengths.max().item())

    # 1-based target positions t = 1..T_tgt.
    t = torch.arange(
        1, L_tgt_max + 1, device=device, dtype=torch.float32
    ).unsqueeze(0)

    src_len_f = source_lengths.to(torch.float32).unsqueeze(1)
    tgt_len_f = target_lengths.to(torch.float32).unsqueeze(1)

    # j = Round(T_src * t / T_tgt), 1-based.
    # floor(x + 0.5) implements ordinary round-to-nearest.
    source_indices = torch.floor(
        src_len_f * t / tgt_len_f + 0.5
    ).long() - 1

    source_indices = source_indices.clamp(min=0)
    source_indices = torch.minimum(
        source_indices,
        (source_lengths - 1).unsqueeze(1),
    )

    gather_index = source_indices.unsqueeze(-1).expand(
        B, L_tgt_max, D
    )
    copied = torch.gather(
        source_embeddings,
        dim=1,
        index=gather_index,
    )

    target_mask = length_to_mask(
        target_lengths, L_tgt_max
    ).to(device=device, dtype=torch.bool)

    copied = copied * target_mask.unsqueeze(-1).to(copied.dtype)
    return copied, target_mask


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        alpha: float = 1.0,
        beta: float = None,
    ):
        super().__init__()
        self.alpha = alpha
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

        if beta is not None:
            _deepnorm_init_attention(self.self_attn, beta)
            _deepnorm_init_ffn(self.ffn, beta)

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.self_attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = self.norm1(
            self.residual_connection(self.dropout1(attn_out), h)
        )

        ffn_out = self.ffn(h)
        h = self.norm2(
            self.residual_connection(self.dropout2(ffn_out), h)
        )
        return h


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        dropout,
        alpha: float = 1.0,
        beta: float = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                alpha=alpha,
                beta=beta,
            )
            for _ in range(n_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            if (
                self.gradient_checkpointing
                and self.training
                and h.requires_grad
            ):
                h = checkpoint(
                    layer,
                    h,
                    mask,
                    use_reentrant=False,
                )
            else:
                h = layer(h, mask=mask)
        return h


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        alpha: float = 1.0,
        beta: float = None,
    ):
        super().__init__()
        self.alpha = alpha

        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        if beta is not None:
            _deepnorm_init_attention(self.self_attn, beta)
            _deepnorm_init_attention(self.cross_attn, beta)
            _deepnorm_init_ffn(self.ffn, beta)

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        z,
        q_canon,
        query_mask,
        source_mask,
    ):
        self_kpm = ~query_mask if query_mask is not None else None
        attn_out, _ = self.self_attn(
            q_canon,
            q_canon,
            q_canon,
            key_padding_mask=self_kpm,
            need_weights=False,
        )
        q_canon = self.norm1(
            self.residual_connection(
                self.dropout1(attn_out),
                q_canon,
            )
        )

        cross_kpm = ~source_mask if source_mask is not None else None
        cross_out, _ = self.cross_attn(
            q_canon,
            z,
            z,
            key_padding_mask=cross_kpm,
            need_weights=False,
        )
        q_canon = self.norm2(
            self.residual_connection(
                self.dropout2(cross_out),
                q_canon,
            )
        )

        ffn_out = self.ffn(q_canon)
        q_canon = self.norm3(
            self.residual_connection(
                self.dropout3(ffn_out),
                q_canon,
            )
        )
        return q_canon


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        max_length=2048,
        dropout=0.1,
        alpha: float = 1.0,
        beta: float = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                alpha=alpha,
                beta=beta,
            )
            for _ in range(n_layers)
        ])

        # No canonical_query_library anymore.
        # The initial decoder canvas is copied from source token embeddings.
        self.input_norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = False

    def forward(
        self,
        z,
        source_embeddings,
        l_star,
        source_mask,
    ):
        B = z.shape[0]
        device = z.device

        if source_embeddings.shape != z.shape:
            raise ValueError(
                "source_embeddings and z must have identical [B, L_src, D] "
                f"shapes, got {tuple(source_embeddings.shape)} and "
                f"{tuple(z.shape)}"
            )

        l_star = l_star.to(device=device, dtype=torch.long)
        if l_star.shape != (B,):
            raise ValueError(
                f"l_star must have shape [B], got {tuple(l_star.shape)}"
            )

        l_star_max = int(l_star.max().item())
        if l_star_max > self.max_length:
            raise ValueError(
                f"Canonical length {l_star_max} exceeds "
                f"decoder max_length={self.max_length}."
            )

        # Gu et al. uniform copied source-input canvas.
        query_canonical, query_mask = uniform_copy_to_length(
            source_embeddings=source_embeddings,
            source_mask=source_mask,
            target_lengths=l_star,
        )

        # Target/canonical positional encoding, not source positional encoding.
        pe = sinusoidal_PE(
            l_star_max,
            self.d_model,
            device=device,
            dtype=query_canonical.dtype,
        )
        query_canonical = query_canonical + pe.unsqueeze(0)
        query_canonical = self.input_norm(query_canonical)
        query_canonical = (
            query_canonical
            * query_mask.unsqueeze(-1).to(query_canonical.dtype)
        )

        # Existing decoder stack is intentionally left unchanged so the
        # experiment isolates the decoder-input/canvas change.
        for layer in self.layers:
            if (
                self.gradient_checkpointing
                and self.training
                and query_canonical.requires_grad
            ):
                query_canonical = checkpoint(
                    layer,
                    z,
                    query_canonical,
                    query_mask,
                    source_mask,
                    use_reentrant=False,
                )
            else:
                query_canonical = layer(
                    z,
                    query_canonical,
                    query_mask=query_mask,
                    source_mask=source_mask,
                )

        return query_canonical, query_mask


class MaskedAttentionPooling(nn.Module):
    def __init__(self, d_model, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.score(z).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (z * weights).sum(dim=1)


class LengthPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden: int = 256,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.pool = MaskedAttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        z,
        mask,
        detach_input: bool = True,
    ):
        if detach_input:
            z = z.detach()
        pooled = self.pool(z, mask)
        lengths = mask.sum(dim=1).float().clamp(min=1.0)
        log_len = torch.log(lengths).unsqueeze(-1)
        feat = torch.cat([pooled, log_len], dim=-1)
        raw = self.head(feat).squeeze(-1)
        l_pred = F.softplus(raw) + self.eps
        return l_pred, torch.log(l_pred)


class TLeJEPA(nn.Module):
    def __init__(
        self,
        n_vocab_text,
        n_vocab_phoneme,
        d_model=256,
        n_attn_heads=8,
        enc_layers=6,
        dec_layers=4,
        max_length=4096,
        dropout=0.1,
        use_deepnorm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_deepnorm = use_deepnorm

        c = deepnorm_coeffs(enc_layers, dec_layers)
        if use_deepnorm:
            enc_alpha = c["enc_alpha"]
            enc_beta = c["enc_beta"]
            dec_alpha = c["dec_alpha"]
            dec_beta = c["dec_beta"]
        else:
            enc_alpha = dec_alpha = 1.0
            enc_beta = dec_beta = None
        self.deepnorm_coeffs = c

        self.text_embedding = nn.Embedding(n_vocab_text, d_model)
        self.phoneme_embedding = nn.Embedding(n_vocab_phoneme, d_model)
        nn.init.normal_(
            self.text_embedding.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.normal_(
            self.phoneme_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        self.embed_norm = nn.LayerNorm(d_model)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_attn_heads,
            d_ff=4 * d_model,
            n_layers=enc_layers,
            dropout=dropout,
            alpha=enc_alpha,
            beta=enc_beta,
        )

        self.decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_attn_heads,
            d_ff=4 * d_model,
            n_layers=dec_layers,
            max_length=max_length,
            dropout=dropout,
            alpha=dec_alpha,
            beta=dec_beta,
        )

        self.length_predictor = LengthPredictor(
            d_model=d_model
        )

    def set_gradient_checkpointing(
        self,
        enabled: bool = True,
    ):
        self.encoder.gradient_checkpointing = enabled
        self.decoder.gradient_checkpointing = enabled
        return self

    def _embed(
        self,
        ids,
        use_phoneme: bool,
        return_source_embeddings: bool = False,
    ):
        # Raw token embeddings. These are what get uniformly copied into the
        # decoder canvas. They do NOT yet contain source positional encoding.
        source_embeddings = (
            self.phoneme_embedding(ids)
            if use_phoneme
            else self.text_embedding(ids)
        )

        source_pe = sinusoidal_PE(
            ids.shape[1],
            self.d_model,
            device=ids.device,
            dtype=source_embeddings.dtype,
        )

        # Encoder sees normal source embedding + source PE.
        encoder_input = self.embed_norm(
            source_embeddings + source_pe.unsqueeze(0)
        )

        if return_source_embeddings:
            return encoder_input, source_embeddings
        return encoder_input

    def train_forward(
        self,
        x: dict,
        type: str = "phoneme",
    ) -> dict:
        assert type in ("phoneme", "text")
        use_phoneme = type == "phoneme"

        B, V, L = x["x"].shape
        n_views_aug = V - 1

        # Canonical/original view.
        h0_canon, source_canon = self._embed(
            x["x"][:, 0, :],
            use_phoneme=use_phoneme,
            return_source_embeddings=True,
        )

        # Augmented views remain text views, matching the original code.
        if n_views_aug > 0:
            aug_ids = x["x"][:, 1:, :].reshape(
                B * n_views_aug,
                L,
            )

            h0_aug, source_aug = self._embed(
                aug_ids,
                use_phoneme=False,
                return_source_embeddings=True,
            )

            h0_aug = h0_aug.view(
                B,
                n_views_aug,
                L,
                self.d_model,
            )
            source_aug = source_aug.view(
                B,
                n_views_aug,
                L,
                self.d_model,
            )

            h0 = torch.cat(
                [h0_canon.unsqueeze(1), h0_aug],
                dim=1,
            )
            source_embeddings = torch.cat(
                [source_canon.unsqueeze(1), source_aug],
                dim=1,
            )
        else:
            h0 = h0_canon.unsqueeze(1)
            source_embeddings = source_canon.unsqueeze(1)

        masks = x["mask"]

        h0_flat = h0.reshape(
            B * V,
            L,
            self.d_model,
        )
        source_embeddings_flat = source_embeddings.reshape(
            B * V,
            L,
            self.d_model,
        )
        mask_flat = masks.reshape(
            B * V,
            L,
        )

        # Contextual encoder representation used as decoder K/V.
        z_flat = self.encoder(
            h=h0_flat,
            mask=mask_flat,
        )
        z_v = z_flat.view(
            B,
            V,
            L,
            self.d_model,
        )

        l_pred_flat, _ = self.length_predictor(
            z_flat,
            mask_flat,
        )
        l_v_preds = l_pred_flat.view(B, V)

        # Preserve original training behavior: canonical/original view length
        # is the oracle l* for every view of the same sample.
        l_1 = (
            masks[:, 0, :]
            .sum(dim=1)
            .float()
            .clamp(min=1.0)
        )
        l_v_gts = l_1.unsqueeze(1).expand(B, V)
        l_stars = l_1

        l_star_expanded = l_stars.repeat_interleave(V)

        # Decoder:
        #   Q0 = uniformly copied raw source token embeddings + target PE
        #   K,V = contextual encoder output z_flat
        z_canon_flat, mask_canon_flat = self.decoder(
            z=z_flat,
            source_embeddings=source_embeddings_flat,
            l_star=l_star_expanded,
            source_mask=mask_flat,
        )

        L_star_max = z_canon_flat.shape[1]

        return {
            "z_v": z_v,
            "masks": masks,
            "l_v_preds": l_v_preds,
            "l_v_gts": l_v_gts,
            "z_v_canon": z_canon_flat.view(
                B,
                V,
                L_star_max,
                self.d_model,
            ),
            "masks_v_canon": mask_canon_flat.view(
                B,
                V,
                L_star_max,
            ),
        }

    def forward(
        self,
        text,
        use_phoneme: bool = False,
    ):
        def _one(t):
            t = t.unsqueeze(0) if t.dim() == 1 else t

            mask = torch.ones(
                t.shape,
                dtype=torch.bool,
                device=t.device,
            )

            h, source_embeddings = self._embed(
                t,
                use_phoneme=use_phoneme,
                return_source_embeddings=True,
            )

            z = self.encoder(
                h,
                mask=mask,
            )

            l_hat, _ = self.length_predictor(
                z,
                mask,
            )
            l_star = l_hat.round().long().clamp(min=1)

            zc, _ = self.decoder(
                z=z,
                source_embeddings=source_embeddings,
                l_star=l_star,
                source_mask=mask,
            )
            return zc, l_star

        if isinstance(text, list):
            outs = [_one(t) for t in text]
            return [o[0] for o in outs], [o[1] for o in outs]

        return _one(text)
