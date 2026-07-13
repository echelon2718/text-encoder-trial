import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.utils import length_to_mask

def sinusoidal_PE(length: int, d_model: int, device=None) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
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

    def forward(self, h: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.self_attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = self.norm1(h + self.dropout1(attn_out))
        ffn_out = self.ffn(h)
        h = self.norm2(h + self.dropout2(ffn_out))
        return h

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, mask=mask)
        return h

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, q_canon: torch.Tensor, query_mask: torch.Tensor, source_mask: torch.Tensor):
        '''
        Following the standard implementation of Attention is All You Need by Vaswani et. al.
        '''
        self_kpm = ~query_mask if query_mask is not None else None
        attn_out, _ = self.self_attn(q_canon, q_canon, q_canon, key_padding_mask=self_kpm, need_weights=False)
        q_canon = self.norm1(q_canon + self.dropout1(attn_out))
 
        cross_kpm = ~source_mask if source_mask is not None else None
        cross_out, _ = self.cross_attn(q_canon, z, z, key_padding_mask=cross_kpm, need_weights=False)
        q_canon = self.norm2(q_canon + self.dropout2(cross_out))
 
        ffn_out = self.ffn(q_canon)
        q_canon = self.norm3(q_canon + self.dropout3(ffn_out))
        return q_canon
        
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_length: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.canonical_query_library = nn.Embedding(num_embeddings=max_length, embedding_dim=d_model)
    
    def forward( # CEK LAGI COK BAGIAN INI SALAH, MASK PADDING QUERY SALAH!
        self,
        z: torch.Tensor, # B, L, d_model SATU VIEW
        l_star: torch.Tensor, # B (skalar), KALO TRAINING
        source_mask: torch.Tensor, # B, L, d_model
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = z.shape[0]
        device = z.device

        l_star = l_star.to(device=device, dtype=torch.long)
        l_star_max = int(l_star.max().item())

        query_length = torch.arange(l_star_max, device=device)
        query_canonical = self.canonical_query_library(query_length)
        query_canonical = query_canonical.unsqueeze(0).expand(B, -1, -1)

        pe = sinusoidal_PE(l_star_max, self.d_model, device=device)  # (L*_max, d)
        pe = pe.unsqueeze(0).expand(B, -1, -1).clone() # broadcast to batch size lah

        query_canonical = query_canonical + pe
 
        query_mask = length_to_mask(l_star, l_star_max)  # (B, L*_max) bool
        query_canonical = query_canonical * query_mask.unsqueeze(-1)
 
        for layer in self.layers:
            query_canonical = layer(z, query_canonical, query_mask=query_mask, source_mask=source_mask)

        return query_canonical, query_mask
    
class MaskedAttentionPooling(nn.Module):
    def __init__(self, d_model, hidden: int = 128):
        super(MaskedAttentionPooling, self).__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(z).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (z * weights).sum(dim=1)

class LengthPredictor(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1, eps: float = 1e-6):
        super(LengthPredictor, self).__init__()
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
    
    def forward(self, z: torch.Tensor, mask: torch.Tensor, detach_input: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if detach_input:
            z = z.detach()
        
        pooled = self.pool(z, mask)

        lengths = mask.sum(dim=1).float().clamp(min=1.0)
        log_len = torch.log(lengths).unsqueeze(-1)

        feat = torch.cat([pooled, log_len], dim = -1)
        raw = self.head(feat).squeeze(-1)

        l_pred = F.softplus(raw) + self.eps
        log_l_pred = torch.log(l_pred)
        
        return l_pred, log_l_pred
    
class TLeJEPA(nn.Module):
    def __init__(
        self,
        n_vocab_text: int,
        n_vocab_phoneme: int,
        d_model: int = 256,
        n_attn_heads: int = 8,
        enc_layers: int = 6,
        dec_layers: int = 4,
        max_length: int = 4096,
        dropout: float = 0.1
    ):
        super(TLeJEPA, self).__init__()
        self.d_model = d_model
        self.text_embedding = nn.Embedding(num_embeddings=n_vocab_text, embedding_dim=d_model)
        self.phoneme_embedding = nn.Embedding(num_embeddings=n_vocab_phoneme, embedding_dim=d_model)
        self.encoder = TransformerEncoder(d_model = d_model, n_heads = n_attn_heads, d_ff = 4 * d_model, n_layers = enc_layers, dropout = dropout)
        self.decoder = TransformerDecoder(d_model = d_model, n_heads = n_attn_heads, d_ff = 4 * d_model, n_layers = dec_layers, max_length = max_length, dropout = dropout)
        self.length_predictor = LengthPredictor(d_model = d_model)
    
    def _embed(self, ids: torch.Tensor, use_phoneme: bool) -> torch.Tensor:
        emb = self.phoneme_embedding(ids) if use_phoneme else self.text_embedding(ids)
        pe = sinusoidal_PE(ids.shape[1], self.d_model, device = ids.device)
        return emb + pe.unsqueeze(0)
    
    def train_forward(self, x: dict, type: str = "phoneme") -> dict:
        assert type in ("phoneme", "text"), "type must be either text or phoneme"
        use_phoneme = (type == "phoneme")
        # 1. ENCODE JADI Z1, Z2, ..., ZV
        # 1.1. Encode view utama dulu
        mask_1   = x["mask"][:, 0, :] # mask untuk padding di view utama (view pertama)
        h0_1     = self._embed(x["x"][:, 0, :], use_phoneme=use_phoneme) # embedding + positional encoding untuk view utama
        z_1      = self.encoder(h=h0_1, mask=mask_1)
        l_1      = mask_1.sum(dim=1).float().clamp(min=1.0)
        l_1_pred, _ = self.length_predictor(z_1, mask_1)
    
        n_views_aug = x["x"].shape[1] - 1
        zs, masks, l_preds, l_gt = [], [], [], []

        # 1.2. Encode view yang lain
        for i in range(n_views_aug):
            mask_i   = x["mask"][:, i + 1, :]
            h0_i     = self._embed(x["x"][:, i + 1, :], use_phoneme=False) # embedding + positional encoding untuk view tambahan
            z_i      = self.encoder(h=h0_i, mask=mask_i)
            l_i_pred, _ = self.length_predictor(z_i, mask_i)
            zs.append(z_i)
            masks.append(mask_i)
            l_preds.append(l_i_pred)
            l_gt.append(l_1)
        
        # 1.3. Jadiin satu lewat stack, tujuannya semua view ini termasuk kanonik diterusin ke decoder, jadi decoder tau bentuk kanonik seperti apa.
        z_v = torch.stack([z_1] + zs, dim = 1) # Kumpulkan z1, z2, ..., zV jadi satu tensor, tidak melupakan maskingnya (B, V, L*, d_model)
        l_v_preds = torch.stack([l_1_pred] + l_preds, dim=1) # B, V
        l_v_gts = torch.stack([l_1] + l_gt, dim=1) # B, V
        masks = torch.stack([mask_1] + masks, dim=1) # B, V, L*

        l_stars = l_v_gts[:, 0] # B, skalar, panjang dari view pertama (view utama), teacher forcing
        # print("DEBUG masks:\n", masks.sum(dim=-1), masks.shape)

        # 2. DECODE JADI ZC1, ZC2, ..., ZCV. Semua view dari kanonik sampai non-kanonik diteruskan ke decoder, buat prediksi gimana bentuk asli kanoniknya. Disini tantangan bagi decoder adalah gimana cara supaya memastikan dia konsisten, menerima bentuk kanonik, dan tidak mengubahnya.
        z_canon, masks_canon = [], []
        for i in range(n_views_aug + 1):
            z_i = z_v[:, i, :, :]
            mask_i = masks[:, i, :] # Pakai padding mask dari bentuk kanonik untuk mendukung teacher forcing.
            # print("Z_i shape:", z_i.shape, ", mask_i shape:", mask_i.shape)
            z_canon_i, mask_canon_i = self.decoder(z=z_i, l_star=l_stars, source_mask=mask_i)
            z_canon.append(z_canon_i)
            masks_canon.append(mask_canon_i)
        
        z_v_canon = torch.stack(z_canon, dim=1) # B, V, L*_max, d_model
        masks_v_canon = torch.stack(masks_canon, dim=1) # B, V, L*_max

        return {
            "z_v": z_v,
            "masks": masks,
            "l_v_preds": l_v_preds,
            "l_v_gts": l_v_gts,
            "z_v_canon": z_v_canon,
            "masks_v_canon": masks_v_canon
        }

    def forward(self, text, use_phoneme: bool = False) -> dict:
        # text diasumsikan tensor of token atau list of token tensor (panjang beda-beda)
        if type(text) == list:
            z_canons, l_preds = [], []
            for t in text:
                dummy_mask = torch.arange(0, t.shape[0], device=t.device) < t.shape[0]
                h = self._embed(t.unsqueeze(0), use_phoneme=use_phoneme)
                z = self.encoder(h, mask=None)
                l_hat, _ = self.length_predictor(z, dummy_mask.unsqueeze(0))
                l_star = l_hat.round().long().clamp(min=1)
                zc, _ = self.decoder(z, l_star, source_mask=None)
                z_canons.append(zc)
                l_preds.append(l_star)
            
            return z_canons, l_preds
        
        else:
            dummy_mask = torch.arange(0, text.shape[0], device=text.device) < text.shape[0]
            h = self._embed(text.unsqueeze(0), use_phoneme=use_phoneme)
            z = self.encoder(h, mask=None)
            l_hat, _ = self.length_predictor(z, dummy_mask.unsqueeze(0))
            l_star = l_hat.round().long().clamp(min=1)
            zc, _ = self.decoder(z, l_star, source_mask=None)
            
            return zc, l_star
