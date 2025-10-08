import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_GPT(nn.Module):
    def __init__(
        self,
        cond_dim=42,
        seq_len=968,
        d_model=32,
        nhead=2,
        num_layers=1,
        dropout=0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # === 1. Условный токен ===
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # === 2. Learnable embeddings для временных позиций (y_0 ... y_{T-1}) ===
        self.y_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)

        # === 3. Позиционное кодирование
        self.pos_emb = nn.Parameter(torch.randn(seq_len + 1, d_model) * 0.01)

        # === 4. Transformer decoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # чуть более стабильная схема
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)  # дополнительная нормализация

        # === 5. Выходной слой ===
        self.output_proj = nn.Linear(d_model, 1)

        # === 6. Маска ===
        self.register_buffer("causal_mask", self._generate_causal_mask(seq_len + 1))

    def _generate_causal_mask(self, T):
        """
        float маска размером (T, T), где:
        - COND токен (0) доступен всем
        - y_i видит только y_j, j <= i и COND
        """
        mask = torch.full((T, T), float('-inf'))
        mask = torch.triu(mask, diagonal=1)  # обычная causal mask: запрещаем j > i
        mask[:, 0] = 0  # разрешаем всем видеть COND (0-й токен)
        return mask  # shape [T, T]

    def forward(self, c):
        """
        c: [B, cond_dim]
        output: [B, seq_len]
        """
        B = c.size(0)

        # 1. Embed cond
        cond_emb = self.cond_proj(c)  # [B, d_model]

        # 2. Learnable y embeddings
        y_emb = self.y_emb.unsqueeze(0).expand(B, -1, -1)  # [B, seq_len, d_model]

        # 3. Full input sequence: [COND, y_0, ..., y_T-1]
        x = torch.cat([cond_emb.unsqueeze(1), y_emb], dim=1)  # [B, seq_len+1, d_model]

        # 4. Positional embeddings
        x = x + self.pos_emb.unsqueeze(0)  # [B, seq_len+1, d_model]

        # 5. Transformer
        x = self.transformer(x, mask=self.causal_mask)  # [B, seq_len+1, d_model]
        x = self.norm(x)  # [B, seq_len+1, d_model]

        # 6. Discard COND token
        x = x[:, 1:, :]  # [B, seq_len, d_model]

        # 7. Output projection
        out = self.output_proj(x).squeeze(-1)  # [B, seq_len]

        return out


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor):
    # Stable L1 loss
    return F.smooth_l1_loss(recon_x, x, reduction='mean')
