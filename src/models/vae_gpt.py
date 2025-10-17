import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT_emb(nn.Module):
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
        self.cond_dim = cond_dim
        self.d_model = d_model

        # === 1. Каждый признак — в отдельный токен
        self.cond_proj = nn.Linear(1, d_model)

        # === 2. Learnable embeddings для "заглушек" y_0 ... y_{T-1}
        self.y_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)

        # === 3. Позиционные эмбеддинги
        total_len = cond_dim + seq_len
        self.pos_emb = nn.Parameter(torch.randn(total_len, d_model) * 0.01)

        # === 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # === 5. Output
        self.output_proj = nn.Linear(d_model, 1)

        # === 6. Causal mask
        self.register_buffer("causal_mask", self._generate_causal_mask(total_len))

    def _generate_causal_mask(self, T):
        """
        float маска размером (T, T), где:
        - Условные токены (0 .. cond_dim-1) видны всем
        - y_i видит только cond + y_j, j <= i
        """
        mask = torch.full((T, T), float('-inf'))
        mask = torch.triu(mask, diagonal=1)  # запрещаем j > i

        # разрешаем всем видеть все cond токены
        mask[:, :self.cond_dim] = 0
        return mask  # shape [T, T]

    def forward(self, c):
        """
        c: [B, cond_dim]
        output: [B, seq_len]
        """
        B = c.size(0)

        # === 1. Преобразуем каждый признак в токен
        cond_tokens = self.cond_proj(c.unsqueeze(-1))  # [B, cond_dim, d_model]

        # === 2. y "заглушки"
        y_tokens = self.y_emb.unsqueeze(0).expand(B, -1, -1)  # [B, seq_len, d_model]

        # === 3. Объединение cond + y
        x = torch.cat([cond_tokens, y_tokens], dim=1)  # [B, cond_dim + seq_len, d_model]

        # === 4. Позиционное кодирование
        x = x + self.pos_emb.unsqueeze(0)  # [B, total_len, d_model]

        # === 5. Transformer + norm
        x = self.transformer(x, mask=self.causal_mask)
        x = self.norm(x)

        # === 6. Убираем cond токены
        x = x[:, self.cond_dim:, :]  # [B, seq_len, d_model]

        # === 7. Проекция
        out = self.output_proj(x).squeeze(-1)  # [B, seq_len]

        return out


class GPT(nn.Module):
    def __init__(
        self,
        cond_dim=42,
        seq_len=968,
        d_model=64,
        nhead=2,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.cond_dim = cond_dim
        self.d_model = d_model

        # === 1. Проекция признаков (условий) ===
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # === 2. Проекция значений y_t в эмбеддинги ===
        self.y_proj = nn.Linear(1, d_model)

        # === 3. Позиционные эмбеддинги (cond + y)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)

        # === 4. Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # === 5. Выход
        self.output_proj = nn.Linear(d_model, 1)

        # === 6. Каузальная маска
        self.register_buffer("causal_mask", self._generate_causal_mask(seq_len))

    def _generate_causal_mask(self, T):
        mask = torch.full((T, T), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        mask[:, 0] = 0  # cond доступен всем
        return mask

    def forward(self, c, y_input):
        """
        c:       [B, cond_dim]
        y_input: [B, seq_len]  — реальные y, сдвинутые на 1 шаг (y[:, :-1])
        """
        B = c.size(0)

        # === 1. Проекция условий
        cond_token = self.cond_proj(c).unsqueeze(1)  # [B, 1, d_model]

        # === 2. Эмбеддинг реальных y
        y_input = y_input.unsqueeze(-1)              # [B, seq_len, 1]
        y_tokens = self.y_proj(y_input)              # [B, seq_len, d_model]

        # === 3. Собираем вход: [COND, y_0, y_1, ..., y_T-1]
        x = torch.cat([cond_token, y_tokens], dim=1)  # [B, seq_len+1, d_model]

        # === 4. Добавляем позиционные эмбеддинги
        x = x + self.pos_emb.unsqueeze(0)  # [B, seq_len+1, d_model]

        # === 5. Transformer + нормализация
        x = self.transformer(x, mask=self.causal_mask)
        x = self.norm(x)

        # === 6. Отбрасываем cond токен
        x = x[:, 1:, :]  # [B, seq_len, d_model]

        # === 7. Проекция в скаляр
        out = self.output_proj(x).squeeze(-1)  # [B, seq_len]

        return out


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor):
    return F.smooth_l1_loss(recon_x, x, reduction='mean')
