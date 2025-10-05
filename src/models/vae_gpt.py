import torch
import torch.nn as nn

class ConditionalAutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int = 968,
        feature_dim: int = 41,
        d_model: int = 16,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.25
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Эмбеддинг для скалярного значения временного ряда
        self.value_emb = nn.Linear(1, d_model)

        # Эмбеддинг условия (features)
        self.cond_emb = nn.Linear(feature_dim, d_model)

        # Обучаемые позиционные эмбеддинги
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # Transformer decoder (на самом деле encoder с causal mask = decoder-only)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # стабильнее при обучении
            ) for _ in range(num_layers)
        ])

        # Выходной слой: d_model → 1 (скаляр)
        self.output_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        # Хорошая практика для Transformer
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, features: torch.Tensor):
        """
        Args:
            x: (B, L) — частично сгенерированный временной ряд (L <= seq_len)
            features: (B, feature_dim) — условие
        Returns:
            logits: (B, L, 1) — предсказания для каждого шага
        """
        B, L = x.shape
        assert L <= self.seq_len, f"Input length {L} > max {self.seq_len}"

        # Эмбеддинг значений
        x = x.unsqueeze(-1)  # (B, L, 1)
        x_emb = self.value_emb(x)  # (B, L, d_model)

        # Позиционные эмбеддинги
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # (B, L)
        pos_emb = self.pos_emb(pos_ids)  # (B, L, d_model)

        # Условие: повторяем для каждого шага
        cond_emb = self.cond_emb(features).unsqueeze(1)  # (B, 1, d_model)
        cond_emb = cond_emb.expand(-1, L, -1)  # (B, L, d_model)

        # Суммируем всё
        emb = x_emb + pos_emb + cond_emb  # (B, L, d_model)

        # Causal mask (чтобы не видеть будущее)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        # Проход через слои
        for layer in self.layers:
            emb = layer(emb, src_mask=mask)

        # Выход
        out = self.output_proj(emb)  # (B, L, 1)
        return out.squeeze(-1)  # (B, L)