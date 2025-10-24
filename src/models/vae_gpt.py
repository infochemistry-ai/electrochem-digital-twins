import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.y_proj = nn.Linear(1, d_model)
        self.y_start_emb = nn.Parameter(torch.randn(1, d_model) * 0.01)
        self.pos_emb = nn.Parameter(torch.randn(seq_len + 1, d_model) * 0.01)

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

        self.output_proj = nn.Linear(d_model, 1)
        self.register_buffer("causal_mask", self._generate_causal_mask(seq_len + 1))

    def _generate_causal_mask(self, T):
        mask = torch.full((T, T), float('-inf'), device=self.pos_emb.device)
        mask = torch.triu(mask, diagonal=1)
        mask[:, 0] = 0  # cond доступен всем
        return mask

    def forward(self, c, y_input):
        B = c.size(0)
        cond_token = self.cond_proj(c).unsqueeze(1)  # [B, 1, d_model]

        y_input_expanded = y_input.unsqueeze(-1)        # [B, seq_len, 1]
        y_tokens = self.y_proj(y_input_expanded)        # [B, seq_len, d_model]

        y_start_emb_expanded = self.y_start_emb.unsqueeze(0).expand(B, -1, -1)  # [B, 1, d_model]

        # Вставляем y_start_emb как первый токен, а y_tokens сдвигаем вправо
        y_tokens = torch.cat([y_start_emb_expanded, y_tokens[:, :-1, :]], dim=1)  # [B, seq_len, d_model]

        x = torch.cat([cond_token, y_tokens], dim=1)  # [B, seq_len+1, d_model]

        pos_emb = self.pos_emb[:x.size(1), :]         # [seq_len+1, d_model]
        x = x + pos_emb.unsqueeze(0)                   # [B, seq_len+1, d_model]

        causal_mask = self._generate_causal_mask(x.size(1))  # [seq_len+1, seq_len+1]

        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)

        x = x[:, 1:, :]  # Отбросили cond токен

        out = self.output_proj(x).squeeze(-1)  # [B, seq_len]

        return out
    
    @torch.no_grad()
    def generate(self, c, max_len=None):
        self.eval()
        device = c.device
        batch_size = c.size(0)
        max_len = max_len or self.seq_len

        y_input = torch.zeros(batch_size, 0, device=device)

        generated_tokens = []

        for _ in range(max_len):
            y_hat = self(c, y_input)  # [B, seq_len_current]

            next_token = y_hat[:, -1].unsqueeze(1)  # [B, 1]

            y_input = torch.cat([y_input, next_token], dim=1)

            generated_tokens.append(next_token)

        generated_seq = torch.cat(generated_tokens, dim=1)

        return generated_seq


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor):
    return F.smooth_l1_loss(recon_x, x, reduction='mean')
