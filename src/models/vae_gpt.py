import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT(nn.Module):
    def __init__(
        self,
        chem_dim: int,
        conc_dim: int,
        seq_len: int = 968,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.chem_dim = chem_dim
        self.conc_dim = conc_dim
        self.d_model = d_model

        # --- condition projections (2 tokens) ---
        self.chem_proj = nn.Linear(chem_dim, d_model)
        self.conc_proj = nn.Linear(conc_dim, d_model)

        # --- time-series token projection ---
        self.y_proj = nn.Linear(1, d_model)

        # --- start-of-sequence token (<BOS> for y) ---
        self.y_start_emb = nn.Parameter(torch.randn(1, d_model) * 0.01)

        # positions: 2 cond + seq_len y tokens
        self.pos_emb = nn.Parameter(torch.randn(seq_len + 2, d_model) * 0.01)

        # --- transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # --- output ---
        self.output_proj = nn.Linear(d_model, 1)

    def _generate_causal_mask(self, T: int, device):
        """
        Causal mask where BOTH conditional tokens are visible to all positions.
        Token order:
            0: chem_token
            1: conc_token
            2: y_start
            3+: y_0, y_1, ...
        """
        mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        mask[:, :2] = 0  # chem + conc visible to all
        return mask

    def forward(self, c_chem: torch.Tensor, c_conc: torch.Tensor, y_input: torch.Tensor):
        """
        c_chem: [B, chem_dim]
        c_conc: [B, conc_dim]
        y_input: [B, T]  where T <= seq_len
        """
        B, T = y_input.shape

        # --- condition tokens ---
        chem_token = self.chem_proj(c_chem).unsqueeze(1)  # [B, 1, d_model]
        conc_token = self.conc_proj(c_conc).unsqueeze(1)  # [B, 1, d_model]

        # --- project y ---
        y_tokens = self.y_proj(y_input.unsqueeze(-1))     # [B, T, d_model]

        # --- shift right and insert y_start ---
        y_start = self.y_start_emb.unsqueeze(0).expand(B, -1, -1)  # [B, 1, d_model]
        if T > 0:
            y_tokens = torch.cat([y_start, y_tokens[:, :-1, :]], dim=1)
        else:
            y_tokens = y_start

        # --- full sequence ---
        x = torch.cat([chem_token, conc_token, y_tokens], dim=1)  # [B, T+2, d_model]

        # --- positional encoding ---
        pos = self.pos_emb[: x.size(1)]
        x = x + pos.unsqueeze(0)

        # --- transformer ---
        mask = self._generate_causal_mask(x.size(1), x.device)
        x = self.transformer(x, mask=mask)
        x = self.norm(x)

        # --- drop condition tokens ---
        x = x[:, 2:, :]  # [B, T, d_model]

        # --- output ---
        out = self.output_proj(x).squeeze(-1)  # [B, T]
        return out

    @torch.no_grad()
    def generate(self, desc, conc, max_len=None):
        self.eval()
        device = desc.device
        batch_size = desc.size(0)
        max_len = max_len or self.seq_len

        y_input = torch.zeros(batch_size, 0, device=device)

        generated_tokens = []

        for _ in range(max_len):
            y_hat = self(desc, conc, y_input)  # [B, seq_len_current]

            next_token = y_hat[:, -1].unsqueeze(1)  # [B, 1]

            y_input = torch.cat([y_input, next_token], dim=1)

            generated_tokens.append(next_token)

        generated_seq = torch.cat(generated_tokens, dim=1)

        return generated_seq


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor):
    return F.smooth_l1_loss(recon_x, x, reduction='mean')
