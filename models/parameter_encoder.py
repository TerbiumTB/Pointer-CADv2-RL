import torch
import torch.nn as nn



class GatedMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)  # 生成门控信号
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        x = self.fc2(self.dropout(self.act(self.fc1(x))))
        return self.norm(gate * x)


class ParameterEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_freqs=16, mlp_hidden=64):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for RoPE"
        self.embed_dim = embed_dim

        # ===== Fourier & Raw 编码部分 =====
        self.register_buffer("freq_bands", 2 ** torch.linspace(0, num_freqs - 1, num_freqs))
        self.mlp_fourier = GatedMLPBlock(num_freqs * 2, mlp_hidden, embed_dim)
        self.mlp_raw = GatedMLPBlock(1, mlp_hidden, embed_dim)

        self.freq_scale = nn.Parameter(torch.ones(1))
        self.rope_base = nn.Parameter(torch.tensor(1000.0))

        self.norm = nn.LayerNorm(embed_dim)

    # ========= 数值编码 =========
    def encode_scalar_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(1)
        x_log = torch.sign(x) * torch.log1p(torch.abs(x))

        x_log = x_log / (x_log.abs().max() + 1e-6)
        xb = x_log * self.freq_scale * self.freq_bands.to(x.device)
        f = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
        emb_fourier = self.mlp_fourier(f)
        emb_raw = self.mlp_raw(x_log)
        emb = 0.5 * (emb_fourier + emb_raw)
        return self.norm(emb)

    # ========= 旋转位置编码 (RoPE) =========
    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D]
        positions: [N] 位置id
        """
        D = x.shape[-1]
        half = D // 2
        freqs = torch.arange(half, device=x.device).float() / half
        inv_freq = self.rope_base ** (-freqs)
        theta = positions[:, None] * inv_freq[None, :]  # [N, half]
        sin, cos = torch.sin(theta), torch.cos(theta)

        x1, x2 = x[..., :half], x[..., half:]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

    def dict_process(self, param_dict: dict) -> dict[str, torch.Tensor]:
        """
        param_dict:
            {
                "length": [..] or Tensor[..],
                "angle":  [..] or Tensor[..],
            }
        """
        result = {}
        for key, vals in param_dict.items():
            if not isinstance(vals, torch.Tensor):
                vals = torch.tensor(vals, dtype=torch.float32)

            if vals.numel() == 0:
                result[key] = torch.zeros((0, self.embed_dim), device=vals.device, dtype=torch.float32)
                continue

            emb = self.encode_scalar_tensor(vals)
            positions = torch.arange(len(vals), device=emb.device, dtype=torch.float32)

            emb = self.apply_rope(emb, positions)
            result[key] = emb
        return result

    def forward(self, param):
        if isinstance(param, dict):
            return self.dict_process(param)
        elif isinstance(param, list):
            return [self.dict_process(x) for x in param if isinstance(x, dict)]
        else:
            raise NotImplementedError


if __name__ == "__main__":
    encoder = ParameterEncoder(embed_dim=128, num_freqs=16, mlp_hidden=64)
    length = torch.tensor([10.0, 20.0, 30.0, 40.0])
    angle = torch.tensor([0.0, 45.0, 90.0, 180.0])
    param_dict = {
        "length": length,
        "angle": angle,
    }
    emb_dict = encoder([param_dict])
    print(emb_dict[0]["length"].shape)  # Expected: [4, 128]