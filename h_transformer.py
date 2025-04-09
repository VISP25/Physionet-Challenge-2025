import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, t):
        # Move inv_freq to the same device as the input tensor 't'
        self.inv_freq = self.inv_freq.to(t.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb

class HAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj_out = nn.Linear(dim, dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).transpose(1, 2), qkv)

        # Apply Rotary Embeddings
        pos = self.rotary_emb(torch.arange(n, device=x.device).float())
        q, k = q * pos, k * pos

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj_out(out)

class HTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.attn = HAttention(dim, heads, dim_head)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

class HTransformer1D(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_channels):
        super().__init__()
        self.proj = nn.Linear(num_channels, dim)
        self.layers = nn.ModuleList([
            HTransformerBlock(dim, heads, dim_head, mlp_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class ECGHTransformer(nn.Module):
    def __init__(self, input_size=4096, num_channels=12, num_classes=2):
        super(ECGHTransformer, self).__init__()
        
        self.transformer = HTransformer1D(
            dim=768, depth=4, heads=4, dim_head=192, mlp_dim=2048, num_channels=num_channels
        )

        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # The input x is already moved to the correct device, no need to move it again here
        x = x.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, channels)
        transformer_output = self.transformer(x)
        pooled_output = transformer_output.mean(dim=1)  # Mean pooling over the sequence length
        output = self.classifier(pooled_output)
        return output


        
