"""From-scratch LSTM/CNN/TCN/Transformer. Each takes (batch, seq_len) vocab ids
(0=pad) and returns (batch, n_classes) logits - same interface for all four."""
import math

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Unidirectional LSTM; classifies from the hidden state at each sequence's
    last real (non-pad) position. Skips pack_padded_sequence on purpose - it
    measured ~10x slower here on MPS, and gathering the last valid position
    from the full padded output is exact anyway (forward recurrence can't see
    trailing padding)."""

    def __init__(self, vocab_size, n_classes, embed_dim=100, hidden_dim=128,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1)
        emb = self.embedding(x)
        output, _ = self.lstm(emb)  # (B, L, hidden_dim)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.size(2))
        last_hidden = output.gather(1, idx).squeeze(1)
        return self.fc(self.dropout(last_hidden))


class CNNClassifier(nn.Module):
    """Parallel 1D convolutions (filter sizes = local n-gram widths) + max-over-time pooling."""

    def __init__(self, vocab_size, n_classes, embed_dim=100, num_filters=100,
                 filter_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), n_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (B, embed_dim, L); pad -> zero vector
        pooled = [torch.relu(conv(emb)).amax(dim=2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class _TCNBlock(nn.Module):
    """Causal, dilated conv pair with a residual connection (Bai et al. 2018 style)."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.chomp = pad
        self.dropout = nn.Dropout(dropout)

    def _causal_conv(self, x, conv):
        out = conv(x)
        return out[:, :, :-self.chomp] if self.chomp else out

    def forward(self, x):
        out = self.dropout(torch.relu(self._causal_conv(x, self.conv1)))
        out = self.dropout(torch.relu(self._causal_conv(out, self.conv2)))
        return torch.relu(out + x)


class TCNClassifier(nn.Module):
    """Stack of causal dilated residual blocks; receptive field grows as 2**levels."""

    def __init__(self, vocab_size, n_classes, embed_dim=100, channels=100,
                 levels=4, kernel_size=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_proj = nn.Conv1d(embed_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            _TCNBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout) for i in range(levels)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, n_classes)
        # geometric sum of per-block growth, each block has two causal convs of the same dilation
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** levels - 1)

    def forward(self, x):
        mask = (x != 0).unsqueeze(1).float()  # (B, 1, L)
        h = self.input_proj(self.embedding(x).transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        pooled = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        return self.fc(self.dropout(pooled))


class _SelfAttention(nn.Module):
    """Multi-head self-attention that also returns its weights, for the attention analysis."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_mask):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, Dh)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.out(out), attn


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.attn = _SelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out, attn_weights = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights


class TransformerClassifier(nn.Module):
    """Small from-scratch encoder: token + learned positional embeddings, self-attention."""

    def __init__(self, vocab_size, n_classes, max_len, d_model=100, n_heads=4,
                 ff_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, return_attn=False):
        mask = x != 0  # (B, L) bool, True = real token
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        attn_maps = []
        for block in self.blocks:
            h, attn = block(h, mask)
            attn_maps.append(attn)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        logits = self.fc(self.dropout(pooled))
        return (logits, attn_maps) if return_attn else logits
