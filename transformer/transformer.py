import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------
# Position Embedding
# -----------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# -----------------------------------------------------
# Multi-Head Attention
# -----------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        def shape(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = shape(self.q(query))  # [B, h, T, d]
        K = shape(self.k(key))
        V = shape(self.v(value))

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V  # [B, h, T, d]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out(out)


# -----------------------------------------------------
# Feed Forward layer
# -----------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# -----------------------------------------------------
# Encoder Layer
# -----------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = x + self.dropout(self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), src_mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


# -----------------------------------------------------
# Decoder Layer
# -----------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        x = x + self.dropout(self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.ln2(x), enc_out, enc_out, memory_mask))
        x = x + self.dropout(self.ff(self.ln3(x)))
        return x


# -----------------------------------------------------
# Full Transformer Model
# -----------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=4, d_ff=2048, dropout=0.1):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, tgt_len):
        mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        return mask  # [T, T]

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos(self.embed(src))
        tgt = self.pos(self.embed(tgt))

        for layer in self.encoder:
            src = layer(src, src_mask)

        for layer in self.decoder:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        return self.lm_head(tgt)
