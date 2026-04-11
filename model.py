import math

import torch
import torch.nn as nn


class BinarySuffixTransformer(nn.Module):
    """
    Encoder-only classifier matching Charton's encoder architecture.

    - Learned token + positional embeddings
    - N-layer TransformerEncoder (pre-norm, GELU)
    - Mean-pool over non-pad positions
    - Linear classification head -> 2^p classes
    """

    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        max_len=64,
        dropout=0.1,
        pad_id=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_scale = math.sqrt(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)

        h = self.token_embed(x) * self.embed_scale + self.pos_embed(positions)
        h = self.embed_dropout(h)

        pad_mask = (x == self.pad_id) if self.pad_id is not None else None
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        if pad_mask is not None:
            mask = (~pad_mask).unsqueeze(-1).float()
            h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        h = self.layer_norm(h)
        return self.classifier(h)

    def get_layer_representations(self, x):
        """Return per-layer pooled hidden states (for linear probes)."""
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.token_embed(x) * self.embed_scale + self.pos_embed(positions)

        pad_mask = (x == self.pad_id) if self.pad_id is not None else None
        if pad_mask is not None:
            mask = (~pad_mask).unsqueeze(-1).float()
        else:
            mask = torch.ones(B, L, 1, device=x.device)

        representations = []
        for layer in self.encoder.layers:
            h = layer(h, src_key_padding_mask=pad_mask)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            representations.append(pooled.detach())
        return representations


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
