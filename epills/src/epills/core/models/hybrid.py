"""
The Stage 2 network: a PyTorch port of the FER model in FER/exp-tuning.ipynb (geometry branch left out).

    image (B, 3, 224, 224), RGB in [0, 1]
      ├─ VGG16 (ImageNet) ─► CBAM ─► global average pool ─► Dense 256, ReLU ──┐
      └─ ViT-B/16 (google/vit-base-patch16-224-in21k) ─► pooler output (768) ─┴─► Capsule Fusion Block (256)
                                                                                    ─► Transformer Refinement Block (64)
      ─► [+ measured size and colour, optional] ─► embedding head ─► 256-d, L2-normalised
                                                 ─► auxiliary heads: shape (softmax), colour (multi-label)

Ported as faithfully as the frameworks allow: same layer sizes and wiring, Keras initialisers (Glorot uniform,
zero bias), Keras batch-norm and layer-norm settings, and Keras-style multi-head attention (key_dim per head).
Differences from the FER notebook, all deliberate:
  - VGG16 uses torchvision's ImageNet weights, so it gets torchvision's normalisation (RGB, mean/std) instead of
    Keras' caffe preprocessing (BGR, mean only). Each set of weights gets the preprocessing it was trained with.
  - The 7-class softmax is replaced by the Stage 2 heads (embedding + auxiliary shape and colour).
  - refine_pool="concat" is an extra option (4 tokens x 64 = 256-d without the mean-pool bottleneck); the default
    "mean" is the FER behaviour.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VIT_CHECKPOINT = "google/vit-base-patch16-224-in21k"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def keras_dense(in_features, out_features):
    """nn.Linear initialised like a Keras Dense layer (Glorot uniform weights, zero bias)."""
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def keras_batch_norm(features):
    """BatchNorm1d with Keras defaults (momentum 0.99 -> PyTorch 0.01, epsilon 1e-3)."""
    return nn.BatchNorm1d(features, eps=1e-3, momentum=0.01)


class CBAM(nn.Module):
    """The FER notebook's attention_block: channel attention (separate avg and max MLPs), then spatial attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_dense = keras_dense(channels, channels // reduction)
        self.max_dense = keras_dense(channels, channels // reduction)
        self.channel_dense = keras_dense(channels // reduction, channels)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        nn.init.xavier_uniform_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)

    def forward(self, x):  # (B, C, H, W)
        avg = F.relu(self.avg_dense(x.mean(dim=(2, 3))))
        mx = F.relu(self.max_dense(x.amax(dim=(2, 3))))
        x = x * torch.sigmoid(self.channel_dense(avg + mx))[:, :, None, None]
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(pooled))


def squash(s, dim=-1, eps=1e-7):
    norm_sq = (s ** 2).sum(dim=dim, keepdim=True)
    return norm_sq / (1 + norm_sq) * s / torch.sqrt(norm_sq + eps)


class CapsuleFusionBlock(nn.Module):
    """AdvancedCapsuleFusionBlock: per-branch MLP + batch norm, then dynamic routing with a residual update."""

    def __init__(self, input_dims, capsule_dim=256, num_iterations=2, dropout_rate=0.4):
        super().__init__()
        self.num_iterations = num_iterations
        self.mlps = nn.ModuleList(
            nn.Sequential(keras_dense(d, capsule_dim), nn.ReLU(), nn.Dropout(dropout_rate), keras_dense(capsule_dim, capsule_dim))
            for d in input_dims)
        self.bns = nn.ModuleList(keras_batch_norm(capsule_dim) for _ in input_dims)

    def forward(self, features):
        u = torch.stack([bn(mlp(f)) for f, mlp, bn in zip(features, self.mlps, self.bns)], dim=1)  # (B, n, D)
        b = torch.zeros(u.shape[:2], device=u.device, dtype=u.dtype)
        v_prev = None
        for i in range(self.num_iterations):
            c = torch.softmax(b, dim=1)
            v = squash((c.unsqueeze(-1) * u).sum(dim=1))
            if i > 0:
                v = 0.5 * v + 0.5 * v_prev
            v_prev = v
            b = b + (u * v.unsqueeze(1)).sum(dim=-1)
        return v


class KerasMultiHeadAttention(nn.Module):
    """Self-attention like tf.keras MultiHeadAttention(num_heads, key_dim): every head has key_dim dimensions."""

    def __init__(self, token_dim, num_heads, key_dim):
        super().__init__()
        self.num_heads, self.key_dim = num_heads, key_dim
        self.q, self.k, self.v = (keras_dense(token_dim, num_heads * key_dim) for _ in range(3))
        self.out = keras_dense(num_heads * key_dim, token_dim)

    def forward(self, x):  # (B, T, D)
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.num_heads, self.key_dim).transpose(1, 2)  # (B, H, T, K)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.key_dim), dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, self.num_heads * self.key_dim))


class TransformerEncoderBlock(nn.Module):
    """Post-norm encoder block: attention + dropout + residual + layer norm, then FFN + dropout + residual + norm."""

    def __init__(self, token_dim, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.mha = KerasMultiHeadAttention(token_dim, num_heads, key_dim=token_dim)
        self.ffn = nn.Sequential(keras_dense(token_dim, ff_dim), nn.ReLU(), keras_dense(ff_dim, token_dim))
        self.norm1 = nn.LayerNorm(token_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(token_dim, eps=1e-6)
        self.drop1, self.drop2 = nn.Dropout(dropout_rate), nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.norm1(x + self.drop1(self.mha(x)))
        return self.norm2(x + self.drop2(self.ffn(x)))


class TransformerRefinementBlock(nn.Module):
    """AdvancedTransformerRefinementBlock: project to tokens, add learned positions, encode, then pool."""

    def __init__(self, in_dim=256, num_tokens=4, token_dim=64, num_heads=4, ff_dim=128, num_layers=2,
                 dropout_rate=0.4, pool="mean"):
        super().__init__()
        self.num_tokens, self.token_dim, self.pool = num_tokens, token_dim, pool
        self.proj = keras_dense(in_dim, num_tokens * token_dim)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens, token_dim).uniform_(-0.05, 0.05))  # Keras "uniform"
        self.layers = nn.ModuleList(TransformerEncoderBlock(token_dim, num_heads, ff_dim, dropout_rate)
                                    for _ in range(num_layers))
        self.out_dim = token_dim if pool == "mean" else num_tokens * token_dim

    def forward(self, x):
        tokens = self.proj(x).view(-1, self.num_tokens, self.token_dim) + self.pos_embedding
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens.mean(dim=1) if self.pool == "mean" else tokens.flatten(1)


class HybridPillNet(nn.Module):
    """
    VGG16+CBAM and ViT-B/16, fused by the Capsule Fusion Block and refined by the Transformer Refinement Block,
    with Stage 2 heads. forward() returns a dict: embedding (L2-normalised), shape_logits, colour_logits.
    pretrained=False builds the same network with random weights (tests, or before loading a checkpoint).
    """

    def __init__(self, embedding_dim=256, num_shapes=0, num_colours=0, num_measured=0, refine_pool="mean",
                 pretrained=True, vit_checkpoint=VIT_CHECKPOINT):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        from transformers import ViTConfig, ViTModel

        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features  # (B, 512, 7, 7)
        self.cbam = CBAM(512)
        self.vgg_dense = keras_dense(512, 256)
        self.vit = (ViTModel.from_pretrained(vit_checkpoint) if pretrained
                    else ViTModel(ViTConfig.from_pretrained(vit_checkpoint)))
        vit_dim = self.vit.config.hidden_size
        self.fusion = CapsuleFusionBlock([256, vit_dim], capsule_dim=256, num_iterations=2, dropout_rate=0.4)
        self.refine = TransformerRefinementBlock(in_dim=256, pool=refine_pool)
        head_in = self.refine.out_dim
        self.measured = None
        if num_measured:
            self.measured = nn.Sequential(keras_dense(num_measured, 32), nn.ReLU())
            head_in += 32
        self.embedding_head = keras_dense(head_in, embedding_dim)
        self.shape_head = keras_dense(self.refine.out_dim, num_shapes) if num_shapes else None
        self.colour_head = keras_dense(self.refine.out_dim, num_colours) if num_colours else None
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.set_phase(1)

    def set_phase(self, phase):
        """
        The FER two-phase schedule. Phase 1: ViT and VGG16 blocks 1-3 frozen; VGG16 blocks 4-5, CBAM, fusion,
        refinement and heads train. Phase 2: everything trains.
        """
        for p in self.parameters():
            p.requires_grad = True
        if phase == 1:
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.vgg[:17].parameters():  # features[17] is block4_conv1
                p.requires_grad = False

    def features(self, images):
        """The refined fused feature (B, refine.out_dim) for RGB images in [0, 1]."""
        vgg = self.vgg((images - self.imagenet_mean) / self.imagenet_std)
        vgg = F.relu(self.vgg_dense(self.cbam(vgg).mean(dim=(2, 3))))
        vit = self.vit(pixel_values=(images - 0.5) / 0.5).pooler_output
        return self.refine(self.fusion([vgg, vit]))

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """A trained model from a best.pt written by scripts/train_stage2.py, in eval mode. Returns (model, ckpt)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        mcfg, vocab = ckpt["config"]["model"], ckpt["vocab"]
        model = cls(embedding_dim=mcfg["embedding_dim"], refine_pool=mcfg["refine_pool"],
                    num_shapes=len(vocab["shapes"]) if mcfg["aux_heads"] else 0,
                    num_colours=len(vocab["colours"]) if mcfg["aux_heads"] else 0, pretrained=False)
        model.load_state_dict(ckpt["model"])
        return model.to(device).eval(), ckpt

    def forward(self, images, measured=None):
        refined = self.features(images)
        head_in = refined
        if self.measured is not None:
            if measured is None:
                raise ValueError("this model was built with measured inputs; pass measured=")
            head_in = torch.cat([refined, self.measured(measured)], dim=1)
        out = {"embedding": F.normalize(self.embedding_head(head_in), dim=1)}
        if self.shape_head is not None:
            out["shape_logits"] = self.shape_head(refined)
        if self.colour_head is not None:
            out["colour_logits"] = self.colour_head(refined)
        return out
