"""
Stage 2 losses: L = L_metric + lambda_shape * L_shape + lambda_colour * L_colour (the project plan), with L_metric
either ArcFace or batch-hard triplet (experiment E4 compares them).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """Additive angular margin loss (Deng et al., 2019) over the training labels; embeddings must be unit length."""

    def __init__(self, embedding_dim, num_classes, margin=0.3, scale=32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin, self.scale = margin, scale

    def forward(self, embeddings, labels):
        cos = F.linear(embeddings, F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos.gather(1, labels[:, None])) + self.margin)
        logits = cos.scatter(1, labels[:, None], target) * self.scale
        return F.cross_entropy(logits.float(), labels)


def batch_hard_triplet(embeddings, labels, margin=0.2):
    """Batch-hard triplet loss (Hermans et al., 2017) on cosine distance; needs P x K batches."""
    dist = 1 - embeddings @ embeddings.T
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    hardest_pos = dist.masked_fill(~same | eye, -math.inf).amax(dim=1)
    hardest_neg = dist.masked_fill(same, math.inf).amin(dim=1)
    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    return F.relu(hardest_pos - hardest_neg + margin)[valid].mean()


def auxiliary_losses(outputs, shape, colour):
    """Shape: cross-entropy, ignoring unlabelled (-1). Colour: binary cross-entropy (two-tone pills have 2 colours)."""
    losses = {}
    if "shape_logits" in outputs:
        losses["shape"] = F.cross_entropy(outputs["shape_logits"].float(), shape, ignore_index=-1)
    if "colour_logits" in outputs:
        losses["colour"] = F.binary_cross_entropy_with_logits(outputs["colour_logits"].float(), colour)
    return losses
