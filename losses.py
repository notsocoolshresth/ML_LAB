"""
Loss functions for binary landslide segmentation.
Combines Dice Loss and Focal Loss to handle severe class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal loss.
    Dice handles region overlap, Focal handles pixel-level class imbalance.
    """

    def __init__(self, dice_weight=0.5, focal_weight=0.5, focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits, targets):
        d = self.dice_loss(logits, targets)
        f = self.focal_loss(logits, targets)
        return self.dice_weight * d + self.focal_weight * f


class TverskyLoss(nn.Module):
    """
    Tversky Loss — generalization of Dice that lets you weight FP vs FN.
    Setting alpha > beta penalizes FN more → higher recall.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets_flat = targets.view(-1)

        tp = (probs * targets_flat).sum()
        fp = (probs * (1 - targets_flat)).sum()
        fn = ((1 - probs) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class CombinedLoss(nn.Module):
    """
    Best-performing loss: Tversky + Focal.
    Tversky with alpha=0.3, beta=0.7 emphasizes recall (catching landslide pixels),
    Focal handles the overwhelming non-landslide background.
    """

    def __init__(self):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, logits, targets):
        return 0.5 * self.tversky(logits, targets) + 0.5 * self.focal(logits, targets)


def _to_seg_boundary(outputs):
    if isinstance(outputs, dict):
        return outputs.get("seg"), outputs.get("boundary")
    if isinstance(outputs, (tuple, list)):
        seg = outputs[0]
        boundary = outputs[1] if len(outputs) > 1 else None
        return seg, boundary
    return outputs, None


def compute_boundary_targets(targets):
    """Morphological gradient target for edge supervision."""
    dilated = F.max_pool2d(targets, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-targets, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


class BoundaryAwareLoss(nn.Module):
    """Segmentation loss plus optional boundary supervision."""

    def __init__(self, seg_loss=None, boundary_weight=0.2):
        super().__init__()
        self.seg_loss = seg_loss if seg_loss is not None else CombinedLoss()
        self.boundary_weight = boundary_weight

    def forward(self, outputs, targets):
        seg_logits, boundary_logits = _to_seg_boundary(outputs)
        loss = self.seg_loss(seg_logits, targets)

        if boundary_logits is not None:
            boundary_targets = compute_boundary_targets(targets)
            boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
            loss = loss + self.boundary_weight * boundary_loss

        return loss
