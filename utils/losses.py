import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, n_classes=2, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)                    # (B, C, W, H, D)
        targets_oh = F.one_hot(targets, self.n_classes)     # (B, W, H, D, C)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        inter = (probs * targets_oh).sum(dims)
        union = (probs + targets_oh).sum(dims)
        dice  = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class SupLoss(nn.Module):
    """CE + Dice on labeled data."""
    def __init__(self, n_classes=2):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss(n_classes)

    def forward(self, logits, targets):
        return (self.ce(logits, targets) + self.dice(logits, targets)) / 2


def entropy_loss_full(probs):
    """Mean Shannon entropy over all voxels: -sum_c p_c * log(p_c)."""
    return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()


def entropy_loss_masked(probs, mask):
    """
    Shannon entropy applied only on voxels where mask > 0.
    mask: (B, W, H, D) float tensor — 1 on disagreement voxels, 0 elsewhere.
    Returns scalar 0 if mask is entirely zero (no disagreement).
    """
    H = -(probs * torch.log(probs + 1e-8)).sum(dim=1)   # (B, W, H, D)
    n = mask.sum()
    if n == 0:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    return (H * mask).sum() / n
