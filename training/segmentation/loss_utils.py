import torch
import torch.nn.functional as F

def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probs * targets).sum(dim=(2,3))
    union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def loss_fn(logits, targets, alpha=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    dice = dice_loss(logits, targets)
    return alpha * bce + (1 - alpha) * dice