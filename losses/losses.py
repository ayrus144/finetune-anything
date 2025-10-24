'''
@copyright ziqi-jin
You can create custom loss function in this file, then import the created loss in ./__init__.py and add the loss into AVAI_LOSS
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset9 import threshold

ALPHA = 0.8
GAMMA = 2


# example
class CustomLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, y):
        pass

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-6, ignore_classes=0, weight=None):
        """
        Multi-class Focal Loss using softmax.

        Args:
            num_classes: Number of classes - taken directly from config file
            ignore_classes: Class labels to be ignored - usually background(0)
            eps (float): Avoids division by zero.
        """
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_classes = ignore_classes if ignore_classes is not None else []
        self.weight = weight  # Optional: class-wise weighting

    def forward(self, inputs, targets, iou_inputs):
        """
        Args:
            inputs: (B, C, H, W) - raw logits
            targets: (B, H, W) - class indices

        Returns:
            loss: Scalar dice loss
        """
        # Apply softmax to get class probabilities
        inputs = F.softmax(inputs, dim=1)  # (B, C, H, W)

        # Clamp targets to valid range for one-hot (handles ignored class labels safely)
        targets_clamped = targets.clamp(0, self.num_classes - 1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes=self.num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute Dice score per class
        dims = (0, 2, 3)  # sum over batch, height, width
        intersection = (inputs * targets_one_hot).sum(dim=dims)  # (C,)
        cardinality = inputs.sum(dim=dims) + targets_one_hot.sum(dim=dims)  # (C,)
        dice_per_class = (2. * intersection + self.eps) / (cardinality + self.eps)  # (C,)

        # Apply class weights or inverse volume weighting
        if self.weight is not None:
            weight = torch.tensor(self.weight, dtype=inputs.dtype, device=inputs.device)
            dice_per_class = dice_per_class * weight
            return 1.0 - dice_per_class.mean()
        else:
            # Class weighted Dice loss
            target_volumes = targets_one_hot.sum(dim=dims)  # (C,)
            class_weights = 1.0 / (target_volumes + self.eps)  # (C,)
            dice_per_class = dice_per_class * class_weights

            # Mask out ignored classes from final average
            if self.ignore_classes:
                ignore = torch.tensor(self.ignore_classes, dtype=torch.long, device=inputs.device)
                mask = torch.ones(self.num_classes, dtype=torch.bool, device=inputs.device)
                mask[ignore] = False
                dice_per_class = dice_per_class[mask]
                class_weights = class_weights[mask]

            return 1.0 - (dice_per_class.sum() / class_weights.sum())


class MultiClassFocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=ALPHA, gamma=GAMMA, reduction='mean'):
        """
        Multi-class Focal Loss using softmax.

        Args:
            num_classes: Number of classes - taken directly from config file
            gamma (float): Focusing parameter.
            alpha (float, list or Tensor): Optional class-wise weighting (length = num_classes).
            reduction (str): 'mean', 'sum', or 'None'
        """
        super(MultiClassFocalLoss, self).__init__()
        self.num_classes = num_classes
        if isinstance(alpha, float) and num_classes > 1:
            # weight background less and foreground more
            if alpha > 0.5:
                self.alpha = [1-alpha] + [alpha] * (self.num_classes - 1)
            else:
                self.alpha = [1-ALPHA] + [ALPHA] * (self.num_classes - 1)
        else:
            self.alpha = alpha # list or float
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, iou_inputs):
        """
        inputs: (B, C, H, W) - raw logits
        targets: (B, H, W) - class indices

        Returns:
            loss: Scalar focal loss if reduction is 'mean' or 'sum', else per-pixel focal loss map.
        """
        # Convert logits to log probabilities
        log_probs = F.log_softmax(inputs, dim=1)  # (B, C, H, W)
        probs = torch.exp(log_probs)  # (B, C, H, W)

        # Gather probabilities and log-probabilities for true classes
        targets = targets.long()  # Ensure long type
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B, H, W)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B, H, W)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Handle alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, list):
                alpha = torch.tensor(self.alpha, dtype=inputs.dtype, device=inputs.device)
                at = alpha[targets]  # (B, H, W)
            else:
                at = torch.full_like(pt, fill_value=self.alpha)
            loss = -at * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape (B, H, W)
