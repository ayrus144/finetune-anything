import torch.nn as nn
from .losses import CustomLoss, MultiClassDiceLoss, MultiClassFocalLoss

AVAI_LOSS = {'ce': nn.CrossEntropyLoss, 'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,
             'test_custom': CustomLoss, 'mse': nn.MSELoss,
             'dice_loss': MultiClassDiceLoss, 'focal_loss': MultiClassFocalLoss}


def get_losses(losses, num_classes):
    loss_dict = {}
    for name in losses:
        assert name in AVAI_LOSS, print('{name} is not supported, please implement it first.'.format(name=name))
        if losses[name].params is not None:
            loss_dict[name] = AVAI_LOSS[name](num_classes=num_classes, **losses[name].params)
        else:
            loss_dict[name] = AVAI_LOSS[name](num_classes=num_classes)
    return loss_dict
