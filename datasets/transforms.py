import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
from omegaconf.dictconfig import DictConfig
import torch.nn as nn

AVIAL_TRANSFORM = {'resize': T.Resize, 'to_tensor': T.ToTensor}

def get_transforms(transforms: DictConfig):
    T_list = []
    for t_name in transforms.keys():
        assert t_name in AVIAL_TRANSFORM, "{T_name} is not supported transform, please implement it and add it to " \
                                          "AVIAL_TRANSFORM first.".format(T_name=t_name)
        if transforms[t_name].params is not None:
            T_list.append(AVIAL_TRANSFORM[t_name](**transforms[t_name].params))
        else:
            T_list.append(AVIAL_TRANSFORM[t_name]())
    return T.Compose(T_list)


class CustomTransform(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class ComposePair:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ResizePair:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class ToTensorPair:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.pil_to_tensor(target).squeeze(0)
        # target is not converted to tensor with to_tensor() as it also normalizes to [0, 1]
        # Doesn't need to be type torch.tensor for most torchvision.transforms to work, PIL works fine
        return image, target

AVIAL_JOINT_TRANSFORM = {'resize': ResizePair, 'random_hflip': RandomHorizontalFlipPair, 'to_tensor': ToTensorPair}

def get_joint_transforms(joint_transforms: DictConfig):
    JT_list = []
    for jt_name in joint_transforms.keys():
        assert jt_name in AVIAL_JOINT_TRANSFORM, "{JT_name} is not supported transform, please implement it and add it to " \
                                          "AVIAL_JOINT_TRANSFORM first.".format(JT_name=jt_name)
        if joint_transforms[jt_name].params is not None:
            JT_list.append(AVIAL_JOINT_TRANSFORM[jt_name](**joint_transforms[jt_name].params))
        else:
            JT_list.append(AVIAL_JOINT_TRANSFORM[jt_name]())
    return ComposePair(JT_list)