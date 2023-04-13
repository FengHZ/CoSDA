import torch.nn as nn
from .resnet import get_resnet
import torch
import torch.nn.utils.weight_norm as weightNorm

feature_dict = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class ResNetBackBone(nn.Module):
    def __init__(self, backbone, pretrained=True, data_parallel=True, bottleneck_dim=512,
                 channels_per_group=0, mask_embedding=False):
        super(ResNetBackBone, self).__init__()
        encoder = get_resnet(backbone, pretrained=pretrained, channels_per_group=channels_per_group)
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
        else:
            self.encoder = encoder
        self.bottleneck = nn.Sequential()
        fc = nn.Linear(feature_dict[backbone], bottleneck_dim)
        fc.apply(init_weights)
        self.bottleneck.add_module("fc", fc)
        if channels_per_group == 0:
            self.bottleneck.add_module("bn", nn.BatchNorm1d(bottleneck_dim, affine=True))
        else:
            self.bottleneck.add_module("gn", nn.GroupNorm(int(bottleneck_dim / channels_per_group), bottleneck_dim,
                                                          affine=True))
        if data_parallel:
            self.bottleneck = nn.DataParallel(self.bottleneck)
        if mask_embedding:
            self.mask_embedding = nn.Embedding(2, bottleneck_dim)
            if data_parallel:
                self.mask_embedding = nn.DataParallel(self.mask_embedding)
        else:
            self.mask_embedding = None

    def forward(self, x, **kwargs):
        feature = self.encoder(x)
        feature = torch.flatten(feature, 1)
        feature = self.bottleneck(feature)
        if self.mask_embedding is None:
            return feature
        elif kwargs["no_embedding"]:
            return feature
        else:
            ## kwargs["t"] : chose an embedding from mask_embedding (2, bottlneck_dim)
            ## kwargs["s"] : a big scale to generate a smooth, near-binary value from sigmoid
            t0 = torch.LongTensor([0]).cuda()
            mask_0 = torch.sigmoid(self.mask_embedding(t0) * kwargs["s"])
            if kwargs["t"] == 0:
                feature = feature * mask_0
            elif kwargs["t"] == 1:
                t1 = torch.LongTensor([1]).cuda()
                mask_1 = nn.Sigmoid()(self.mask_embedding(t1) * kwargs["s"])
                feature = feature * mask_1
            if kwargs["all_mask"]:
                t1 = torch.LongTensor([1]).cuda()
                mask_1 = nn.Sigmoid()(self.mask_embedding(t1) * kwargs["s"])
                feature0 = feature * mask_0
                feature1 = feature * mask_1
                return (feature0, feature1), (mask_0, mask_1)
            else:
                return feature, mask_0


class ResNetClassifier(nn.Module):
    def __init__(self, classes=126, data_parallel=True, bottleneck_dim=512):
        super(ResNetClassifier, self).__init__()
        linear = nn.Sequential()
        fc = weightNorm(nn.Linear(bottleneck_dim, classes), name="weight")
        fc.apply(init_weights)
        linear.add_module("fc", fc)
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, feature):
        feature = self.linear(feature)
        return feature
