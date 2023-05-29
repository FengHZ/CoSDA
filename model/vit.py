import timm
import torch
import torch.nn as nn

from model.resnetda import init_weights


class VitBackBone(nn.Module):

    def __init__(self,
                 name="vit_base_patch16_224",
                 pretrained=True,
                 bottleneck_dim=512,
                 mixed_precision=False):
        super().__init__()
        self.mixed_precision = mixed_precision
        self.encoder = timm.create_model(name, pretrained=pretrained)
        in_features = self.encoder.head.in_features
        self.encoder.reset_classifier(0)  # encoder.head = nn.Identity
        self.bottleneck = nn.Sequential()
        fc = nn.Linear(in_features, bottleneck_dim)
        fc.apply(init_weights)
        self.bottleneck.add_module("fc", fc)
        self.bottleneck.add_module("bn",
                                   nn.BatchNorm1d(bottleneck_dim, affine=True))

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            x = self.encoder(x)
            x = self.bottleneck(x)
        return x
