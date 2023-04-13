from .edgemix import get_edgemix_model
from .identity import get_identity_model
from torch import nn


class DataPreprocess(nn.Module):
    def __init__(self, **kwargs):
        super(DataPreprocess, self).__init__()
        if kwargs["method"] == 'identity':
            model = get_identity_model(**kwargs)
        elif kwargs["method"] == 'edgemix':
            model = get_edgemix_model(**kwargs)
        else:
            raise ValueError('Unknown augment method: {}'.format(kwargs["method"]))
        self.augment_model = model

    def forward(self, x):
        return self.augment_model(x)
