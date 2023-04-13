import torch


def get_identity_model(**kwargs):
    model = torch.nn.Identity()
    return model
