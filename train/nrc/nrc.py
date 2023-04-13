import torch
from torch.utils.data import DataLoader
from typing import Tuple
from model.resnetda import ResNetBackBone, ResNetClassifier
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def distill_from_nrc(feature_bank: torch.Tensor, score_bank: torch.Tensor,
                     idx: int, temperature=0.07, confidence_gate=0.9,
                     k=4, m=3, num_classes=65) -> Tuple[torch.Tensor, torch.Tensor]:
    """generate pseduo label using kNN. **Duplicated**

    Args:
        feature_bank (torch.Tensor): _description_
        score_bank (torch.Tensor): _description_
        idx (int): _description_
        temperature (float, optional): the temperature for distillation. Defaults to 0.07.
        confidence_gate (float, optional): used by mask to filter. Defaults to 0.9.
        k (int, optional): # of neighbors. Defaults to 4.
        m (int, optional): # of neighbor of neighbors Defaults to 3.
        num_classes (int, optional): the # of classes in a dataset. Defaults to 65.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: pseduo label and corresponding mask
    """
    distance = feature_bank[idx] @ feature_bank.T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=k + 1)
    idx_near = idx_near[:, 1:]
    score_near = score_bank[idx_near]

    feature_near = feature_bank[idx_near]
    feature_bank_reshape = feature_bank.unsqueeze(0).expand(feature_near.shape[0], -1, -1)
    distance2 = feature_near @ feature_bank_reshape.permute(0, 2, 1)
    _, idx_near_near = torch.topk(distance2, dim=-1, largest=True, k=m + 1)
    idx_near_near = idx_near_near[:, :, 1:]
    idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).cuda()
    match = (idx_near_near == idx_expanded).sum(-1).float()

    # weight = torch.ones_like(idx_near, dtype=torch.float32)
    weight = torch.where(match > 0., match, 0.1 * torch.ones_like(match))
    weight_expanded = weight.unsqueeze(-1).expand(-1, -1, num_classes).cuda()
    score_weighted = score_near * weight_expanded
    sum_of_weight = torch.sum(weight_expanded, dim=1)
    pseduo_label = torch.sum(score_weighted, dim=1) / sum_of_weight

    distillated_pseduo_label = torch.softmax(pseduo_label / temperature, dim=1)  # correspond to knowledge
    pseduo_mask = (torch.max(distillated_pseduo_label, dim=1)[
                       0] > confidence_gate).float().cuda()  # correspond to knowledge_mask

    return distillated_pseduo_label, pseduo_mask


def build_banks(train_dloader: DataLoader, bottleneck_dim: int, num_classes: int,
                backbone: ResNetBackBone, classifier: ResNetClassifier, preprocess=None) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """Build feature bank and score bank.

    Args:
        train_dloader (DataLoader): target dataloader 
        bottleneck_dim (int): 256
        num_classes (int): the # of classes
        backbone (_type_): network backbone
        classifier (_type_): network classifier
        preprocess: data augmentations for models.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: feature bank and score bank. 
    """
    backbone.eval()
    classifier.eval()
    num_samples = len(train_dloader.dataset)
    feature_bank = torch.zeros(num_samples, bottleneck_dim, dtype=torch.float32).cuda()
    score_bank = torch.zeros(num_samples, num_classes, dtype=torch.float32).cuda()
    with torch.no_grad():
        for item in train_dloader:
            image = item[0]
            idx = item[-1]
            image = image.cuda()
            with torch.no_grad():
                image = preprocess(image)
            feature = backbone(image)
            feature_norm = F.normalize(feature)
            feature_bank[idx] = feature_norm.detach().clone()

            score = classifier(feature)
            score_softmax = nn.Softmax(-1)(score)
            score_bank[idx] = score_softmax.detach().clone()
    return feature_bank, score_bank


def get_losses(backbone: ResNetBackBone, classifier: ResNetClassifier,
               image: torch.Tensor, feature_bank: torch.Tensor, score_bank: torch.Tensor,
               idx: int, k=4, m=3, num_classes=65, ema=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """get three loss components of NRC.

    Args:
        backbone (ResNetBackBone):
        classifier (ResNetClassifier): 
        image (torch.Tensor): image
        feature_bank (torch.Tensor): feature bank
        score_bank (torch.Tensor): score bank
        idx (int): _description_
        k (int, optional): # of neighbors. Defaults to 4.
        m (int, optional): # of neighbor of neighbors. Defaults to 3.
        num_classes (int, optional): number of classes. Defaults to 65.
        ema: (bool, optional): whether to update banks or not.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: L_N, L_E and L_div
    """
    feature_t = backbone(image)
    output_t = classifier(feature_t)
    output_softmax = torch.softmax(output_t, dim=1)
    with torch.no_grad():
        if ema:  # update feature bank and score bank.
            feature_bank[idx] = 0.8 * feature_bank[idx] + 0.2 * F.normalize(feature_t).detach().clone()
            score_bank[idx] = 0.8 * score_bank[idx] + 0.2 * output_softmax.detach().clone()
        else:
            feature_bank[idx] = F.normalize(feature_t).detach().clone()
            score_bank[idx] = output_softmax.detach().clone()
    # copy the output_softmax k times with Batch*k*Classes
    distance = feature_bank[idx] @ feature_bank.T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=k + 1)
    # find the k nearest index in memory bank, Batch*k
    idx_near = idx_near[:, 1:]
    # Batch*k*classes
    score_near = score_bank[idx_near]
    # Batch*1*1, the index of current datas
    idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).cuda()
    # Batch*k*dim
    feature_near = feature_bank[idx_near]
    # Batch*N*dim WTF we have to copy feature_bank Batch times?
    feature_bank_reshape = feature_bank.unsqueeze(0).expand(feature_near.shape[0], -1, -1)
    # Batch*k*N size matrix, [b,:,:] means the b-th sample's k neighbours' distance with N samples
    distance2 = feature_near @ feature_bank_reshape.permute(0, 2, 1)
    _, idx_near_near = torch.topk(distance2, dim=-1, largest=True, k=m + 1)
    # Batch*k*m, [b,:,:] means the b-th sample's k neighbours' m nearest samples
    idx_near_near = idx_near_near[:, :, 1:]
    # check whether the B data's k neighbours' m nearest samples containing the data itself.
    # match: Batch*k. match[b,k]=1 means the b-th data's k-th neighbour is the mutual neighbour
    match = (idx_near_near == idx_expanded).sum(-1).float()
    # weight: Batch*k the mutual neighbour with 1, others with 0.1
    weight = torch.where(match > 0., match, 0.1 * torch.ones_like(match))
    # use the mutual weight to calculate the inner product
    # For L_N
    sample_weight = torch.sum(weight, dim=1)
    neighbor_weight = weight / sample_weight.view(-1, 1)
    pseudo_ln = torch.einsum("bk,bkc->bc", neighbor_weight, score_near)
    l_n = torch.mean(torch.sum(F.kl_div(output_softmax, pseudo_ln, reduction='none'), dim=1) * sample_weight)    # For L_E
    # Batch*(km)*c
    score_near_near = score_bank[idx_near_near]
    score_near_near = score_near_near.contiguous().view(score_near_near.shape[0], -1, num_classes)
    # Batch*c
    pseudo_le = torch.mean(score_near_near, dim=1)
    l_e = torch.mean(torch.sum(F.kl_div(output_softmax, pseudo_le, reduction="none"), dim=1)) * 0.1 * k * m
    # For Regularization
    mean_score = torch.mean(output_softmax, dim=0)
    l_div = torch.sum(mean_score * torch.log(mean_score + 1e-5))
    return l_n, l_e, l_div


def nrc_train(train_dloader, backbone, classifier, backbone_optimizer, classifier_optimizer, batch_per_epoch,
              bottleneck_dim=256, num_classes=65, k=6, m=4, preprocess=None):
    feature_bank, score_bank = build_banks(train_dloader, bottleneck_dim, num_classes, backbone=backbone,
                                           classifier=classifier, preprocess=preprocess)
    backbone.train()
    classifier.train()
    for i, item in enumerate(train_dloader):  # change from (image, _) to item
        image = item[0]
        idx = item[-1]
        if i >= batch_per_epoch:
            break
        image = image.cuda()
        with torch.no_grad():
            image = preprocess(image)
        # reset grad
        backbone_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        l_n, l_e, l_div = get_losses(backbone, classifier, image,
                                     feature_bank, score_bank, idx, k, m, num_classes)
        task_loss_t = l_n + l_e + l_div
        task_loss_t.backward()
        backbone_optimizer.step()
        classifier_optimizer.step()


def nrc_train_ema(feature_bank, score_bank, train_dloader, backbone, classifier, backbone_optimizer,
                  classifier_optimizer, batch_per_epoch, num_classes=65, k=6, m=4, preprocess=None):
    """train_nrc, update feature_bank and score bank during executing `get_losses`.
        w/o build_banks(different from nrc_train)
    Args:
        refer to nrc_train.
    """
    backbone.train()
    classifier.train()
    for i, item in enumerate(train_dloader):  # change from (image, _) to item
        image = item[0]
        idx = item[-1]
        if i >= batch_per_epoch:
            break
        image = image.cuda()
        with torch.no_grad():
            image = preprocess(image)
        # reset grad
        backbone_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        l_n, l_e, l_div = get_losses(backbone, classifier, image,
                                     feature_bank, score_bank, idx, k, m, num_classes, ema=True)
        task_loss_t = l_n + l_e + l_div
        task_loss_t.backward()
        backbone_optimizer.step()
        classifier_optimizer.step()
