import torch
from model.resnetda import ResNetBackBone, ResNetClassifier
from typing import Tuple
import torch.nn.functional as F
import numpy as np
from train.nrc.nrc import build_banks


def kd_nrc_losses(backbone: ResNetBackBone, classifier: ResNetClassifier,
                  image: torch.Tensor, feature_bank: torch.Tensor, score_bank: torch.Tensor,
                  idx: int, k=4, m=3, num_classes=65, ema=True, beta=2, temperature=0.01) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        feature_t = backbone(image)
        output_t = classifier(feature_t)
        output_softmax = torch.softmax(output_t, dim=1)
        if ema:  # update feature bank and score bank.
            feature_bank[idx] = 0.8 * feature_bank[idx] + 0.2 * F.normalize(feature_t).detach().clone()
            score_bank[idx] = 0.8 * score_bank[idx] + 0.2 * output_softmax.detach().clone()
        else:
            feature_bank[idx] = F.normalize(feature_t).detach().clone()
            score_bank[idx] = output_softmax.detach().clone()
    # prepare mixup
    if beta > 0:
        lam = np.random.beta(beta, beta)
        # set high lamb in the left
        lam = max(lam, 1 - lam)
    else:
        lam = 1
    batch_size = image.size(0)
    index = torch.randperm(batch_size).cuda()
    mixed_image = lam * image + (1 - lam) * image[index, :]
    mixed_output_t = classifier(backbone(mixed_image))
    mixed_output_softmax = torch.softmax(mixed_output_t, dim=1)
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
    pseudo_ln = torch.softmax(pseudo_ln / temperature, dim=1)
    mixed_pseudo_ln = lam * pseudo_ln + (1 - lam) * pseudo_ln[index, :]
    mixed_l_n = torch.mean(
        torch.sum(F.kl_div(mixed_output_softmax, mixed_pseudo_ln, reduction='none'), dim=1) * sample_weight)
    # For L_E
    # Batch*(km)*c
    score_near_near = score_bank[idx_near_near]
    score_near_near = score_near_near.contiguous().view(score_near_near.shape[0], -1, num_classes)
    # Batch*c
    pseudo_le = torch.mean(score_near_near, dim=1)
    pseudo_le = torch.softmax(pseudo_le / temperature, dim=1)
    mixed_pseudo_le = lam * pseudo_le + (1 - lam) * pseudo_le[index, :]
    mixed_l_e = torch.mean(
        torch.sum(F.kl_div(mixed_output_softmax, mixed_pseudo_le, reduction="none"), dim=1)) * 0.1 * k * m
    # For Regularization
    mean_score = torch.mean(mixed_output_softmax, dim=0)
    mixed_l_div = torch.sum(mean_score * torch.log(mean_score + 1e-5))
    return mixed_l_n, mixed_l_e, mixed_l_div


def kd_nrc_train(train_dloader, backbone, classifier, backbone_optimizer, classifier_optimizer, batch_per_epoch, beta=2,
                 temperature=0.01, bottleneck_dim=256, num_classes=65, k=6, m=4, feature_bank=None, score_bank=None,
                 preprocess=None):
    if feature_bank is None or score_bank is None:
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
        l_n, l_e, l_div = kd_nrc_losses(backbone, classifier, image, feature_bank, score_bank, idx, k, m, num_classes,
                                        ema=True, beta=beta, temperature=temperature)
        task_loss_t = l_n + l_e + l_div
        task_loss_t.backward()
        backbone_optimizer.step()
        classifier_optimizer.step()
