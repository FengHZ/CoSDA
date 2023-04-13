import torch
from model.resnetda import ResNetBackBone, ResNetClassifier
from typing import Tuple
import torch.nn.functional as F
import numpy as np
from train.aad.aad import build_banks


def kd_aad_losses(backbone: ResNetBackBone, classifier: ResNetClassifier,
                  image: torch.Tensor, feature_bank: torch.Tensor, score_bank: torch.Tensor,
                  idx: int, k=4, ema=True, beta=2, alpha=1.0) -> Tuple[
    torch.Tensor, torch.Tensor]:
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
    sample_weight = score_near.size(1)
    # For l_a
    # Batch*c
    pseudo_la = torch.mean(score_near, dim=1)
    mixed_pseduo_la = lam * pseudo_la + (1 - lam) * pseudo_la[index, :]
    mixed_la = torch.mean(
        torch.sum(F.kl_div(mixed_output_softmax, mixed_pseduo_la, reduction='none'), dim=1)) * sample_weight
    # For l_d
    batch_size = image.size(0)
    mask = (torch.ones(batch_size, batch_size) - torch.eye(batch_size)).cuda()
    # Batch*Batch
    dot = alpha * mixed_output_softmax @ mixed_output_softmax.T
    mixed_dot_masked = dot * mask
    mixed_ld = torch.mean(torch.sum(mixed_dot_masked, dim=-1))
    return mixed_la, mixed_ld


def kd_aad_train(train_dloader, backbone, classifier, backbone_optimizer, classifier_optimizer, batch_per_epoch, beta=2,
                 bottleneck_dim=256, num_classes=65, k=6, feature_bank=None, score_bank=None,
                 alpha=1.0, preprocess=None):
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
        l_a, l_d = kd_aad_losses(backbone, classifier, image, feature_bank, score_bank, idx, k, True, beta, alpha)
        task_loss_t = l_a + l_d
        task_loss_t.backward()
        backbone_optimizer.step()
        classifier_optimizer.step()
