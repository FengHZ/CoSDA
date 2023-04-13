import torch
from typing import Tuple
from model.resnetda import ResNetBackBone, ResNetClassifier
import torch.nn as nn
import torch.nn.functional as F
from train.nrc.nrc import build_banks


def get_losses(backbone: ResNetBackBone, classifier: ResNetClassifier,
               image: torch.Tensor, feature_bank: torch.Tensor, score_bank: torch.Tensor,
               idx: int, k=4, alpha=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_t = backbone(image)
    output_t = classifier(feature_t)
    output_softmax = nn.Softmax(dim=1)(output_t)
    distance = feature_bank[idx] @ feature_bank.T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=k + 1)
    idx_near = idx_near[:, 1:]
    score_near = score_bank[idx_near]
    sample_weight = score_near.size(1)
    pseudo_la = torch.mean(score_near, dim=1)
    l_a = torch.mean(torch.sum(F.kl_div(output_softmax, pseudo_la, reduction='none'), dim=1) * sample_weight)
    batch_size = image.size(0)
    mask = (torch.ones(batch_size, batch_size) - torch.eye(batch_size)).cuda()
    dot = output_softmax @ output_softmax.T
    dot_masked = dot * mask
    l_d = alpha * torch.mean(torch.sum(dot_masked, dim=-1))

    return l_a, l_d


def aad_train(train_dloader, backbone, classifier, backbone_optimizer,
              classifier_optimizer, batch_per_epoch,
              bottleneck_dim=256, num_classes=65, k=6, alpha=1.0, preprocess=None):
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
        l_a, l_d = get_losses(backbone, classifier, image,
                              feature_bank, score_bank, idx, k, alpha)
        task_loss_t = l_a + l_d
        task_loss_t.backward()
        backbone_optimizer.step()
        classifier_optimizer.step()


def alpha_decay(alpha_in, gamma, epoch):
    alpha_out = alpha_in * (gamma ** epoch)
    return alpha_out
