import numpy as np
from utils.avgmeter import AverageMeter
import torch


def distill_knowledge(score, confidence_gate, temperature=0.07):
    predict = torch.softmax(score, dim=1)
    # get the knowledge with weight and mask
    max_p, max_p_class = predict.max(1)
    knowledge_mask = (max_p > confidence_gate).float().cuda()
    knowledge = torch.softmax(score / temperature, dim=1)
    return knowledge, knowledge_mask


def cosda(train_dloader, teacher_backbone, teacher_classifier, student_backbone, student_classifier,
         backbone_optimizer, classifier_optimizer, batch_per_epoch, confidence_gate, beta=2,
         temperature=1, preprocess=None, reg_alpha=0.1, only_mi=False):
    utilized_ratio = AverageMeter()
    teacher_backbone.train()
    teacher_classifier.train()
    student_backbone.train()
    student_classifier.train()
    for i, (image_t, *_) in enumerate(train_dloader):
        if i >= batch_per_epoch:
            break
        image_t = image_t.cuda()
        with torch.no_grad():
            image_t = preprocess(image_t)
        # reset grad
        backbone_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        # use source model to do predict
        with torch.no_grad():
            score = teacher_classifier(teacher_backbone(image_t))
            knowledge, knowledge_mask = distill_knowledge(score, confidence_gate, temperature=temperature)
        if beta > 0:
            lam = np.random.beta(beta, beta)
            # set high lamb in the left
            lam = max(lam, 1 - lam)
        else:
            lam = 1
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_consensus = lam * knowledge + (1 - lam) * knowledge[index, :]
        mixed_output = student_classifier(student_backbone(mixed_image))
        mixed_log_softmax = torch.log_softmax(mixed_output, dim=1)
        consistency_loss = torch.sum(
            knowledge_mask * torch.sum(-1 * mixed_consensus * mixed_log_softmax, dim=1)) / torch.sum(
            knowledge_mask)
        # set regularization
        output = student_classifier(student_backbone(image_t))
        softmax_output = torch.softmax(output, dim=1)
        margin_output = torch.mean(softmax_output, dim=0)
        log_softmax_output = torch.log_softmax(output, dim=1)
        log_margin_output = torch.log(margin_output + 1e-5)
        mutual_info_loss = -1 * torch.mean(
            torch.sum(softmax_output * (log_softmax_output - log_margin_output), dim=1))
        if only_mi:
            task_loss = reg_alpha * mutual_info_loss
        else:
            task_loss = consistency_loss + reg_alpha * mutual_info_loss
        task_loss.backward()
        backbone_optimizer.step()
        classifier_optimizer.step()
        ratio = float(torch.mean(knowledge_mask))
        utilized_ratio.update(ratio, knowledge_mask.size(0))
    print("The sample ratio in utilize :{}%".format(round(utilized_ratio.avg, 4) * 100))
