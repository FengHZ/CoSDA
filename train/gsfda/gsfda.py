import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from train.shot.shot_plus_utils import CrossEntropyLabelSmooth
from utils.avgmeter import AverageMeter
from train.utils import scaler_step

def gsfda_build_banks(train_dloader, bottleneck_dim, num_classes, backbone, classifier, preprocess):
    backbone.eval()
    classifier.eval()
    num_samples = len(train_dloader.dataset)
    feature_bank = torch.zeros(num_samples, bottleneck_dim)
    score_bank = torch.zeros(num_samples, num_classes).cuda()
    with torch.no_grad():
        for batch_id, batch in enumerate(train_dloader):
            image = batch[0].cuda()
            idx = batch[2]
            with torch.no_grad():
                image = preprocess(image)
            feature, _ = backbone(image, s=100, t=1, all_mask=False, no_embedding=False)
            feature_norm = F.normalize(feature)
            if feature_bank.dtype != feature_norm.dtype:
                feature_bank = feature_bank.to(feature_norm.dtype)
                score_bank = score_bank.to(feature_norm.dtype)
            feature_bank[idx] = feature_norm.detach().clone().cpu()

            score = classifier(feature)
            score_softmax = nn.Softmax(-1)(score)
            score_bank[idx] = score_softmax.detach().clone()
    return feature_bank, score_bank


def gsfda_pretrain(train_dloader_list, backbone_list, classifier_list,
                   optimizer_list, classifier_optimizer_list,
                   batch_per_epoch, class_num, reg_par=0.75, preprocess=None, scaler=None):
    for model in backbone_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    task_criterion = CrossEntropyLabelSmooth(class_num).cuda()
    for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list,
                                                                                 backbone_list, classifier_list,
                                                                                 optimizer_list,
                                                                                 classifier_optimizer_list):
        for i, (image_s, label_s, *_) in enumerate(train_dloader):
            if i > batch_per_epoch:
                break
            image_s = image_s.cuda()
            if preprocess is not None:
                with torch.no_grad():
                    image_s = preprocess(image_s)

            label_s = label_s.long().cuda()
            optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            feature_s, masks = model(image_s, t=0, s=100, all_mask=True, no_embedding=False)
            output_s1 = classifier(feature_s[0])
            output_s2 = classifier(feature_s[1])

            # sparsity regularization for domain attention
            reg = 0
            count = 0
            for m in masks[0]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            for m in masks[1]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            reg /= count
            classifier_loss = task_criterion(output_s1, label_s) + task_criterion(output_s2, label_s) + reg_par * reg

            scaler_step(scaler, classifier_loss, [optimizer, classifier_optimizer])


def gsfda_train(train_dloader, backbone, backbone_optimizer, classifier, classifier_optimizer,
                fea_bank, score_bank, batch_per_epoch, class_num, bottleneck_dim,
                epsilon=1e-5, gen_par=1, k=2, preprocess=None, scaler=None):
    backbone.train()
    classifier.train()

    for batch_id, batch in enumerate(train_dloader):

        if batch_id > batch_per_epoch:
            break

        inputs = batch[0].cuda()
        idx = batch[2]
        with torch.no_grad():
            inputs = preprocess(inputs)

        backbone_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        features, masks = backbone(inputs, t=1, s=100, all_mask=False, no_embedding=False)
        outputs = classifier(features)
        outputs_sm = nn.Softmax(dim=1)(outputs)
        outputs_re = outputs_sm.unsqueeze(1)

        with torch.no_grad():
            output_f_norm = F.normalize(features)
            fea_bank[idx].fill_(-0.1)  # do not use the current mini-batch in fea_bank
            output_f_ = output_f_norm.cpu().detach().clone()
            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=k)
            score_near = score_bank[idx_near]  # batch x K x num_class
            score_near = score_near.permute(0, 2, 1)

            # update banks
            fea_bank[idx] = output_f_.detach().clone().cpu()
            score_bank[idx] = outputs_sm.detach().clone()

        const = torch.log(torch.bmm(outputs_re, score_near)).sum(-1)
        loss = -torch.mean(const)

        msoftmax = outputs_sm.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + epsilon))
        loss += gentropy_loss * gen_par

        for n, p in backbone.bottleneck.named_parameters():
            if n.find('fc') != -1:
                if n.find('bias') == -1:
                    mask_ = ((1 - masks)).view(-1, 1).expand(bottleneck_dim, 2048).cuda()
                    p.grad.data *= mask_
                else:  # no bias here
                    mask_ = ((1 - masks)).squeeze().cuda()
                    p.grad.data *= mask_
            elif n.find('bn') != -1:
                mask_ = ((1 - masks)).view(-1).cuda()
                p.grad.data *= mask_

        for n, p in classifier.named_parameters():
            if n.find('weight_v') != -1:
                masks__ = masks.view(1, -1).expand(class_num, bottleneck_dim)
                mask_ = ((1 - masks__)).cuda()
                p.grad.data *= mask_
        scaler_step(scaler, loss, [backbone_optimizer, classifier_optimizer])


def gsfda_test_per_domain(domain_name, test_dloader, backbone, classifier, epoch, writer, num_classes=345,
                          top_5_accuracy=True, is_src=True, no_embedding=False):
    backbone.eval()
    classifier.eval()
    domain_loss = AverageMeter()
    tmp_score = []
    tmp_label = []
    task_criterion = nn.CrossEntropyLoss().cuda()
    embedding_verbose = " (no_embedding)" if no_embedding is True else ""
    mask_t = 0 if is_src else 1
    for _, item in enumerate(test_dloader):  # change from (image, _) to item
        image = item[0]
        label = item[1]
        image = image.cuda()
        label = label.long().cuda()
        with torch.no_grad():
            if no_embedding:
                feature = backbone(image, t=mask_t, s=100, all_mask=False, no_embedding=no_embedding)
            else:
                feature, _ = backbone(image, t=mask_t, s=100, all_mask=False, no_embedding=no_embedding)
            output = classifier(feature)
        label_onehot = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
        task_loss = task_criterion(output, label)
        domain_loss.update(float(task_loss.item()), image.size(0))
        tmp_score.append(torch.softmax(output, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot)
    if type(writer) == SummaryWriter:
        writer.add_scalar(tag="domain_{}_loss{}".format(domain_name, embedding_verbose), scalar_value=domain_loss.avg,
                          global_step=epoch + 1)
    else:
        writer.log({"domain_{}_loss{}".format(domain_name, embedding_verbose): domain_loss.avg}, step=epoch + 1)

    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    _, y_pred = torch.topk(tmp_score, k=5, dim=1)
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    if type(writer) == SummaryWriter:
        writer.add_scalar(tag="domain_{}_accuracy_top1{}".format(domain_name, embedding_verbose),
                          scalar_value=top_1_accuracy,
                          global_step=epoch + 1)
    else:
        writer.log({"domain_{}_accuracy_top1{}".format(domain_name, embedding_verbose): top_1_accuracy}, step=epoch + 1)
    print("Domain :{}, Top1 Accuracy{}:{}".format(domain_name, embedding_verbose, top_1_accuracy))
    if top_5_accuracy:
        top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
        if type(writer) == SummaryWriter:
            writer.add_scalar(tag="domain_{}_accuracy_top5{}".format(domain_name, embedding_verbose),
                              scalar_value=top_5_accuracy_s,
                              global_step=epoch + 1)
        else:
            writer.log({"domain_{}_accuracy_top5{}".format(domain_name, embedding_verbose): top_5_accuracy_s},
                       step=epoch + 1)


def gsfda_visda17_test_per_domain(domain_name, test_dloader, backbone, classifier, epoch, writer, is_src=True,
                                  no_embedding=False):
    """Test accuracy of the dataset VisDA2017 with the method gsfda. 
       Use sklearn to compute confusion matrix, and calculate accuracy of each class.
       Classes are in the following order:
       `[plane, bcycl, bus, car, horse, knife, mcycl, person, plant, sktbrd, train, truck]`
    """
    backbone.eval()
    classifier.eval()
    dataset_size = len(test_dloader.dataset)
    all_output = torch.zeros(dataset_size, 12).cuda()
    y_true = torch.zeros(dataset_size).long().cuda()
    embedding_verbose = " (no_embedding)" if no_embedding is True else ""
    mask_t = 0 if is_src else 1
    for batch_idx, item in enumerate(test_dloader):
        image = item[0]
        label = item[1]
        image = image.cuda()
        label = label.long().cuda()
        batch_size = image.size(0)
        y_true[batch_size * batch_idx: batch_size * (batch_idx + 1)] = label
        with torch.no_grad():
            feature = backbone(image, t=mask_t, s=100, all_mask=False, no_embedding=no_embedding)[0]
            output = classifier(feature)
        all_output[batch_size * batch_idx: batch_size * (batch_idx + 1)] = output
    _, y_pred = torch.max(all_output, 1)
    matrix = confusion_matrix(y_true.cpu(), y_pred.cpu())
    top_1_accuracy_list = matrix.diagonal() / matrix.sum(axis=1) * 100
    avg_top_1_accuracy = top_1_accuracy_list.mean()
    top_1_accuracy_str_list = [f"{accuracy_per_class:.2f}" for accuracy_per_class in top_1_accuracy_list]
    if type(writer) == SummaryWriter:
        writer.add_scalar(tag=f"domain_{domain_name}_accuracy_top1{embedding_verbose}",
                          scalar_value=avg_top_1_accuracy, global_step=epoch + 1)
    else:
        writer.log({f"domain_{domain_name}_accuracy_top1{embedding_verbose}": avg_top_1_accuracy}, step=epoch + 1)
    print(f"Domain: {domain_name}, Top1 Accuracy: {avg_top_1_accuracy:.2f}")
    print(f"Top1 Accuracy List: [{' '.join(top_1_accuracy_str_list)}]")
