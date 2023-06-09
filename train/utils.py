import torch.nn as nn
from utils.avgmeter import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix


def mixup_data(image, label, beta=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(beta, beta)
    batch_size = image.size(0)
    index = torch.randperm(batch_size).cuda()

    mixed_image = lam * image + (1 - lam) * image[index, :]
    label_a, label_b = label, label[index]
    return mixed_image, label_a, label_b, lam


def mixup_criterion(criterion, prediction, label_a, label_b, lam):
    """
    :param criterion: the cross entropy criterion
    :param prediction: y_pred
    :param label_a: label = lam * label_a + (1-lam)* label_b
    :param label_b: label = lam * label_a + (1-lam)* label_b
    :param lam: label = lam * label_a + (1-lam)* label_b
    :return:  cross_entropy(pred,label)
    """
    return lam * criterion(prediction, label_a) + (1 - lam) * criterion(prediction, label_b)


def pretrain(train_dloader_list, backbone_list, classifier_list, optimizer_list, classifier_optimizer_list,
             batch_per_epoch, mixup, preprocess=None, **kwargs):
    task_criterion = nn.CrossEntropyLoss().cuda()
    domain_num = len(train_dloader_list)
    for model in backbone_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # Train model locally on source domains
    for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list,
                                                                                 backbone_list,
                                                                                 classifier_list,
                                                                                 optimizer_list,
                                                                                 classifier_optimizer_list):
        for i, (image_s, label_s, *_) in enumerate(train_dloader):
            if i >= batch_per_epoch:
                break
            # reset grad
            optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            # prepare training data on each source domain
            image_s = image_s.cuda()
            if preprocess is not None:
                with torch.no_grad():
                    image_s = preprocess(image_s)
            label_s = label_s.long().cuda()
            if mixup:
                mixed_image, label_a, label_b, lam = mixup_data(image_s, label_s, kwargs['beta'])
                feature_s = model(mixed_image)
                output_s = classifier(feature_s)
                task_loss_s = mixup_criterion(task_criterion, output_s, label_a, label_b, lam)
            else:
                feature_s = model(image_s)
                output_s = classifier(feature_s)
                task_loss_s = task_criterion(output_s, label_s)
            task_loss_s.backward()
            optimizer.step()
            classifier_optimizer.step()


def test_per_domain(domain_name, test_dloader, backbone, classifier, epoch, writer, num_classes=345,
                    top_5_accuracy=True):
    backbone.eval()
    classifier.eval()
    domain_loss = AverageMeter()
    tmp_score = []
    tmp_label = []
    task_criterion = nn.CrossEntropyLoss().cuda()
    for _, item in enumerate(test_dloader):  # change from (image, _) to item
        image = item[0]
        label = item[1]
        image = image.cuda()
        label = label.long().cuda()
        with torch.no_grad():
            output = classifier(backbone(image))
        label_onehot = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
        task_loss = task_criterion(output, label)
        domain_loss.update(float(task_loss.item()), image.size(0))
        tmp_score.append(torch.softmax(output, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot)
    if type(writer) == SummaryWriter:
        writer.add_scalar(tag="domain_{}_loss".format(domain_name), scalar_value=domain_loss.avg, global_step=epoch + 1)
    else:
        writer.log({"domain_{}_loss".format(domain_name): domain_loss.avg}, step=epoch + 1)
    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    _, y_pred = torch.topk(tmp_score, k=5, dim=1)
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    if type(writer) == SummaryWriter:
        writer.add_scalar(tag="domain_{}_accuracy_top1".format(domain_name), scalar_value=top_1_accuracy,
                          global_step=epoch + 1)
    else:
        writer.log({"domain_{}_accuracy_top1".format(domain_name): top_1_accuracy}, step=epoch + 1)
    print("Domain :{}, Top1 Accuracy:{}".format(domain_name, top_1_accuracy))
    if top_5_accuracy:
        top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
        if type(writer) == SummaryWriter:
            writer.add_scalar(tag="domain_{}_accuracy_top5".format(domain_name), scalar_value=top_5_accuracy,
                              global_step=epoch + 1)
        else:
            writer.log({"domain_{}_accuracy_top5".format(domain_name): top_5_accuracy}, step=epoch + 1)
    return top_1_accuracy


def visda17_test_per_domain(domain_name, test_dloader, backbone, classifier, epoch, writer):
    """Test accuracy of the dataset VisDA2017. Use sklearn to compute confusion matrix, and calculate accuracy
       of each class.
       Classes are in the following order:
       `[plane, bcycl, bus, car, horse, knife, mcycl, person, plant, sktbrd, train, truck]`
    """
    backbone.eval()
    classifier.eval()
    dataset_size = len(test_dloader.dataset)
    all_output = torch.zeros(dataset_size, 12).cuda()
    y_true = torch.zeros(dataset_size).long().cuda()
    for batch_idx, item in enumerate(test_dloader):
        image = item[0]
        label = item[1]
        image = image.cuda()
        label = label.long().cuda()
        batch_size = image.size(0)
        y_true[batch_size * batch_idx: batch_size * (batch_idx + 1)] = label
        with torch.no_grad():
            output = classifier(backbone(image))
        all_output[batch_size * batch_idx: batch_size * (batch_idx + 1)] = output
    _, y_pred = torch.max(all_output, 1)
    matrix = confusion_matrix(y_true.cpu(), y_pred.cpu())
    top_1_accuracy_list = matrix.diagonal() / matrix.sum(axis=1) * 100
    avg_top_1_accuracy = top_1_accuracy_list.mean()
    top_1_accuracy_str_list = [f"{accuracy_per_class:.2f}" for accuracy_per_class in top_1_accuracy_list]
    if type(writer) == SummaryWriter:
        writer.add_scalar(tag=f"domain_{domain_name}_accuracy_top1", scalar_value=avg_top_1_accuracy,
                          global_step=epoch + 1)
    else:
        writer.log({f"domain_{domain_name}_accuracy_top1": avg_top_1_accuracy}, step=epoch + 1)
    print(f"Domain: {domain_name}, Top1 Accuracy: {avg_top_1_accuracy:.2f}")
    print(f"Top1 Accuracy List: [{' '.join(top_1_accuracy_str_list)}]")
