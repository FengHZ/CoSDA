import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from train.shot.shot_plus_utils import rotate_batch_with_labels, Entropy, CrossEntropyLabelSmooth
from train.utils import scaler_step

def obtain_label(loader, backbone, classifier, threshold, distance_type="cosine"):
    start_test = True

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            inputs, labels, *_ = batch
            inputs = inputs.cuda()
            feas = backbone(inputs)
            outputs = classifier(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    if distance_type == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], distance_type)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]
    return predict.astype('int')


def test_per_domain_rot(loader, backbone, rot_classifier):
    backbone.eval()
    rot_classifier.eval()

    start_test = True
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            inputs, *_ = batch
            inputs = inputs.cuda()
            r_labels = np.random.randint(0, 4, len(inputs))
            r_inputs = rotate_batch_with_labels(inputs, r_labels)
            r_labels = torch.from_numpy(r_labels)
            r_inputs = r_inputs.cuda()

            f_outputs = backbone(inputs)
            f_r_outputs = backbone(r_inputs)

            r_outputs = rot_classifier(torch.cat((f_outputs, f_r_outputs), 1))
            if start_test:
                all_output = r_outputs.float().cpu()
                all_label = r_labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, r_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, r_labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    log_str = '\tAccuracy = {:.2f}%'.format(accuracy * 100)
    print(log_str)
    return accuracy


def train_target_rot(train_dloaders, backbone, rot_classifier, rot_optimizer, batch_per_epoch, scaler=None):
    # We only train the rotation classifier
    backbone.eval()
    rot_classifier.train()

    for batch_id, batch in enumerate(train_dloaders):

        if batch_id > batch_per_epoch:
            break

        inputs_test, *_ = batch
        inputs_test = inputs_test.cuda()

        r_labels_target = np.random.randint(0, 4, len(inputs_test))
        r_inputs_target = rotate_batch_with_labels(inputs_test, r_labels_target)
        r_labels_target = torch.from_numpy(r_labels_target).cuda()
        r_inputs_target = r_inputs_target.cuda()

        f_outputs = backbone(inputs_test)
        f_r_outputs = backbone(r_inputs_target)
        r_outputs_target = rot_classifier(torch.cat((f_outputs, f_r_outputs), 1))

        rotation_loss = nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
        # rot_optimizer.zero_grad()
        # rotation_loss.backward()
        # rot_optimizer.step()
        scaler_step(scaler, rotation_loss, [rot_optimizer])


def shot_pretrain(train_dloader_list, backbone_list, classifier_list,
                  optimizer_list, classifier_optimizer_list,
                  batch_per_epoch, class_num, preprocess=None, scaler=None):
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
            output_s = classifier(model(image_s))
            classifier_loss = task_criterion(output_s, label_s)
            
            # classifier_loss.backward()
            scaler.scale(classifier_loss).backward()
            # optimizer.step()
            # classifier_optimizer.step()
            scaler.step(optimizer)
            scaler.step(classifier_optimizer)
            scaler.update()


def shot_train(train_dloader, backbone, backbone_optimizer, classifier, batch_per_epoch, epoch, dataset_name,
               threshold=0, softmax_epsilon=1e-5, cls_par=0.3, ent_par=1.0, gent=True, ssl_par=0.6, shot_plus=True,
               rot_classifier=None, rot_optimizer=None, preprocess=None, scaler=None):
    if shot_plus is True:
        rot_classifier.train()

    classifier.eval()

    backbone.eval()
    mem_label = obtain_label(train_dloader, backbone, classifier, threshold)
    mem_label = torch.from_numpy(mem_label).cuda()
    backbone.train()

    for batch_id, batch in enumerate(train_dloader):

        if batch_id > batch_per_epoch:
            break
        inputs = batch[0].cuda()
        idx = batch[2]
        with torch.no_grad():
            inputs = preprocess(inputs)
        features = backbone(inputs)
        outputs = classifier(features)

        backbone_optimizer.zero_grad()

        ## classifier_loss: pseudo label loss
        classifier_loss = torch.tensor(0.0).cuda()
        if cls_par > 0:
            pred = mem_label[idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs, pred)
            classifier_loss *= cls_par
            if epoch == 0 and dataset_name == "VisDA2017":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if ent_par > 0:
            softmax_out = nn.Softmax(dim=1)(outputs)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + softmax_epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * ent_par
            classifier_loss += im_loss
        # classifier_loss.backward()
        scaler.scale(classifier_loss).backward()

        if ssl_par > 0 and shot_plus is True:
            rot_optimizer.zero_grad()
            r_labels_target = np.random.randint(0, 4, len(inputs))
            r_inputs_target = rotate_batch_with_labels(inputs, r_labels_target)
            r_labels_target = torch.from_numpy(r_labels_target).cuda()
            r_inputs_target = r_inputs_target.cuda()

            f_outputs = backbone(inputs)
            f_outputs = f_outputs.detach()
            f_r_outputs = backbone(r_inputs_target)
            r_outputs_target = rot_classifier(torch.cat((f_outputs, f_r_outputs), 1))

            rotation_loss = ssl_par * nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
            # rotation_loss.backward()
            scaler.scale(rotation_loss).backward()

        # backbone_optimizer.step()
        scaler.step(backbone_optimizer)
        if shot_plus is True:
            # rot_optimizer.step()
            scaler.step(rot_classifier)
        scaler.update()