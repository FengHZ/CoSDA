import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

def obtain_label(loader, backbone, classifier, threshold, distance_type="cosine"):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = backbone(inputs) # [batch_size, bottleneck_dim]
            outputs = classifier(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0) # [len(dataset), bottleneck_dim]
                all_output = torch.cat((all_output, outputs.float().cpu()), 0) # [len(dataset), C]
                all_label = torch.cat((all_label, labels.float()), 0) # [len(dataset)]

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1) # [len(dataset)]

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1) 
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t() 
                                                                      
                                                                      
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1) # C
    aff = all_output.float().cpu().numpy() # [len(dataset), C]
    initc = aff.transpose().dot(all_fea) # [C, len(dataset)] @ [len(dataset), bottleneck_dim + 1] -> [C, bottleneck_dim + 1]
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None]) # [] / [C, 1] -> [C, bottleneck_dim + 1]
    cls_count = np.eye(K)[predict].sum(axis=0) # [C, C] -> [len(dataset), C] -> [C]
    labelset = np.where(cls_count>threshold) 
    # print(np.shapelabelset.size())
    labelset = labelset[0] 
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], distance_type) # 1 - F.normalize(all_fea) @ F.normalize(initc[labelset]).T
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], distance_type)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
        
    return pred_label.astype('int')

def eval_initial(memory, loader, netB, netC, class_num, K):
    """Initialize the memory bank after one epoch warm up"""
    netB.eval()
    netC.eval()

    features = torch.zeros(memory.num_samples, memory.num_features).cuda()
    labels = torch.zeros(memory.num_samples).long().cuda()
    outputs = torch.zeros(memory.num_samples, class_num).cuda()
    with torch.no_grad():
        for i, item in enumerate(loader):
            imgs = item[0]
            idxs = item[-1]
            imgs = imgs.cuda()
            feature = netB(imgs)
            output = netC(feature)
            features[idxs] = feature
            labels[idxs] = (class_num + idxs).long().cuda()
            outputs[idxs] = torch.softmax(output,dim=-1)
            
        for i in range(class_num):
            rank_out = outputs[:,i]
            _,r_idx = torch.topk(rank_out,K)
            labels[r_idx] = i

        memory.features = F.normalize(features, dim=1)
        memory.labels = labels
    del features, labels, outputs

def dac_train(train_dloader, backbone, classifier, backbone_optimizer, batch_per_epoch,
                   num_classes=65, epoch=0, K=300, k=5, threshold=0,
                   confident_gate=0.97, cls_par=0.4, im_par=0.1, con_par=1.0, mmd_par=0.3, 
                   memory=None, target_test_dloader=None):
    backbone.eval()
    classifier.eval()
    mem_label = obtain_label(target_test_dloader, backbone, classifier, threshold)
    mem_label = torch.from_numpy(mem_label).cuda()
    memory.pred_labels = mem_label
    backbone.train()
    for i, (images_weak, _, images_strong_1, images_strong_2, idx) in enumerate(train_dloader):
        if i >= batch_per_epoch:
            break
        images_weak, images_strong_1, images_strong_2 = images_weak.cuda(), images_strong_1.cuda(), images_strong_2.cuda()
        backbone.train()
        features_w = backbone(images_weak)
        outputs_w = classifier(features_w)
        
        features_s = backbone(images_strong_1)
        outputs_s = classifier(features_s)
        
        features_s1 = backbone(images_strong_2)
        outputs_s1 = classifier(features_s1)
        
        with torch.no_grad():
            p_l = torch.softmax(outputs_w, dim=-1)
            max_prob, tmp_label = torch.max(p_l, dim=-1)
            mask = max_prob.ge(confident_gate).float()
            origin_label = memory.labels[idx]
            memory.labels[idx] = (tmp_label * mask + (1 - mask) * origin_label).long()
            
        if epoch > 0:
            pred = mem_label[idx]
            ce_criterion = nn.CrossEntropyLoss()
            l_ce = ce_criterion(outputs_w, pred) + ce_criterion(outputs_s, pred) + ce_criterion(outputs_s1, pred)
        else:
            l_ce = 0
        softmax_out = nn.Softmax(dim=1)(outputs_w)
        entropy_loss = torch.mean(torch.sum(-softmax_out * torch.log(softmax_out + 1e-5), dim=1))
        msoftmax_out = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax_out * torch.log(msoftmax_out + 1e-5))
        im_loss = entropy_loss - gentropy_loss
        l_con, l_mmd = 0, 0
        if epoch > 1 or (epoch == 1 and i > 0):
            l_con, l_mmd = memory(F.normalize(features_w, dim=1), 
                                F.normalize(features_s, dim=1), 
                                F.normalize(features_s1, dim=1), idx, k)
        loss = l_ce * cls_par + im_loss * im_par + l_con * con_par + l_mmd * mmd_par
        backbone_optimizer.zero_grad()
        loss.backward()
        backbone_optimizer.step()
        
    if epoch == 0:
        eval_initial(memory, train_dloader, backbone, classifier, num_classes, K)   