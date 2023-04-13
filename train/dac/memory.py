import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd


class MB(autograd.Function):
    
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def mb(inputs, indexes, features, momentum=0.5):
    return MB.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class MemoryBank(nn.Module):
    def __init__(self, num_features, num_samples, class_num, temp=0.05, momentum=0.2):
        super(MemoryBank, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        # self.args = args
        self.class_num = class_num

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features).cuda())
        # the source-like samples labels
        self.register_buffer('labels', torch.zeros(num_samples).long().cuda())
        # the psuedo-labels
        self.register_buffer('pred_labels', torch.zeros(num_samples).long().cuda())

    def forward(self, inputs, inputs_s, inputs_s1, indexes, k=10):
        # inputs: B*hidden_dim, features: L*hidden_dim
        inputs_out = mb(inputs, indexes, self.features, self.momentum)
        inputs_out /= self.temp  # B*L
        
        # generate local information
        B = inputs.size(0)
        local = (inputs.mm(inputs_s.t()) + inputs.mm(inputs_s1.t()))/2 # B*B
        _ , neibor_idx = torch.topk(inputs_out,k)  # B*k
        neibor_ftr = self.features[neibor_idx].permute(0,2,1) #B*2048*k
        
        _local=(torch.bmm(inputs.unsqueeze(1),neibor_ftr)).sum(-1) # B * 1
        local = (local + _local.expand_as(local))*(torch.eye(B).cuda())
        local /= self.temp


        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        # Achieve adaptive contrastive learning
        targets = self.labels[indexes].clone()
        labels = self.labels.clone()
        
        sim = torch.zeros(labels.max()+1, B).float().cuda()  # L * B 
        sim.index_add_(0, labels, inputs_out.t().contiguous())
        # add the local information
        sim.index_add_(0, targets, local.contiguous())
        
        
        nums = torch.zeros(labels.max()+1, 1).float().cuda() 
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        
        nums_help = torch.zeros(labels.max()+1, 1).float().cuda() 
        nums_help.index_add_(0, targets, torch.ones(B,1).float().cuda())
        nums+=(nums_help>0).float()*(k+1)
        #avoid divide 0
        mask = (nums>0).float() 
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        
        loss1 = F.nll_loss(torch.log(masked_sim+1e-6), targets)

        
        # Adaptation loss (MMD with memory bank)
        target_mask = (labels>=self.class_num).float()
        target_idx = torch.nonzero(target_mask,as_tuple=True)
        source_idx = torch.nonzero(1-target_mask,as_tuple=True)
        target_label = self.pred_labels[target_idx].clone()
        source_label = self.pred_labels[source_idx].clone()
        
        ad_sim_target = torch.zeros(self.class_num,B).float().cuda()
        ad_sim_target.index_add_(0, target_label, inputs_out.t().contiguous()[target_idx])
        t_nums = torch.zeros(self.class_num, 1).float().cuda()
        t_nums.index_add_(0, target_label, torch.ones(self.num_samples,1).float().cuda()[target_idx])
        ad_sim_source = torch.zeros(self.class_num,B).float().cuda()
        ad_sim_source.index_add_(0, source_label, inputs_out.t().contiguous()[source_idx])
        s_nums = torch.zeros(self.class_num, 1).float().cuda()
        s_nums.index_add_(0, source_label, torch.ones(self.num_samples,1).float().cuda()[source_idx])
        t_nums_mask =  (t_nums>0).float() 
        s_nums_mask =  (s_nums>0).float() 
        ad_sim_target /= (t_nums*t_nums_mask+(1-t_nums_mask)).clone().expand_as(ad_sim_target)
        ad_sim_source /= (s_nums*s_nums_mask+(1-s_nums_mask)).clone().expand_as(ad_sim_source)
        # B*C
        ad_sim_target = ad_sim_target.t()
        ad_sim_source = ad_sim_source.t()
        
        batch_labels = self.pred_labels[indexes].clone()
        ad_help_t = torch.index_select(ad_sim_target,1,batch_labels)
        ad_help_s = torch.index_select(ad_sim_source,1,batch_labels)
        ad_t = torch.diagonal(ad_help_t).unsqueeze(1)
        ad_s = torch.diagonal(ad_help_s).unsqueeze(1)
        ad_sim = torch.cat((ad_s,ad_t),1) #B*2
        
        # if self.args.ad_method == 'LMMD': 
        #     batch_mask = target_mask[indexes].clone()
        #     batch_help = torch.ones(B).cuda()*(-1)+2*batch_mask # target-like 1, source-like -1
        #     batch_help = batch_help.unsqueeze(1)
        #     batch_label = torch.cat((batch_help*-1,batch_help),1) # B*2
        #     loss2 = (ad_sim*batch_label).sum(1).mean()
        # else:
        batch_mask = (targets<self.class_num).long() # if source-like adapt to the target features
        exp_ad = torch.exp(ad_sim)
        ad_sums = exp_ad.sum(1,keepdim=True) + 1e-6
        exp_sims = exp_ad/ad_sums
        loss2 = (F.nll_loss(torch.log(exp_sims+1e-6), batch_mask, reduction='none')).mean()
        return loss1,loss2