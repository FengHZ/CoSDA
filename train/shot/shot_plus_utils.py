import torch
import torch.utils.data
import torch.nn as nn
import torch.distributed as dist

# Assumes that tensor is (nchannels, height, width)
def _tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def _tensor_rot_180(x):
    return x.flip(2).flip(1)


def _tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def _rotate_single_with_label(img, label):
    if label == 1:
        img = _tensor_rot_90(img)
    elif label == 2:
        img = _tensor_rot_180(img)
    elif label == 3:
        img = _tensor_rot_270(img)
    return img


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        img = _rotate_single_with_label(img, label)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            local_rank = dist.get_rank()
            targets = targets.cuda(local_rank)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
