import PIL
import torch
import torch.jit
import torchvision.transforms as transforms
import train.cotta.cotta_augment as cotta_aug
from utils.avgmeter import AverageMeter
from train.utils import scaler_step

def get_cotta_transforms(gaussian_std: float = 0.005, soft=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = -1.0, 1.0

    p_hflip = 0.5

    cotta_transforms = transforms.Compose([
        # cotta_aug.Clip(clip_min, clip_max),
        cotta_aug.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.Resampling.BILINEAR,
            fill=0
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        cotta_aug.GaussianNoise(0, gaussian_std),
        # cotta_aug.Clip(clip_min, clip_max)
    ])
    return cotta_transforms


@torch.jit.script
def softmax_entropy(score, score_t):  # -> torch.Tensor:
    """Entropy of softmax distribution from scores."""
    return -0.5 * (score_t.softmax(1) * score.log_softmax(1)).sum(1) - 0.5 * (
            score.softmax(1) * score_t.log_softmax(1)).sum(1)


def cotta_train(train_dloader, teacher_backbone, teacher_classifier, student_backbone, student_classifier,
                initial_state, backbone_optimizer, classifier_optimizer, batch_per_epoch, preprocess=None,
                aug_times=32, rst=0.01, ap=0.92, confidence_gate=0.8, scaler=None):
    utilized_ratio = AverageMeter()
    teacher_backbone.train()
    teacher_classifier.train()
    student_backbone.train()
    student_classifier.train()
    augmentations = get_cotta_transforms()
    for i, (image_t, *_) in enumerate(train_dloader):
        if i >= batch_per_epoch:
            break
        image_t = image_t.cuda()
        with torch.no_grad():
            image_t = preprocess(image_t)
        # reset grad
        backbone_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        # build labels with teacher model and dataaug
        with torch.no_grad():
            score_t = teacher_classifier(teacher_backbone(image_t))
            p_t = score_t.softmax(1)
            anchor_prob = p_t.max(1)[0]
            if anchor_prob.mean(0) < ap:
                # do augmentation 32 times
                score_augments = []
                for _ in range(aug_times):
                    score_aug_ = teacher_classifier(teacher_backbone(augmentations(image_t))).detach()
                    score_augments.append(score_aug_)
                score_t = torch.stack(score_augments).mean(0)
            else:
                pass
        predict_t = torch.softmax(score_t, dim=1)
        # get the knowledge with weight and mask
        max_p, max_p_class = predict_t.max(1)
        knowledge_mask = (max_p > confidence_gate).float().cuda()
        score_s = student_classifier(student_backbone(image_t))
        knowledge = torch.softmax(score_t / 0.3, dim=1)
        task_loss = torch.sum(
            knowledge_mask * torch.sum(-1 * knowledge * score_s.log_softmax(dim=1), dim=1)) / torch.sum(
            knowledge_mask)
        # task_loss = torch.sum(knowledge_mask * softmax_entropy(score_s, score_t / 0.3)) / torch.sum(
        #     knowledge_mask)
        # task_loss = softmax_entropy(score_s, score_t / 0.3).mean()
        scaler_step(scaler, task_loss, [backbone_optimizer, classifier_optimizer])
        """
        Update teacher model with ema and momentum 0.999
        The original CoTA do not work in DA benchmarks, teacher should change slowly
        We use our Ema strategy, change teacher in epoch-level and use BN-Ema for the running mean and running var
        in BatchNorm layers.
        """
        # teacher_backbone = cotta_ema(teacher_backbone, student_backbone, alpha)
        # teacher_classifier = cotta_ema(teacher_classifier, student_classifier, alpha)
        # stochastic restore the source params
        for nm, m in student_backbone.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < rst).float().cuda()
                    with torch.no_grad():
                        p.data = initial_state["backbone"][f"{nm}.{npp}"] * mask + p * (1. - mask)
        for nm, m in student_classifier.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < rst).float().cuda()
                    with torch.no_grad():
                        p.data = initial_state["classifier"][f"{nm}.{npp}"] * mask + p * (1. - mask)
        ratio = float(torch.mean(knowledge_mask))
        utilized_ratio.update(ratio, knowledge_mask.size(0))
    print("The sample ratio in utilize :{}%".format(round(utilized_ratio.avg, 4) * 100))
