from datasets.DomainNet import get_domainnet_dloader
from datasets.OfficeHome import get_officehome_dloader
from train.dac.OfficeHomeDaC import get_officehome_dac_dloader
from train.dac.DomainNetDaC import get_domainnet_dac_dloader
from train.dac.Office31DaC import get_office31_dac_dloader
from train.dac.Visda2017DaC import get_visda17_dac_dloader
from datasets.Office31 import get_office31_dloader
from datasets.Visda2017 import get_visda17_dloader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.resnetda import ResNetBackBone, ResNetClassifier
import torch
import torch.nn as nn
import torch.distributed as dist
from train.gsfda.gsfda import gsfda_test_per_domain, gsfda_visda17_test_per_domain
from train.utils import test_per_domain, visda17_test_per_domain
import wandb
from os import path
import shutil
import os
from torch.utils.tensorboard import SummaryWriter
from utils.avgmeter import load_bn_statistics
from train.nrc.nrc import build_banks
from train.gsfda.gsfda import gsfda_build_banks
from train.dac.memory import MemoryBank
from copy import deepcopy

from model.vit import VitBackBone


def build_pretrained_filepath(base_path, configs):
    pretrained_folder_name = "pretrain_parameters"
    if configs["ModelConfig"]["pretrain_strategy"] == "mixup":
        pretrained_folder_name = pretrained_folder_name + "_mixup_{}".format(
            configs["ModelConfig"]["pretrain_beta"])
    elif configs["ModelConfig"]["pretrain_strategy"] == "shot":
        pretrained_folder_name = pretrained_folder_name + "_shot"
    elif configs["ModelConfig"]["pretrain_strategy"] == "Gsfda":
        pretrained_folder_name = pretrained_folder_name + "_Gsfda"
    elif configs["ModelConfig"]["pretrain_strategy"] in ["none", "regular"]:
        pass
    elif configs["ModelConfig"]["pretrain_strategy"] == "edgemix":
        pretrained_folder_name = pretrained_folder_name + "_edgemix"
    elif configs["ModelConfig"]["pretrain_strategy"] == "shot+edgemix":
        pretrained_folder_name = pretrained_folder_name + "_shot_edgemix"
    else:
        raise NotImplementedError(
            "Pretrain Strategy {} not implemented".format(
                configs["ModelConfig"]["pretrain_strategy"]))

    # build model
    if configs["ModelConfig"]["channels_per_group"] == 0:
        file_path = "{}/{}/{}/source_{}_backbone_{}.pth.tar".format(
            base_path, configs["DataConfig"]["dataset"],
            pretrained_folder_name, configs["DAConfig"]["source_domain"],
            configs["ModelConfig"]["backbone"])
    else:
        file_path = "{}/{}/{}/source_{}_backbone_{}_gn_{}.pth.tar".format(
            base_path, configs["DataConfig"]["dataset"],
            pretrained_folder_name, configs["DAConfig"]["source_domain"],
            configs["ModelConfig"]["backbone"],
            configs["ModelConfig"]["channels_per_group"])
    return file_path, pretrained_folder_name


def build_dataloaders(dataset, source_domain, target_domain, method):
    # build dataset
    if dataset == "OfficeHome":
        domains = ["Art", "Clipart", "Product", "Real_World"]
        num_classes = 65
        if source_domain not in domains or target_domain not in domains:
            raise NotImplementedError(
                "Source {} -> Target {} Not Implemented".format(
                    source_domain, target_domain))
        dataloader = get_officehome_dloader
        if method == "DaC":
            dataloader = get_officehome_dac_dloader
    elif dataset == "Office31":
        domains = ['amazon', 'webcam', 'dslr']
        num_classes = 31
        if source_domain not in domains or target_domain not in domains:
            raise NotImplementedError(
                "Source {} -> Target {} Not Implemented".format(
                    source_domain, target_domain))
        dataloader = get_office31_dloader
        if method == "DaC":
            dataloader = get_office31_dac_dloader
    elif dataset == "DomainNet":
        domains = [
            'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
        ]
        num_classes = 345
        if source_domain not in domains or target_domain not in domains:
            raise NotImplementedError(
                "Source {} -> Target {} Not Implemented".format(
                    source_domain, target_domain))
        dataloader = get_domainnet_dloader
        if method == "DaC":
            dataloader = get_domainnet_dac_dloader
    elif dataset == "VisDA2017":
        domains = ["Synthetic", "Real"]
        num_classes = 12
        if source_domain not in domains or target_domain not in domains:
            raise NotImplementedError(
                "Source {} -> Target {} Not Implemented".format(
                    source_domain, target_domain))
        dataloader = get_visda17_dloader
        if method == "DaC":
            dataloader = get_visda17_dac_dloader
    else:
        raise NotImplementedError("Dataset {} not implemented".format(dataset))
    return dataloader, domains, num_classes


def build_optimizers(backbone, classifier, configs):
    # if configs["ModelConfig"]["backbone"].startswith("resnet"):
    method = configs["DAConfig"]["method"]
    if method in {"SHOT"}:
        params = [{
            'params': backbone.module.encoder.parameters(),
            'lr': configs["TrainingConfig"]["learning_rate_begin"] * 0.1
        }, {
            'params': backbone.module.bottleneck.parameters(),
            'lr': configs["TrainingConfig"]["learning_rate_begin"] * 1.0
        }]
    elif method == "Gsfda" and configs["GsfdaConfig"]["diff_lr"] is True:
        params = [{
            'params': backbone.module.encoder.parameters(),
            'lr': configs["TrainingConfig"]["learning_rate_begin"] * 0.1
        }, {
            'params': backbone.module.bottleneck.parameters(),
            'lr': configs["TrainingConfig"]["learning_rate_begin"] * 1.0
        }, {
            'params': backbone.module.mask_embedding.parameters(),
            'lr': configs["TrainingConfig"]["learning_rate_begin"] * 1.0
        }]
    elif method in {"AaD", "NRC", "AaD+CoSDA", "NRC+CoSDA", "DaC"
                    } and configs["DataConfig"]["dataset"] == "VisDA2017":
        params = [{
            'params': backbone.module.encoder.parameters(),
            'lr': configs["TrainingConfig"]["learning_rate_begin"] * 0.1
        }, {
            'params': backbone.module.bottleneck.parameters(),
            'lr': configs["TrainingConfig"]["learning_rate_begin"] * 1.0
        }]
    else:
        params = [{
            'params': backbone.module.parameters(),
            'lr': configs["TrainingConfig"]["learning_rate_begin"]
        }]
    
    backbone_optimizer = SGD(
        params,
        momentum=configs["TrainingConfig"]["momentum"],
        weight_decay=configs["TrainingConfig"]["weight_decay"])
    classifier_optimizer = SGD(
        classifier.parameters(),
        momentum=configs["TrainingConfig"]["momentum"],
        lr=configs["TrainingConfig"]["learning_rate_begin"],
        weight_decay=configs["TrainingConfig"]["weight_decay"])
    backbone_scheduler = CosineAnnealingLR(
        backbone_optimizer,
        configs["TrainingConfig"]["total_epochs"],
        eta_min=configs["TrainingConfig"]["learning_rate_end"])
    classifier_scheduler = CosineAnnealingLR(
        classifier_optimizer,
        configs["TrainingConfig"]["total_epochs"],
        eta_min=configs["TrainingConfig"]["learning_rate_end"])
    return backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler


def build_models(file_path, configs, num_classes, scaler, mixed_precision=False):
    local_rank = dist.get_rank()
    pretrained_parameters = torch.load(file_path)
    mask_embedding = (configs["DAConfig"]["method"] == "Gsfda")
    if configs["TrainingConfig"]["ema"]:
        if configs["ModelConfig"]["backbone"].startswith('resnet'):
            teacher_backbone = ResNetBackBone(
                configs["ModelConfig"]["backbone"],
                True,
                bottleneck_dim=configs["ModelConfig"]["bottleneck_dim"],
                channels_per_group=configs["ModelConfig"]
                ["channels_per_group"],
                mask_embedding=mask_embedding, mixed_precision=mixed_precision).cuda()
        else: # ViT
            teacher_backbone = VitBackBone(
                configs["ModelConfig"]["backbone"], True,
                configs["ModelConfig"]["bottleneck_dim"],
                mixed_precision=mixed_precision).cuda()
        teacher_backbone = nn.SyncBatchNorm.convert_sync_batchnorm(
            teacher_backbone)
        teacher_backbone = nn.parallel.DistributedDataParallel(
            teacher_backbone, device_ids=[local_rank])
        teacher_classifier = ResNetClassifier(
            num_classes,
            bottleneck_dim=configs["ModelConfig"]["bottleneck_dim"], mixed_precision=mixed_precision).cuda()
        teacher_classifier = nn.parallel.DistributedDataParallel(
            teacher_classifier, device_ids=[local_rank])
        teacher_backbone.load_state_dict(pretrained_parameters["backbone"])
        teacher_classifier.load_state_dict(pretrained_parameters["classifier"])
    else:
        teacher_backbone = None
        teacher_classifier = None
    if configs["ModelConfig"]["backbone"].startswith('resnet'):
        backbone = ResNetBackBone(
            configs["ModelConfig"]["backbone"],
            True,
            bottleneck_dim=configs["ModelConfig"]["bottleneck_dim"],
            channels_per_group=configs["ModelConfig"]["channels_per_group"],
            mask_embedding=mask_embedding, mixed_precision=mixed_precision).cuda()
    else:
        backbone = VitBackBone(configs["ModelConfig"]["backbone"], True,
                               configs["ModelConfig"]["bottleneck_dim"],
                               mixed_precision=mixed_precision).cuda()
    backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    backbone = nn.parallel.DistributedDataParallel(backbone,
                                                    device_ids=[local_rank])

    classifier = ResNetClassifier(
        num_classes,
        bottleneck_dim=configs["ModelConfig"]["bottleneck_dim"], mixed_precision=mixed_precision).cuda()
    classifier = nn.parallel.DistributedDataParallel(classifier,
                                                     device_ids=[local_rank])
    backbone.load_state_dict(pretrained_parameters["backbone"])
    classifier.load_state_dict(pretrained_parameters["classifier"])
    scaler.load_state_dict(pretrained_parameters['scaler'])
    return teacher_backbone, teacher_classifier, backbone, classifier


def build_writer(args,
                 configs,
                 project="CoSDA",
                 entity="CoSDA",
                 multi_target=False):
    local_rank = dist.get_rank()
    if multi_target is True:
        wandb_name = "C_source_{} -> C_target_{} ".format(
            configs["DAConfig"]["source_domain"],
            configs["DAConfig"]["target_domain"])

        tensorboard_path = path.join(
            args.log_path, configs["DataConfig"]["dataset"],
            "Multi-target_{}->{}".format(configs["DAConfig"]["source_domain"],
                                         configs["DAConfig"]["target_domain"]),
            "method_{}".format(configs["DAConfig"]["method"]))

    else:
        wandb_name = "{}, Adaptation: {} -> {}".format(
            configs["DAConfig"]["method"],
            configs["DAConfig"]["source_domain"],
            configs["DAConfig"]["target_domain"])
        tensorboard_path = path.join(
            args.log_path, configs["DataConfig"]["dataset"],
            "SourceFree_{}->{}".format(configs["DAConfig"]["source_domain"],
                                       configs["DAConfig"]["target_domain"]),
            "method_{}_train_time:{}".format(configs["DAConfig"]["method"],
                                             args.train_time))
    if local_rank == 0:
        if args.writer == "wandb":
            writer = wandb.init(reinit=True,
                                project=project,
                                entity=entity,
                                name=wandb_name,
                                config=configs)
        elif args.writer == "tensorboard":
            writer_log_dir = tensorboard_path
            print("create writer in {}".format(writer_log_dir))
            if os.path.exists(writer_log_dir):
                # flag = input("{} will be removed, input yes to continue:".format(tensorboard_path))
                flag = "yes"
                if flag == "yes":
                    shutil.rmtree(writer_log_dir, ignore_errors=True)
            writer = SummaryWriter(log_dir=writer_log_dir)
        else:
            raise NotImplementedError()
    else:
        writer = None
    return writer


def build_method_preprocess_items(args, configs, target_train_dloader,
                                  student_backbone, student_classifier,
                                  preprocess, num_classes):
    preprocess_items = {}
    if ("NRC" in configs["DAConfig"]["method"] and configs["NRCConfig"]["ema"]
            is True) or ("AaD" in configs["DAConfig"]["method"]
                         and configs["AaDConfig"]["ema"] is True):
        feature_bank, score_bank = build_banks(
            target_train_dloader, configs["ModelConfig"]["bottleneck_dim"],
            num_classes, student_backbone, student_classifier, preprocess)
        preprocess_items["feature_bank"] = feature_bank
        preprocess_items["score_bank"] = score_bank
    elif "Gsfda" in configs["DAConfig"]["method"]:
        feature_bank, score_bank = gsfda_build_banks(
            target_train_dloader, configs["ModelConfig"]["bottleneck_dim"],
            num_classes, student_backbone, student_classifier, preprocess)
        preprocess_items["feature_bank"] = feature_bank
        preprocess_items["score_bank"] = score_bank
    elif "DaC" in configs["DAConfig"]["method"]:
        memory_dac = MemoryBank(configs["ModelConfig"]["bottleneck_dim"],
                                len(target_train_dloader.dataset), num_classes,
                                configs["DaCConfig"]["temperature"],
                                configs["DaCConfig"]["momentum"]).cuda()
        preprocess_items["memory_dac"] = memory_dac
    elif "SHOT" in configs["DAConfig"]["method"]:
        if configs["SHOTConfig"]["shot_plus"] is True:
            rot_classifier = ResNetClassifier(
                4, bottleneck_dim=2 *
                configs["ModelConfig"]["bottleneck_dim"]).cuda()
            _, pretrained_folder_name = build_pretrained_filepath(
                args.base_path, configs)
            if configs["ModelConfig"]["channels_per_group"] == 0:
                file_path = "{}/{}/{}/src_{}_tar_{}_rot_classifier.pth.tar".format(
                    args.base_path, configs["DataConfig"]["dataset"],
                    pretrained_folder_name,
                    configs["DAConfig"]["source_domain"],
                    configs["DAConfig"]["target_domain"])
            else:
                file_path = "{}/{}/{}/src_{}_tar_{}_rot_classifier_gn_{}.pth.tar".format(
                    args.base_path, configs["DataConfig"]["dataset"],
                    pretrained_folder_name,
                    configs["DAConfig"]["source_domain"],
                    configs["DAConfig"]["target_domain"],
                    configs["ModelConfig"]["channels_per_group"])
            pretrained_parameters = torch.load(file_path)
            rot_classifier.load_state_dict(
                pretrained_parameters['rot_classifier'])
            rot_optimizer = torch.optim.SGD(
                rot_classifier.parameters(),
                momentum=configs["TrainingConfig"]["momentum"],
                lr=configs["TrainingConfig"]["learning_rate_begin"],
                weight_decay=configs["TrainingConfig"]["weight_decay"])
            rot_scheduler = CosineAnnealingLR(
                rot_optimizer,
                configs["TrainingConfig"]["total_epochs"],
                eta_min=configs["TrainingConfig"]["learning_rate_end"])
        else:
            rot_classifier = None
            rot_optimizer = None
            rot_scheduler = None
        preprocess_items["rot_classifier"] = rot_classifier
        preprocess_items["rot_optimizer"] = rot_optimizer
        preprocess_items["rot_scheduler"] = rot_scheduler
    elif "CoTTA" in configs["DAConfig"]["method"]:
        initial_state_backbone = deepcopy(student_backbone.state_dict())
        initial_state_classifier = deepcopy(student_classifier.state_dict())
        preprocess_items["initial_state"] = {
            "backbone": initial_state_backbone,
            "classifier": initial_state_classifier
        }
    else:
        pass
    return preprocess_items


def build_accuracy_evaluation(source_test_dloader,
                              target_test_dloader,
                              num_classes,
                              backbone,
                              classifier,
                              writer,
                              configs,
                              epoch=-1,
                              source_bn_statistics=None,
                              target_bn_statistics=None):
    if configs["DAConfig"]["method"] == "Gsfda":
        if configs["TrainingConfig"]["ema"]:
            if source_bn_statistics is not None:
                load_bn_statistics(backbone, source_bn_statistics)

        if configs["DataConfig"]["dataset"] == "VisDA2017":
            gsfda_visda17_test_per_domain(
                configs["DAConfig"]["source_domain"],
                source_test_dloader,
                backbone,
                classifier,
                epoch,
                writer=writer,
                is_src=True,
                no_embedding=configs["GsfdaConfig"]["no_embedding"])
            gsfda_visda17_test_per_domain(
                configs["DAConfig"]["target_domain"],
                target_test_dloader,
                backbone,
                classifier,
                epoch,
                writer=writer,
                is_src=False,
                no_embedding=configs["GsfdaConfig"]["no_embedding"])
        else:
            gsfda_test_per_domain(
                configs["DAConfig"]["source_domain"],
                source_test_dloader,
                backbone,
                classifier,
                epoch,
                writer=writer,
                num_classes=num_classes,
                top_5_accuracy=(num_classes > 10),
                is_src=True,
                no_embedding=configs["GsfdaConfig"]["no_embedding"])
            gsfda_test_per_domain(
                configs["DAConfig"]["target_domain"],
                target_test_dloader,
                backbone,
                classifier,
                epoch,
                writer=writer,
                num_classes=num_classes,
                top_5_accuracy=(num_classes > 10),
                is_src=False,
                no_embedding=configs["GsfdaConfig"]["no_embedding"])
    else:
        if configs["TrainingConfig"]["ema"]:
            if source_bn_statistics is not None:
                load_bn_statistics(backbone, source_bn_statistics)
        if configs["DataConfig"]["dataset"] == "VisDA2017":
            visda17_test_per_domain(configs["DAConfig"]["source_domain"],
                                    source_test_dloader,
                                    backbone,
                                    classifier,
                                    epoch,
                                    writer=writer)
            visda17_test_per_domain(configs["DAConfig"]["target_domain"],
                                    target_test_dloader,
                                    backbone,
                                    classifier,
                                    epoch,
                                    writer=writer)
        else:
            test_per_domain(configs["DAConfig"]["source_domain"],
                            source_test_dloader,
                            backbone,
                            classifier,
                            epoch,
                            writer=writer,
                            num_classes=num_classes,
                            top_5_accuracy=(num_classes > 10))
            test_per_domain(configs["DAConfig"]["target_domain"],
                            target_test_dloader,
                            backbone,
                            classifier,
                            epoch,
                            writer=writer,
                            num_classes=num_classes,
                            top_5_accuracy=(num_classes > 10))
    if target_bn_statistics is not None:
        load_bn_statistics(backbone, target_bn_statistics)


def build_pretrained_dataloader_and_models(args, configs):
    local_rank = dist.get_rank()
    # set the dataloader list, backbone list, optimizer list, optimizer schedule list
    train_dloaders = []
    test_dloaders = []
    backbones = []
    classifiers = []
    # build dataset
    if configs["DataConfig"]["dataset"] == "Office31":
        domains = ['amazon', 'webcam', 'dslr']
        num_classes = 31
        if configs["DataConfig"]["absent_domain"] in domains:
            domains.remove(configs["DataConfig"]["absent_domain"])
        dataset_loader = get_office31_dloader
    elif configs["DataConfig"]["dataset"] == "DomainNet":
        domains = [
            'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
        ]
        num_classes = 345
        if configs["DataConfig"]["absent_domain"] in domains:
            domains.remove(configs["DataConfig"]["absent_domain"])
        dataset_loader = get_domainnet_dloader
    elif configs["DataConfig"]["dataset"] == "OfficeHome":
        domains = ["Art", "Clipart", "Product", "Real_World"]
        num_classes = 65
        if configs["DataConfig"]["absent_domain"] in domains:
            domains.remove(configs["DataConfig"]["absent_domain"])
        dataset_loader = get_officehome_dloader
    elif configs["DataConfig"]["dataset"] == "VisDA2017":
        domains = ["Real", "Synthetic"]
        num_classes = 12
        if configs["DataConfig"]["absent_domain"] in domains:
            domains.remove(configs["DataConfig"]["absent_domain"])
        dataset_loader = get_visda17_dloader
    else:
        raise NotImplementedError("Dataset {} not implemented".format(
            configs["DataConfig"]["dataset"]))
    for domain in domains:
        source_train_dloader, source_test_dloader = dataset_loader(
            args.base_path, domain, configs["TrainingConfig"]["batch_size"],
            args.workers)
        train_dloaders.append(source_train_dloader)
        test_dloaders.append(source_test_dloader)
        use_mask = (configs["TrainingConfig"]["method"] == "Gsfda")
        if configs["ModelConfig"]["backbone"].startswith('resnet'):
            backbones.append(
                ResNetBackBone(
                    configs["ModelConfig"]["backbone"],
                    bottleneck_dim=configs["ModelConfig"]["bottleneck_dim"],
                    channels_per_group=configs["ModelConfig"]
                    ["channels_per_group"],
                    mask_embedding=use_mask, mixed_precision=args.mixed_precision).cuda())
        else:
            vit_backbone = VitBackBone(configs["ModelConfig"]["backbone"],
                                       bottleneck_dim=configs["ModelConfig"]
                                       ["bottleneck_dim"], mixed_precision=args.mixed_precision).cuda()
            vit_backbone = nn.SyncBatchNorm.convert_sync_batchnorm(
                vit_backbone)
            vit_backbone = nn.parallel.DistributedDataParallel(
                vit_backbone, device_ids=[local_rank])
            backbones.append(vit_backbone)
        classifier = ResNetClassifier(
            num_classes,
            bottleneck_dim=configs["ModelConfig"]["bottleneck_dim"], mixed_precision=args.mixed_precision).cuda()
        classifier = nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank])
        classifiers.append(classifier)
    # initialize with the same parameter, especially for federated learning
    for backbone in backbones[1:]:
        backbone.load_state_dict(backbones[0].state_dict())
    for classifier in classifiers[1:]:
        classifier.load_state_dict(classifiers[0].state_dict())
    return train_dloaders, test_dloaders, backbones, classifiers, num_classes, domains


def build_pretrained_optimizers(configs, backbones, classifiers):
    backbone_optimizers = []
    classifier_optimizers = []
    backbone_optimizer_schedulers = []
    classifier_optimizer_schedulers = []
    for backbone in backbones:
        if configs["TrainingConfig"]["method"] == "Gsfda":
            params = [{
                'params':
                backbone.module.encoder.parameters(),
                'lr':
                configs["TrainingConfig"]["learning_rate_begin"] * 0.1
            }, {
                'params':
                backbone.module.bottleneck.parameters(),
                'lr':
                configs["TrainingConfig"]["learning_rate_begin"] * 1.0
            }, {
                'params':
                backbone.mask_embedding.parameters(),
                'lr':
                configs["TrainingConfig"]["learning_rate_begin"] * 1.0
            }]
        else:
            # if configs["ModelConfig"]["backbone"].startswith('resnet'):
            params = [{
                'params':
                backbone.module.encoder.parameters(),
                'lr':
                configs["TrainingConfig"]["learning_rate_begin"] * 0.1
            }, {
                'params':
                backbone.module.bottleneck.parameters(),
                'lr':
                configs["TrainingConfig"]["learning_rate_begin"] * 1.0
            }]
        backbone_optimizers.append(
            SGD(params,
                momentum=configs["TrainingConfig"]["momentum"],
                weight_decay=configs["TrainingConfig"]["weight_decay"]))
    for classifier in classifiers:
        classifier_optimizers.append(
            SGD(classifier.parameters(),
                momentum=configs["TrainingConfig"]["momentum"],
                lr=configs["TrainingConfig"]["learning_rate_begin"],
                weight_decay=configs["TrainingConfig"]["weight_decay"]))
    # create the optimizer scheduler with cosine annealing schedule
    for optimizer in backbone_optimizers:
        backbone_scheduler = CosineAnnealingLR(
            optimizer,
            configs["TrainingConfig"]["total_epochs"],
            eta_min=configs["TrainingConfig"]["learning_rate_end"])
        backbone_optimizer_schedulers.append(backbone_scheduler)
    for classifier_optimizer in classifier_optimizers:
        classifier_scheduler = CosineAnnealingLR(
            classifier_optimizer,
            configs["TrainingConfig"]["total_epochs"],
            eta_min=configs["TrainingConfig"]["learning_rate_end"])
        classifier_optimizer_schedulers.append(classifier_scheduler)
    return backbone_optimizers, classifier_optimizers, backbone_optimizer_schedulers, classifier_optimizer_schedulers


def build_pretrained_writer(args, configs, entity, project="Pretraining"):
    if args.writer == "wandb":
        writer = wandb.init(project=project,
                            entity=entity,
                            name="Time: {}, Data: {}, Model: {}".format(
                                args.train_time,
                                configs["DataConfig"]["dataset"],
                                configs["ModelConfig"]["backbone"]),
                            config=configs)
    elif args.writer == "tensorboard":
        writer_log_dir = path.join(
            args.log_path, configs["DataConfig"]["dataset"], "pretrain",
            "train_time:{}_all_sources".format(args.train_time))
        print("create writer in {}".format(writer_log_dir))
        if os.path.exists(writer_log_dir):
            # flag = input("{} train_time:{} will be removed, input yes to continue:".format(
            #     configs["DataConfig"]["dataset"], args.train_time))
            flag = "yes"
            if flag == "yes":
                shutil.rmtree(writer_log_dir, ignore_errors=True)
        writer = SummaryWriter(log_dir=writer_log_dir)
    else:
        raise NotImplementedError()
    return writer


def build_pretrained_shot(configs):
    rot_classifier = ResNetClassifier(
        4, bottleneck_dim=2 * configs["ModelConfig"]["bottleneck_dim"]).cuda()
    rot_optimizer = SGD(rot_classifier.parameters(),
                        momentum=configs["TrainingConfig"]["momentum"],
                        lr=configs["TrainingConfig"]["learning_rate_begin"],
                        weight_decay=configs["TrainingConfig"]["weight_decay"])
    rot_scheduler = CosineAnnealingLR(
        rot_optimizer,
        configs["TrainingConfig"]["total_epochs"],
        eta_min=configs["TrainingConfig"]["learning_rate_end"])
    return rot_classifier, rot_optimizer, rot_scheduler
