import os
from os import path
from utils.reproducibility import setup_seed

setup_seed()
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from train.utils import pretrain, test_per_domain, visda17_test_per_domain
from train.shot.shot_plus import shot_pretrain, train_target_rot, test_per_domain_rot
from train.gsfda.gsfda import gsfda_pretrain, gsfda_test_per_domain, gsfda_visda17_test_per_domain
from train.dataaug.preprocess import DataPreprocess
from utils.main_utils import run, init_ddp
from utils.integration_utils import build_pretrained_dataloader_and_models, build_pretrained_optimizers, \
    build_pretrained_writer, build_pretrained_shot


def main(local_rank, args, configs):
    init_ddp(local_rank)
    scaler = GradScaler(enabled=args.mixed_precision)
    train_dloaders, test_dloaders, backbones, classifiers, num_classes, domains = build_pretrained_dataloader_and_models(
        args, configs)
    backbone_optimizers, classifier_optimizers, backbone_optimizer_schedulers, classifier_optimizer_schedulers = build_pretrained_optimizers(
        configs, backbones, classifiers)
    # create perprocess for edge mixup
    if "edgemix" in configs["TrainingConfig"]["method"]:
        data_preprocess = DataPreprocess(method="edgemix",
                                         lam=configs["TrainingConfig"]["lam"])
    else:
        data_preprocess = DataPreprocess(method="identity")
    # create the event to save log info
    writer = build_pretrained_writer(
        args, configs, entity=args.entity) if local_rank == 0 else None
    # begin train
    if local_rank == 0:
        print(
            "Begin the {} time's training, Dataset:{}, Domains {}, Method {}".
            format(args.train_time, configs["DataConfig"]["dataset"], domains,
                   configs["TrainingConfig"]["method"]))
    batch_per_epoch = round(configs["TrainingConfig"]["epoch_samples"] /
                            configs["TrainingConfig"]["batch_size"])
    # train backbone
    for epoch in range(configs["TrainingConfig"]["total_epochs"]):
        for dloader in train_dloaders:
            dloader.sampler.set_epoch(epoch)
        for dloader in test_dloaders:
            dloader.sampler.set_epoch(epoch)
        if local_rank == 0:
            print("\nBegin epoch: {}/{}".format(
                epoch, configs["TrainingConfig"]["total_epochs"]))
        if "edgemix" in configs["TrainingConfig"]["method"] and epoch + configs["TrainingConfig"]["finetune_epochs"] == \
                configs["TrainingConfig"]["total_epochs"]:
            data_preprocess = DataPreprocess(method="identity")
        if "shot" in configs["TrainingConfig"]["method"]:
            shot_pretrain(train_dloaders,
                          backbones,
                          classifiers,
                          backbone_optimizers,
                          classifier_optimizers,
                          batch_per_epoch=batch_per_epoch,
                          class_num=num_classes,
                          preprocess=data_preprocess,
                          scaler=scaler)
        elif configs["TrainingConfig"]["method"] == "mixup":
            pretrain(train_dloaders,
                     backbones,
                     classifiers,
                     backbone_optimizers,
                     classifier_optimizers,
                     batch_per_epoch=batch_per_epoch,
                     mixup=True, scaler=scaler,
                     beta=configs["TrainingConfig"]["beta"])
        elif configs["TrainingConfig"]["method"] == "regular":
            pretrain(train_dloaders,
                     backbones,
                     classifiers,
                     backbone_optimizers,
                     classifier_optimizers,
                     batch_per_epoch=batch_per_epoch,
                     mixup=False, scaler=scaler)
        elif configs["TrainingConfig"]["method"] == "edgemix":
            pretrain(train_dloaders,
                     backbones,
                     classifiers,
                     backbone_optimizers,
                     classifier_optimizers,
                     batch_per_epoch=batch_per_epoch,
                     mixup=False,
                     preprocess=data_preprocess, scaler=scaler)
        elif configs["TrainingConfig"]["method"] == "Gsfda":
            gsfda_pretrain(train_dloaders,
                           backbones,
                           classifiers,
                           backbone_optimizers,
                           classifier_optimizers,
                           batch_per_epoch=batch_per_epoch,
                           class_num=num_classes,
                           reg_par=configs["TrainingConfig"]["reg_par"],
                           preprocess=data_preprocess, scaler=scaler)
        else:
            raise NotImplementedError()
        # if (epoch + 1) % 5 == 0:
        for domain_name, test_dloader, backbone, classifier in zip(
                domains, test_dloaders, backbones, classifiers):
            if configs["TrainingConfig"]["method"] == "Gsfda" and configs[
                    "DataConfig"]["dataset"] == "VisDA2017":
                gsfda_visda17_test_per_domain(
                    domain_name,
                    test_dloader,
                    backbone,
                    classifier,
                    epoch,
                    writer=writer,
                    is_src=True,
                    no_embedding=configs["TrainingConfig"]["no_embedding"])
            elif configs["TrainingConfig"]["method"] == "Gsfda":
                gsfda_test_per_domain(
                    domain_name,
                    test_dloader,
                    backbone,
                    classifier,
                    epoch,
                    writer=writer,
                    num_classes=num_classes,
                    top_5_accuracy=(num_classes > 10),
                    is_src=True,
                    no_embedding=configs["TrainingConfig"]["no_embedding"])
            elif configs["DataConfig"]["dataset"] == "VisDA2017":
                visda17_test_per_domain(domain_name, test_dloader, backbone,
                                        classifier, epoch, writer)
            else:
                test_per_domain(domain_name,
                                test_dloader,
                                backbone,
                                classifier,
                                epoch,
                                writer=writer,
                                num_classes=num_classes,
                                top_5_accuracy=(num_classes > 10))
        for scheduler in backbone_optimizer_schedulers:
            scheduler.step()
        for scheduler in classifier_optimizer_schedulers:
            scheduler.step()
        # save backbones every 10 epochs
        if epoch == 0 or (epoch + 1) % 10 == 0 and local_rank == 0:
            for domain_name, backbone, classifier in zip(
                    domains, backbones, classifiers):
                if configs["ModelConfig"]["channels_per_group"] == 0:
                    file_name = "source_{}_backbone_{}.pth.tar".format(
                        domain_name, configs["ModelConfig"]["backbone"])
                else:
                    file_name = "source_{}_backbone_{}_gn_{}.pth.tar".format(
                        domain_name, configs["ModelConfig"]["backbone"],
                        configs["ModelConfig"]["channels_per_group"])
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "domain": domain_name,
                        "backbone": backbone.state_dict(),
                        "classifier": classifier.state_dict(),
                        "scaler": scaler.state_dict()
                    },
                    filename=file_name,
                    args=args,
                    configs=configs)
    if configs["TrainingConfig"]["method"] == "shot++":
        for src_domain, src_backbone in zip(domains, backbones):
            for tar_domain, tar_dloader in zip(domains, train_dloaders):
                if tar_domain == src_domain:
                    continue
                rot_classifier, rot_optimizer, rot_scheduler = build_pretrained_shot(
                    configs)
                print("Begin training rotation ssl, Src: {}, Tar: {}".format(
                    src_domain, tar_domain))
                best_acc = 0
                best_classifier = None
                for epoch in range(configs["TrainingConfig"]["total_epochs"]):
                    train_target_rot(tar_dloader, src_backbone, rot_classifier,
                                     rot_optimizer, batch_per_epoch)
                    rot_scheduler.step()
                    print("\tEpoch:{}/{}; ".format(
                        epoch, configs["TrainingConfig"]["total_epochs"]),
                          end=" ")
                    acc = test_per_domain_rot(tar_dloader, src_backbone,
                                              rot_classifier)
                    if acc > best_acc:
                        best_classifier = rot_classifier.state_dict()
                        best_acc = acc
                if configs["ModelConfig"]["channels_per_group"] == 0:
                    file_name = "src_{}_tar_{}_rot_classifier.pth.tar".format(
                        src_domain, tar_domain)
                else:
                    file_name = "src_{}_tar_{}_rot_classifier_gn_{}.pth.tar".format(
                        src_domain, tar_domain,
                        configs["ModelConfig"]["channels_per_group"])
                save_checkpoint(
                    {
                        "src": src_domain,
                        "tar": tar_domain,
                        "rot_classifier": best_classifier,
                    },
                    filename=file_name,
                    args=args,
                    configs=configs)
    dist.destroy_process_group()


def save_checkpoint(state, filename, args, configs):
    folder_name = "pretrain_parameters"
    if configs["TrainingConfig"]["method"] == "mixup":
        folder_name += "_mixup_{}".format(configs["TrainingConfig"]["beta"])
    elif configs["TrainingConfig"]["method"] == "shot":
        folder_name += "_shot"
    elif configs["TrainingConfig"]["method"] == "regular":
        pass
    elif configs["TrainingConfig"]["method"] == "edgemix":
        folder_name += "_edgemix"
    elif configs["TrainingConfig"]["method"] == "shot+edgemix":
        folder_name += "_shot_edgemix"
    elif configs["TrainingConfig"]["method"] == "Gsfda":
        folder_name += "_Gsfda"
    else:
        raise NotImplementedError()
    filefolder = "{}/{}/{}".format(args.ckpt_path,
                                   configs["DataConfig"]["dataset"],
                                   folder_name)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    run('./pretrain/config', main)
