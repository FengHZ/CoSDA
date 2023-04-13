import argparse
import os
import yaml
import time

# Default settings
parser = argparse.ArgumentParser(description='Continual Source-free Domain Adaptation')
# Dataset path under `adaptationcfg`, such as `DomainNet_SHOT.yaml` or `backup/DomainNet/DomainNet_CosDA.yaml`.
parser.add_argument("--config", default="DomainNet.yaml")
# base path where dataset and checkpoint files are kept
parser.add_argument('-bp', '--base-path', default="./")
parser.add_argument("--writer", default="tensorboard", help="tensorboard or wandb")
parser.add_argument('-lp', '--log-path', default="./")
parser.add_argument('-e', '--entity', default="")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
# Train Info Parameters
parser.add_argument('-t', '--train-time', default="1", type=str,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# log path where log file of tensorboard are kept. 
parser.add_argument("--gpu", default="0,1", type=str, metavar='GPU plans to use', help='The GPU id(s) plans to use')
args = parser.parse_args()
# import config files
with open(r"./adaptationcfg/{}".format(args.config)) as file:
    configs = yaml.full_load(file)
# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from utils.reproducibility import setup_seed

setup_seed()
from train.cosda.cosda import cosda
from train.nrc.nrc import nrc_train, nrc_train_ema
from train.shot.shot_plus import shot_train
from train.aad.aad import aad_train, alpha_decay
from train.nrc.kd_nrc import kd_nrc_train
from train.aad.kd_aad import kd_aad_train
from train.gsfda.gsfda import gsfda_train
from utils.ema import moving_weight, exponential_moving_average, bn_statistics_moving_average
from utils.avgmeter import get_bn_statistics
from train.dataaug.preprocess import DataPreprocess
from train.dac.dac import dac_train
from train.cotta.cotta import cotta_train
from utils.integration_utils import build_pretrained_filepath, build_dataloaders, build_models, build_optimizers, \
    build_writer, build_accuracy_evaluation, build_method_preprocess_items


def main(args=args, configs=configs):
    # build datasets
    dataloader, domains, num_classes = build_dataloaders(configs["DataConfig"]["dataset"],
                                                         configs["DAConfig"]["source_domain"],
                                                         configs["DAConfig"]["target_domain"],
                                                         configs["DAConfig"]["method"])
    _, source_test_dloader = dataloader(args.base_path, configs["DAConfig"]["source_domain"],
                                        configs["TrainingConfig"]["batch_size"],
                                        args.workers)
    target_train_dloader, target_test_dloader = dataloader(args.base_path,
                                                           configs["DAConfig"]["target_domain"],
                                                           configs["TrainingConfig"]["batch_size"],
                                                           args.workers)
    # build models
    file_path, pretrained_folder_name = build_pretrained_filepath(args.base_path, configs)
    teacher_backbone, teacher_classifier, student_backbone, student_classifier = build_models(file_path, configs,
                                                                                              num_classes,
                                                                                              multi_gpu=args.data_parallel)
    backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler = build_optimizers(
        student_backbone, student_classifier, configs)
    if configs["TrainingConfig"]["ema"]:
        source_bn_statistics = get_bn_statistics(student_backbone.state_dict())
        target_bn_statistics = None
    else:
        source_bn_statistics = None
        target_bn_statistics = None
    # build writers
    writer = build_writer(args, configs, project="{}_{}".format(configs["DataConfig"]["dataset"],
                                                                configs["DAConfig"]["method"]),
                          entity=args.entity, multi_target=False)
    # begin train
    print("Begin the {} time's source free DA, Method:{}, Dataset:{}, {} -> {}".format(args.train_time,
                                                                                       configs["DAConfig"]["method"],
                                                                                       configs["DataConfig"]["dataset"],
                                                                                       configs["DAConfig"][
                                                                                           "source_domain"],
                                                                                       configs["DAConfig"][
                                                                                           "target_domain"]))

    # build training configs
    batch_per_epoch = round(configs["TrainingConfig"]["epoch_samples"] / configs["TrainingConfig"]["batch_size"])
    data_preprocess = DataPreprocess(**configs["DataAugConfig"])
    # build memory banks for cluster based methods
    method_preprocess_items = build_method_preprocess_items(args, configs, target_train_dloader, student_backbone,
                                                            student_classifier, data_preprocess, num_classes)
    # test the initial accuracy
    build_accuracy_evaluation(source_test_dloader, target_test_dloader, num_classes, student_backbone,
                              student_classifier, writer, configs, epoch=-1)
    for epoch in range(args.start_epoch, configs["TrainingConfig"]["total_epochs"]):
        print("Begin epoch: {}/{}".format(epoch, configs["TrainingConfig"]["total_epochs"] - args.start_epoch))
        if configs["DataAugConfig"]["method"] == "edgemix" and epoch + configs["DataAugConfig"]["finetune_epochs"] == \
                configs["TrainingConfig"]["total_epochs"]:
            data_preprocess = DataPreprocess(method="identity")
        if configs["DAConfig"]["method"] == "CoSDA":
            temperature = moving_weight(epoch, int(configs["TrainingConfig"]["total_epochs"]),
                                        configs["DistillConfig"]["temperature_begin"],
                                        configs["DistillConfig"]["temperature_end"])
            confidence_gate = moving_weight(epoch, configs["TrainingConfig"]["total_epochs"],
                                            configs["DistillConfig"]["confidence_gate_begin"],
                                            configs["DistillConfig"]["confidence_gate_end"])
            print("temperature: {}, confidence_gate: {}".format(temperature, confidence_gate))
            cosda(target_train_dloader, teacher_backbone, teacher_classifier, student_backbone,
                 student_classifier, backbone_optimizer, classifier_optimizer, batch_per_epoch,
                 confidence_gate, beta=configs["DistillConfig"]["beta"], temperature=temperature,
                 preprocess=data_preprocess, reg_alpha=configs["DistillConfig"]["reg_alpha"])
        elif configs["DAConfig"]["method"] == "NRC":
            if epoch > int(0.5 * configs["TrainingConfig"]["total_epochs"]) and \
                    configs["DataConfig"]["dataset"] == "OfficeHome":
                configs["NRCConfig"]["k"] = 5
                configs["NRCConfig"]["m"] = 4
            if configs["NRCConfig"]["ema"] is True:
                nrc_train_ema(method_preprocess_items["feature_bank"], method_preprocess_items["score_bank"],
                              target_train_dloader,
                              student_backbone, student_classifier,
                              backbone_optimizer, classifier_optimizer, batch_per_epoch, num_classes,
                              configs["NRCConfig"]["k"], configs["NRCConfig"]["m"], data_preprocess)
            else:
                nrc_train(target_train_dloader, student_backbone, student_classifier, backbone_optimizer,
                          classifier_optimizer, batch_per_epoch, configs["ModelConfig"]["bottleneck_dim"], num_classes,
                          configs["NRCConfig"]["k"], configs["NRCConfig"]["m"], data_preprocess)
        elif configs["DAConfig"]["method"] == "SHOT":
            shot_train(target_train_dloader, student_backbone, backbone_optimizer, student_classifier,
                       batch_per_epoch, epoch, configs['DataConfig']['dataset'],
                       configs["SHOTConfig"]["threshold"], configs["SHOTConfig"]["softmax_epsilon"],
                       configs["SHOTConfig"]["cls_par"], configs["SHOTConfig"]["ent_par"],
                       configs["SHOTConfig"]["gent"], configs["SHOTConfig"]["ssl_par"],
                       configs["SHOTConfig"]["shot_plus"], rot_classifier=method_preprocess_items["rot_classifier"],
                       rot_optimizer=method_preprocess_items["rot_optimizer"],
                       preprocess=data_preprocess)
            if configs["SHOTConfig"]["shot_plus"] is True:
                method_preprocess_items["rot_scheduler"].step()
        elif configs["DAConfig"]["method"] == "AaD":
            alpha = alpha_decay(alpha_in=configs["AaDConfig"]["alpha"], gamma=configs["AaDConfig"]["gamma"],
                                epoch=epoch)
            aad_train(target_train_dloader, student_backbone, student_classifier, backbone_optimizer,
                      classifier_optimizer, batch_per_epoch, configs["ModelConfig"]["bottleneck_dim"], num_classes,
                      configs["AaDConfig"]["k"], alpha, data_preprocess)
        elif configs["DAConfig"]["method"] == "Gsfda":
            gsfda_train(target_train_dloader, student_backbone, backbone_optimizer, student_classifier,
                        classifier_optimizer,
                        method_preprocess_items["feature_bank"], method_preprocess_items["score_bank"], batch_per_epoch,
                        class_num=num_classes, bottleneck_dim=configs["ModelConfig"]["bottleneck_dim"],
                        epsilon=configs["GsfdaConfig"]["epsilon"], gen_par=configs["GsfdaConfig"]["gen_par"],
                        k=configs["GsfdaConfig"]["k"], preprocess=data_preprocess)
        elif configs["DAConfig"]["method"] == "CoSDA+NRC":
            temperature = moving_weight(epoch, int(configs["TrainingConfig"]["total_epochs"]),
                                        configs["DistillConfig"]["temperature_begin"],
                                        configs["DistillConfig"]["temperature_end"])
            if epoch > 0.5 * configs["TrainingConfig"]["total_epochs"]:
                configs["NRCConfig"]["k"] = 5
                configs["NRCConfig"]["m"] = 4
            if configs["NRCConfig"]["ema"]:
                kd_nrc_train(target_train_dloader, student_backbone,
                             student_classifier, backbone_optimizer,
                             classifier_optimizer, batch_per_epoch, configs["DistillConfig"]["beta"], temperature,
                             configs["ModelConfig"]["bottleneck_dim"], num_classes, configs["NRCConfig"]["k"],
                             configs["NRCConfig"]["m"], method_preprocess_items["feature_bank"],
                             method_preprocess_items["score_bank"], preprocess=data_preprocess)
            else:
                kd_nrc_train(target_train_dloader, student_backbone,
                             student_classifier, backbone_optimizer,
                             classifier_optimizer, batch_per_epoch, configs["DistillConfig"]["beta"],
                             temperature, configs["ModelConfig"]["bottleneck_dim"],
                             num_classes, configs["NRCConfig"]["k"], configs["NRCConfig"]["m"],
                             preprocess=data_preprocess)
        elif configs["DAConfig"]["method"] == "CoSDA+AaD":
            alpha = alpha_decay(alpha_in=configs["AaDConfig"]["alpha"], gamma=configs["AaDConfig"]["gamma"],
                                epoch=epoch)
            if configs["AaDConfig"]["ema"]:
                kd_aad_train(target_train_dloader, student_backbone,
                             student_classifier, backbone_optimizer,
                             classifier_optimizer,
                             batch_per_epoch, configs["DistillConfig"]["beta"],
                             configs["ModelConfig"]["bottleneck_dim"],
                             num_classes, configs["AaDConfig"]["k"], method_preprocess_items["feature_bank"],
                             method_preprocess_items["score_bank"], alpha, preprocess=data_preprocess)
            else:
                kd_aad_train(target_train_dloader, student_backbone,
                             student_classifier, backbone_optimizer,
                             classifier_optimizer,
                             batch_per_epoch, configs["DistillConfig"]["beta"],
                             configs["ModelConfig"]["bottleneck_dim"],
                             num_classes, configs["AaDConfig"]["k"], None, None, alpha,
                             preprocess=data_preprocess)
        elif configs["DAConfig"]["method"] == "DaC":
            dac_train(target_train_dloader, student_backbone,
                    student_classifier, backbone_optimizer,
                    batch_per_epoch, num_classes, epoch, configs["DaCConfig"]["K"],
                    configs["DaCConfig"]["k"], configs["DaCConfig"]["threshold"],
                    configs["DaCConfig"]["confidence_gate"],
                    configs["DaCConfig"]["cls_par"], configs["DaCConfig"]["im_par"],
                    configs["DaCConfig"]["con_par"], configs["DaCConfig"]["mmd_par"],
                    method_preprocess_items["memory_dac"], target_test_dloader)
        elif configs["DAConfig"]["method"] == "CoTTA":
            confidence_gate = moving_weight(epoch, configs["TrainingConfig"]["total_epochs"],
                                            configs["CoTTAConfig"]["confidence_gate_begin"],
                                            configs["CoTTAConfig"]["confidence_gate_end"])
            cotta_train(target_train_dloader, teacher_backbone,
                        teacher_classifier, student_backbone, student_classifier,
                        method_preprocess_items["initial_state"], backbone_optimizer, classifier_optimizer,
                        batch_per_epoch,
                        aug_times=configs["CoTTAConfig"]["aug_times"], rst=configs["CoTTAConfig"]["rst"],
                        ap=configs["CoTTAConfig"]["ap"], confidence_gate=confidence_gate, preprocess=data_preprocess)
        else:
            raise NotImplementedError()
        # teacher moving average
        if configs["TrainingConfig"]["ema"]:
            # perform moving average for running mean and running var of batch norm
            target_bn_statistics = get_bn_statistics(student_backbone.state_dict())
            bn_statistics_moving_average(source_bn_statistics, target_bn_statistics, epoch,
                                         configs["TrainingConfig"]["total_epochs"],
                                         tao_begin=0.95, tao_end=0.99)
            # perform moving average for model weights
            exponential_moving_average(teacher_backbone, student_backbone, epoch,
                                       configs["TrainingConfig"]["total_epochs"],
                                       tao_begin=configs["TrainingConfig"]["tao_begin"],
                                       tao_end=configs["TrainingConfig"]["tao_end"])
            exponential_moving_average(teacher_classifier, student_classifier, epoch,
                                       configs["TrainingConfig"]["total_epochs"],
                                       tao_begin=configs["TrainingConfig"]["tao_begin"],
                                       tao_end=configs["TrainingConfig"]["tao_end"])
            student_backbone.load_state_dict(teacher_backbone.state_dict())
            student_classifier.load_state_dict(teacher_classifier.state_dict())
        build_accuracy_evaluation(source_test_dloader, target_test_dloader, num_classes, student_backbone,
                                  student_classifier, writer, configs, epoch=epoch,
                                  source_bn_statistics=source_bn_statistics, target_bn_statistics=target_bn_statistics)
        backbone_scheduler.step()
        classifier_scheduler.step()
    if args.writer == "wandb":
        writer.finish()


if __name__ == "__main__":
    time_start = time.time()
    main(args, configs)
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
