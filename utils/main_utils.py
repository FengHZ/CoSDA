import argparse
import yaml
import os
import torch
import torch.distributed as dist
import time
import torch.multiprocessing as mp

def prepare(config_base_path: str):
    """parse args and prepare for DDP.

    Args:
        config_base_path (str): base path of configuration. `./pretrain/config` for pretrain, `./adaptationcfg` for single_tar

    Returns:
        _type_: _description_
    """
    # Default settings
    parser = argparse.ArgumentParser(description='CoSDA')
    # Dataset Parameters
    parser.add_argument("--config",
                        default="backup/Visda17/Visda17_CoSDA.yaml")
    parser.add_argument('-bp', '--base-path',
                        default="/data/fhz_11821062")  # dataset path
    parser.add_argument('-cp', '--ckpt-path',
                        default='/home/yangzr/ckpt')  # checkpoint path
    parser.add_argument("--writer",
                        default="tensorboard",
                        help="tensorboard or wandb")
    parser.add_argument('-lp', '--log-path',
                        default="/home/yangzr/logs")  # log path
    parser.add_argument('-e', '--entity', default="zhaorui")
    parser.add_argument('-j',
                        '--workers',
                        default=8,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 8)')
    # Train Strategy Parameters
    parser.add_argument('-t',
                        '--train-time',
                        default=1,
                        type=str,
                        metavar='N',
                        help='the x-th time of training')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-p',
                        '--print-freq',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument("--gpu",
                        default="0,1",
                        type=str,
                        metavar='GPU plans to use',
                        help='The GPU id plans to use')
    parser.add_argument("--port", default="12223", type=str)
    parser.add_argument('--mixed-precision', action='store_true')
    args = parser.parse_args()
    config_path = os.path.join(config_base_path, args.config)
    with open(config_path) as file:
        configs = yaml.full_load(file)
    os.environ['MASTER_ADDR'] = 'localhost'  # 0号机器的IP
    os.environ['MASTER_PORT'] = args.port  # 0号机器的可用端口
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    return args, configs


def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
def run(config_base_path: str, main_func: callable):
    args, configs = prepare(config_base_path)
    time_start = time.time()
    mp.spawn(main_func, args=(args, configs), nprocs=torch.cuda.device_count())
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')