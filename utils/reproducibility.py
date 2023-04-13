import os
import torch
import numpy as np
import random


def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    # avoid the same order for every epoch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
