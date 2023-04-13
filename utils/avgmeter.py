class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_bn_statistics(parameter_dict):
    bn_dict = {}
    for key, value in parameter_dict.items():
        if "running_mean" in key or "running_var" in key:
            bn_dict[key] = value.detach().clone()
    return bn_dict


def load_bn_statistics(model, bn_dict):
    d = model.state_dict()
    for key, value in bn_dict.items():
        d[key] = value.detach().clone()
    model.load_state_dict(d)
