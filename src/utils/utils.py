from itertools import tee

import torch


def pairwise(iterable):
    '''
    pairwise from itertools was added only in python 3.10
    '''
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def seed_all(seed):
    g = torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return g
