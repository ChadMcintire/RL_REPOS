import os
import numpy as np
import random
import torch


def set_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

#you cannot directly use torch.nn.HuberLoss to match the FQF paper, because the original FQF loss function involves quantile regression with the asymmetric weighting:
#This is not supported by torch.nn.HuberLoss, which assumes a symmetric loss between a target and a prediction and does not allow weighting the error per quantile.
def huber_loss(x, kappa):
    abs_x = x.abs()
    quadratic = 0.5 * x.pow(2)
    linear = kappa * (abs_x - 0.5 * kappa)
    return torch.where(abs_x <= kappa, quadratic, linear)
