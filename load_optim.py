"""
Load the desired optimizer.
"""

import torch.optim as optim
from sgd_lr_decay import SGD_LRDecay
from sls import Sls
from adam_lr_decay import Adam_LRDecay
from AdamW_lr_decay import AdamW_LRDecay

def load_optim(params, optim_method, eta0, alpha, c, milestones, T_max, 
               n_batches_per_epoch, nesterov, momentum, weight_decay):
    """
    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        optim_method: which optimizer to use.
        eta0: starting step size.
        alpha: decaying factor for various methods.
        c: used in line search.
        milestones: used for SGD stage decay denoting when to decrease the
            step size, unit in iteration.
        T_max: total number of steps.
        n_batches_per_epoch: number of batches in one train epoch.
        nesterov: whether to use nesterov momentum (True) or not (False).
        momentum: momentum factor used in variants of SGD.
        weight_decay: weight decay factor.

    Outputs:
        an optimizer
    """

    if optim_method == 'SGD' or optim_method == 'SGD_ReduceLROnPlateau':
        optimizer = optim.SGD(params=params, lr=eta0, momentum=momentum,
                              weight_decay=weight_decay, nesterov=nesterov)
    elif optim_method == 'AdaGrad':
        optimizer = optim.Adagrad(params=params, lr=eta0,
                               weight_decay=weight_decay)
    elif optim_method.startswith('Adam') and optim_method.endswith('Decay'):
        if optim_method == 'Adam_Const_Decay':
            scheme = 'const'
        elif optim_method == 'Adam_Exp_Decay':
            scheme = 'exp'
        elif optim_method == 'Adam_1t_Decay':
            scheme = '1t'
        elif optim_method == 'Adam_1sqrt_Decay':
            scheme = '1sqrt'
        elif optim_method == 'Adam_Step_Decay':
            scheme = 'step-decay'
        optimizer = Adam_LRDecay(params=params, scheme=scheme, eta0=eta0,
                               alpha=alpha, milestones=milestones, T_max=T_max, weight_decay=weight_decay, amsgrad=True)
  
    elif optim_method.startswith('ADAMW') and optim_method.endswith('Decay'):
        if optim_method == 'ADAMW_Const_Decay':
            scheme = 'const'
        elif optim_method == 'ADAMW_Exp_Decay':
            scheme = 'exp'
        elif optim_method == 'ADAMW_1t_Decay':
            scheme = '1t'
        elif optim_method == 'ADAMW_1sqrt_Decay':
            scheme = '1sqrt'
        elif optim_method == 'ADAMW_Step_Decay':
            scheme = 'step-decay'
        optimizer = AdamW_LRDecay(params=params, scheme=scheme, eta0=eta0,
                               alpha=alpha, milestones=milestones, T_max=T_max, weight_decay=weight_decay, amsgrad=True)
    elif optim_method.startswith('SGD') and optim_method.endswith('Decay'):
        if optim_method == 'SGD_Const_Decay':
            scheme = 'const'
        elif optim_method == 'SGD_Exp_Decay':
            scheme = 'exp'
        elif optim_method == 'SGD_1t_Decay':
            scheme = '1t'
        elif optim_method == 'SGD_1sqrt_Decay':
            scheme = '1sqrt'
        elif optim_method == 'SGD_Step_Decay':
            scheme = 'step-decay'
        optimizer = SGD_LRDecay(params=params, scheme=scheme, eta0=eta0,
                               alpha=alpha, milestones=milestones, T_max=T_max,
                               momentum=momentum, weight_decay=weight_decay,
                               nesterov=nesterov)
    else:
        raise ValueError("Invalid optimizer: {}".format(optim_method))

    return optimizer
