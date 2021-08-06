import math
import torch
from torch.optim import Optimizer


class Adam_LRDecay(Optimizer):
    r"""Implements Adam algorithm. with LR decay schemes

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, scheme, eta0, alpha, milestones=[], T_max=0, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= eta0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_LRDecay, self).__init__(params, defaults)
        
        self.eta0 = eta0
        self.alpha = alpha
        self.milestones = [int(x) for x in milestones]
        self.cur_round = 0
        self.cur_lr = eta0
        self.T_max = T_max

        # Define the function for computing the current step size for each decay.
        self.get_lr_func = None
        if scheme == 'const':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: eta0
        elif scheme == 'exp':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: cur_lr * alpha
        elif scheme == '1t':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: eta0 / (1.0 + alpha*t)
        elif scheme == '1sqrt':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: eta0 / (1.0 + alpha*(t**0.5))
        elif scheme == 'step-decay':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: cur_lr * alpha if t in milestones else cur_lr
        elif scheme == 'cosine':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: 0.5 * (1 + math.cos(t*math.pi/T_max)) * eta0


    def __setstate__(self, state):
        super(Adam_LRDecay, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None, flag_loss=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.cur_round += 1
        self.cur_lr = self.get_lr_func(self.cur_lr, self.cur_round, self.eta0,
                                       self.alpha, self.milestones, self.T_max)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1


                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt()/ math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt()/ math.sqrt(bias_correction2)).add_(group['eps'])


                step_size = self.cur_lr  / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                #p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom))

        return loss
       
