import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False,
                 dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = []
        if dilation == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class NormalizedEuclideanLoss(nn.Module):
    def __init__(self):
        super(NormalizedEuclideanLoss, self).__init__()

    @staticmethod
    def forward(inputs, targets):
        dim = targets.ndim
        y_n = torch.sum(targets, dim=tuple(range(dim)[1:]) if dim > 3 else None).sqrt()
        return torch.square(inputs / y_n - targets / y_n).sum() / (inputs.shape[0] if dim > 3 else 1)


class MIoU(nn.Module):
    def __init__(self):
        super(MIoU, self).__init__()

    @staticmethod
    def forward(inputs, targets):
        m_iou = 0
        for i in range(4):
            y_min = torch.fmin(inputs, targets)
            iou = 1 - y_min.sum() / torch.sum(inputs + targets - y_min)
            m_iou += iou
            t_size = inputs.shape
            p_size = targets.shape
            sizes = [t_size[-2], t_size[-1], p_size[-2], p_size[-1]]
            for j, k in enumerate(sizes):
                if k % 2 != 0:
                    sizes[j] += 1
                sizes[j] //= 2
            inputs = nn.AdaptiveAvgPool2d((sizes[0], sizes[1]))(inputs.unsqueeze(0)).squeeze(0)
            targets = nn.AdaptiveAvgPool2d((sizes[2], sizes[3]))(targets.unsqueeze(0)).squeeze(0)
        return m_iou / 4


class LSALoss(nn.Module):
    def __init__(self):
        super(LSALoss, self).__init__()

    @staticmethod
    def forward(inputs, targets):
        return NormalizedEuclideanLoss.forward(inputs, targets) + MIoU.forward(inputs, targets)


class ReduceMaxLROnRestart:
    def __init__(self, ratio=0.75):
        self.ratio = ratio

    def __call__(self, eta_min, eta_max):
        return eta_min, eta_max * self.ratio


class ExpReduceMaxLROnIteration:
    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, eta_min, eta_max, iterations):
        return eta_min, eta_max * self.gamma ** iterations


class CosinePolicy:
    def __call__(self, t_cur, restart_period):
        return 0.5 * (1. + math.cos(torch.pi * (t_cur / restart_period)))


class ArccosinePolicy:
    def __call__(self, t_cur, restart_period):
        return math.acos(max(-1, min(1, 2 * t_cur / restart_period - 1))) / math.pi


class TriangularPolicy:
    def __init__(self, triangular_step=0.5):
        self.triangular_step = triangular_step

    def __call__(self, t_cur, restart_period):
        inflection_point = self.triangular_step * restart_period
        point_of_triangle = (
            t_cur / inflection_point if t_cur < inflection_point
            else 1.0 - (t_cur - inflection_point) / (restart_period - inflection_point)
        )
        return point_of_triangle


class CyclicLRWithRestarts(LRScheduler):
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will expand/shrink
        policy: ["cosine", "arccosine", "triangular", "triangular2", "exp_range"]
        min_lr: minimum allowed learning rate
        verbose: print a message on every restart
        gamma: exponent used in "exp_range" policy
        eta_on_restart_cb: callback executed on every restart, adjusts max or min lr
        eta_on_iteration_cb: callback executed on every iteration, adjusts max or min lr
        triangular_step: adjusts ratio of increasing/decreasing phases for triangular policy


    Example:
        >>> scheduler = CyclicLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100, t_mult=2, last_epoch=-1, verbose=False,
                 policy="cosine", policy_fn=None, min_lr=1e-7, eta_on_restart_cb=None, eta_on_iteration_cb=None,
                 gamma=1.0, triangular_step=0.5):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))

        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                group.setdefault('minimum_lr', min_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))

        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

        self.min_lrs = [group['minimum_lr'] for group in optimizer.param_groups]

        self.base_weight_decays = [group['weight_decay'] for group in optimizer.param_groups]

        self.policy = policy
        self.eta_on_restart_cb = eta_on_restart_cb
        self.eta_on_iteration_cb = eta_on_iteration_cb
        if policy_fn is not None:
            self.policy_fn = policy_fn
        elif self.policy == "cosine":
            self.policy_fn = CosinePolicy()
        elif self.policy == "arccosine":
            self.policy_fn = ArccosinePolicy()
        elif self.policy == "triangular":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
        elif self.policy == "triangular2":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_restart_cb = ReduceMaxLROnRestart(ratio=0.5)
        elif self.policy == "exp_range":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_iteration_cb = ExpReduceMaxLROnIteration(gamma=gamma)

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.iteration = 0
        self.total_iterations = 0

        self.t_mult = t_mult
        self.verbose = verbose
        self.restart_period = math.ceil(restart_period)
        self.restarts = 0
        self.t_epoch = -1
        self.epoch = -1

        self.eta_min = 0
        self.eta_max = 1

        self.end_of_period = False
        self.batch_increments = []
        self._set_batch_increment()

    def _on_restart(self):
        if self.eta_on_restart_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_restart_cb(self.eta_min, self.eta_max)

    def _on_iteration(self):
        if self.eta_on_iteration_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_iteration_cb(self.eta_min, self.eta_max, self.total_iterations)

    def get_lr(self, t_cur):
        eta_t = (self.eta_min + (self.eta_max - self.eta_min) * self.policy_fn(t_cur, self.restart_period))

        weight_decay_norm_multi = math.sqrt(self.batch_size / (self.epoch_size * self.restart_period))

        lrs = [min_lr + (base_lr - min_lr) * eta_t for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]
        weight_decays = [
            base_weight_decay * eta_t * weight_decay_norm_multi for base_weight_decay in self.base_weight_decays
        ]

        if (self.t_epoch + 1) % self.restart_period < self.t_epoch:
            self.end_of_period = True

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart {} at epoch {}".format(self.restarts + 1, self.last_epoch))
            self.restart_period = math.ceil(self.restart_period * self.t_mult)
            self.restarts += 1
            self.t_epoch = 0
            self._on_restart()
            self.end_of_period = False

        return zip(lrs, weight_decays)

    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = torch.linspace(0, 1, batches_in_epoch).tolist()

    def step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        self.batch_step()

    def batch_step(self):
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self._on_iteration()
            self.iteration += 1
            self.total_iterations += 1
        except (IndexError):
            raise StopIteration("Epoch size and batch size used in the "
                                "training loop and while initializing "
                                "scheduler should be the same.")

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups, self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay


class CosineAnnealingWarmupRestarts(LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [
                base_lr + (self.max_lr - base_lr) * (
                    1 + math.cos(
                        math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
                    )
                ) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
