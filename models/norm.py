from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch
import functools

from copy import deepcopy
from conf import cfg


class NoNorm(nn.BatchNorm2d):
    """
    This is just placeholder, used when no norm is intended to use
    """

    def forward(self, x):
        return x


class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if (
            bn_layer.track_running_stats
            and bn_layer.running_var is not None
            and bn_layer.running_mean is not None
        ):
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

        self.current_mu = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(
                x, dim=0, unbiased=False, keepdim=False
            )  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(
                var.detach()
            )
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.source_mean.view(1, -1), self.source_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias


class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(
                x, dim=[0, 2, 3], unbiased=False, keepdim=False
            )  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(
                var.detach()
            )
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(
                1, -1, 1, 1
            )

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias


class RobustMedBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            y = x.transpose(0, 1)
            y = y.contiguous().view(x.size(1), -1)
            b_mean = y.median(dim=1)[0]
            b_var = ((y - b_mean.repeat(y.shape[1], 1).T) ** 2).sum(dim=1) / y.shape[1]

            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(
                var.detach()
            )
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(
                1, -1, 1, 1
            )

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias


class RobustMMBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            y = x.transpose(0, 1)
            y = y.contiguous().view(x.size(1), -1)
            b_mean = y.median(dim=1)[0]
            b_var = (torch.abs(y - b_mean[:, None]).median(dim=1)[0]) ** 2

            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(
                var.detach()
            )
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(
                1, -1, 1, 1
            )

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)


class BatchNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        use_tracked_mean=True,
        use_tracked_var=True,
    ):
        nn.BatchNorm2d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.use_tracked_mean = use_tracked_mean
        self.use_tracked_var = use_tracked_var

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)

        if self.training is not True:
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_var is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)
        elif self.training is True:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_mean is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)


class MedNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.5,
        affine=True,
        track_running_stats=True,
        use_tracked_mean=True,
        use_tracked_var=True,
    ):
        nn.BatchNorm2d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.use_tracked_mean = use_tracked_mean
        self.use_tracked_var = use_tracked_var

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.median(dim=1)[0]
        sigma2 = ((y - mu.repeat(y.shape[1], 1).T) ** 2).sum(dim=1) / y.shape[1]

        if self.training is not True:
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_var is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        elif self.training is True:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_mean is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)


# Median + MAD
class MMNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        use_tracked_mean=True,
        use_tracked_var=True,
    ):
        nn.BatchNorm2d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.use_tracked_mean = use_tracked_mean
        self.use_tracked_var = use_tracked_var

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.median(dim=1)[0]
        sigma2 = (torch.abs(y - mu.repeat(y.shape[1], 1).T).median(dim=1)[0]) ** 2

        if self.training is not True:
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_mean is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        elif self.training is True:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_mean is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)


# Median + Scaling MAD
class MsMNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        use_tracked_mean=True,
        use_tracked_var=True,
    ):
        nn.BatchNorm2d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.use_tracked_mean = use_tracked_mean
        self.use_tracked_var = use_tracked_var

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.median(dim=1)[0]

        std = torch.std(y, dim=1)
        mad = torch.abs(y - mu.repeat(y.shape[1], 1).T).median(dim=1)[0]
        scaling_factor = std / mad
        sigma2 = scaling_factor * (mad**2)

        if self.training is not True:
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_mean is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        elif self.training is True:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_mean is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)


# Mean of Median
class MoMNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        use_tracked_mean=True,
        use_tracked_var=True,
    ):
        nn.BatchNorm2d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.use_tracked_mean = use_tracked_mean
        self.use_tracked_var = use_tracked_var

    def cal_median_mean(self, x, subset_size=40):
        batch_size = x.size(0)

        assert batch_size % subset_size == 0, ValueError(
            f"batch size {batch_size} is not divided by subset size {subset_size}"
        )

        perm = torch.randperm(batch_size)

        mean_lst = []
        num_subset = batch_size // subset_size

        for i in range(0, num_subset):
            cur_perm = perm[subset_size * i : subset_size * (i + 1)]
            sub_samples = x[cur_perm]
            mu = (
                sub_samples.transpose(0, 1)
                .reshape(sub_samples.size(1), -1)
                .median(dim=1)[0]
            )
            mean_lst.append(mu)

        mean_lst = torch.stack(mean_lst, dim=0)
        mean_median = torch.mean(mean_lst, dim=0)

        return mean_median

    def forward(self, x):
        self._check_input_dim(x)
        median_mean = self.cal_median_mean(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = median_mean
        sigma2 = ((y - mu.repeat(y.shape[1], 1).T) ** 2).sum(dim=1) / y.shape[1]

        if self.training is not True:
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_mean is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        elif self.training is True:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_tracked_mean and self.running_mean is not None:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var and self.running_mean is not None:
                y = y / torch.sqrt(self.running_var.view(-1, 1) + self.eps)
            else:
                y = y / torch.sqrt(sigma2.view(-1, 1) + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)


def select_norm(norm_type, norm_power=0.2):
    if norm_type == "nobn":
        normlayer = functools.partial(NoNorm)

    # BatchNorm variants
    elif norm_type == "bn":
        normlayer = functools.partial(
            BatchNorm,
            affine=True,
            track_running_stats=True,
            use_tracked_mean=True,
            use_tracked_var=True,
        )
    elif norm_type == "RBN":
        normlayer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "BNwoA":
        normlayer = functools.partial(
            nn.BatchNorm2d, affine=False, track_running_stats=True
        )
    elif norm_type == "BNwoT":
        normlayer = functools.partial(
            BatchNorm,
            affine=True,
            track_running_stats=True,
            use_tracked_mean=False,
            use_tracked_var=False,
        )
    elif norm_type == "BNM":
        normlayer = functools.partial(
            BatchNorm,
            affine=True,
            track_running_stats=True,
            use_tracked_mean=True,
            use_tracked_var=False,
        )

    # MedianNorm variants
    elif norm_type == "med":
        normlayer = functools.partial(
            MedNorm,
            affine=True,
            track_running_stats=True,
            use_tracked_mean=False,
            use_tracked_var=False,
        )
    elif norm_type == "mm":
        normlayer = functools.partial(
            MMNorm,
            affine=True,
            track_running_stats=True,
            use_tracked_mean=False,
            use_tracked_var=False,
        )
    elif norm_type == "msm":
        normlayer = functools.partial(
            MsMNorm,
            affine=True,
            track_running_stats=True,
            use_tracked_mean=False,
            use_tracked_var=False,
        )
    elif norm_type == "mom":
        normlayer = functools.partial(
            MoMNorm,
            affine=True,
            track_running_stats=True,
            use_tracked_mean=False,
            use_tracked_var=False,
        )
    elif norm_type == "robustbn":
        normlayer = functools.partial(nn.BatchNorm2d)
    else:
        Exception("Norm is not selected")

    return normlayer


if __name__ == "__main__":
    a = select_norm("RNT")
    a(2)
    print(a)
