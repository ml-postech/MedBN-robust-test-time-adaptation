from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.utils.data

from torch import Tensor
from typing import Tuple

from models import *
from models import *
from models.norm import (
    RobustBN1d,
    RobustBN2d,
    RobustMedBN2d,
    RobustMMBN2d,
    get_named_submodule,
    set_named_submodule,
)
from models.resnet_norm import Normalized_ResNet_NORM
from models.resnet_norm_cifar100 import Normalized_ResNet_NORM_CIFAR100

from datetime import datetime


def get_model(cfg, device):
    if cfg.CORRUPTION.DATASET in ["cifar10"]:
        if cfg.MODEL.ADAPTATION == "sotta" or cfg.TEST.EMA:
            if cfg.MODEL.ARCH == "resnet26":
                ckpt_path = "./pretrained/cifar10_res_ce.pth"
                net = Normalized_ResNet_NORM(depth=26, norm_type="robustbn")
                checkpoint = torch.load(ckpt_path)["net"]
            check = OrderedDict()
            for key in checkpoint.keys():
                check[key.replace("module.", "")] = checkpoint[key]

            net.load_state_dict(check, strict=True)

            normlayer_names = []
            for name, sub_module in net.named_modules():
                if isinstance(sub_module, nn.BatchNorm1d) or isinstance(
                    sub_module, nn.BatchNorm2d
                ):
                    normlayer_names.append(name)

            for name in normlayer_names:
                bn_layer = get_named_submodule(net, name)
                if isinstance(bn_layer, nn.BatchNorm1d):
                    NewBN = RobustBN1d
                elif isinstance(bn_layer, nn.BatchNorm2d):
                    if cfg.MODEL.NORM == "bn":
                        NewBN = RobustBN2d
                    elif cfg.MODEL.NORM == "med":
                        NewBN = RobustMedBN2d
                    elif cfg.MODEL.NORM == "mm":
                        NewBN = RobustMMBN2d
                else:
                    raise RuntimeError()

                momentum_bn = NewBN(bn_layer, cfg.TEST.BN_MOMENTUM)
                momentum_bn.requires_grad_(True)
                set_named_submodule(net, name, momentum_bn)
            net = net.to(device)
        else:
            if cfg.MODEL.ARCH == "resnet26":
                ckpt_path = "./pretrained/cifar10_res_ce.pth"
                net = Normalized_ResNet_NORM(depth=26, norm_type=cfg.MODEL.NORM)
                net = torch.nn.DataParallel(net)
                checkpoint = torch.load(ckpt_path)
                net.load_state_dict(checkpoint["net"], strict=True)
                net = net.module.to(device)

    elif cfg.CORRUPTION.DATASET in ["cifar100"]:
        num_classes = 100
        if cfg.MODEL.ADAPTATION == "sotta" or cfg.TEST.EMA:
            if cfg.MODEL.ARCH == "resnet26":
                ckpt_path = "./pretrained/cifar100_res_ce.pth"  # edit this path to the checkpoint of the model you want to evaluate
                net = Normalized_ResNet_NORM_CIFAR100(
                    depth=26, norm_type="robustbn", num_classes=num_classes
                )
                checkpoint = torch.load(ckpt_path)["net"]
                check = OrderedDict()
                for key in checkpoint.keys():
                    check[key.replace("module.", "")] = checkpoint[key]

                net.load_state_dict(check, strict=True)

            normlayer_names = []
            for name, sub_module in net.named_modules():
                if isinstance(sub_module, nn.BatchNorm1d) or isinstance(
                    sub_module, nn.BatchNorm2d
                ):
                    normlayer_names.append(name)

            for name in normlayer_names:
                bn_layer = get_named_submodule(net, name)
                if isinstance(bn_layer, nn.BatchNorm1d):
                    NewBN = RobustBN1d
                elif isinstance(bn_layer, nn.BatchNorm2d):
                    if cfg.MODEL.NORM == "bn":
                        NewBN = RobustBN2d
                    elif cfg.MODEL.NORM == "med":
                        NewBN = RobustMedBN2d
                    elif cfg.MODEL.NORM == "mm":
                        NewBN = RobustMMBN2d
                else:
                    raise RuntimeError()

                momentum_bn = NewBN(bn_layer, cfg.TEST.BN_MOMENTUM)
                momentum_bn.requires_grad_(True)
                set_named_submodule(net, name, momentum_bn)
            net = net.to(device)
        else:  # cfg.MODEL.ADAPTATION != "sotta":
            if cfg.MODEL.ARCH == "resnet26":
                ckpt_path = "./pretrained/cifar100_res_ce.pth"  # edit this path to the checkpoint of the model you want to evaluate
                net = Normalized_ResNet_NORM_CIFAR100(
                    depth=26, norm_type=cfg.MODEL.NORM, num_classes=num_classes
                )
                net = torch.nn.DataParallel(net)
                checkpoint = torch.load(ckpt_path)
                net.load_state_dict(checkpoint["net"], strict=True)
                net = net.module.to(device)

    cudnn.benchmark = True
    return net


def get_log_name(cfg):
    date = datetime.today().strftime("%y%m%d")
    arch = cfg.MODEL.ARCH
    norm = cfg.MODEL.NORM
    if cfg.MODEL.NORM == "RBN":
        norm = "bn" if cfg.MODEL.BN_STAT == "mean" else "med"

    adaptation = cfg.MODEL.ADAPTATION + ("_continual" if cfg.MODEL.CONTINUAL else "")
    dataset = cfg.CORRUPTION.DATASET

    if cfg.ATTACK.METHOD is None:
        log_name = f"{date}_{arch}{norm}_{adaptation}_{dataset}"

        return log_name

    # attack
    method = cfg.ATTACK.METHOD
    source = cfg.ATTACK.SOURCE
    target = cfg.ATTACK.TARGETED
    attack = "target attack" if target else "indiscriminate attack"
    eps = cfg.ATTACK.EPS
    alpha = cfg.ATTACK.ALPHA
    steps = cfg.ATTACK.STEPS
    ema = f"_track{cfg.TEST.BN_MOMENTUM}" if cfg.TEST.EMA else ""
    severity = cfg.CORRUPTION.SEVERITY
    batch_size = cfg.TEST.BATCH_SIZE
    option = "_option" if cfg.ATTACK.OPTION else ""
    rand = "_rand" if cfg.ATTACK.RAND else ""
    log_name = f"{date}_{arch}{norm}{ema}{option}{rand}_{adaptation}_{dataset}_s{severity}_steps{steps}_bs{batch_size}_source{source}_{attack}_eps{eps}_alpha{alpha:.4f}"
    return log_name
