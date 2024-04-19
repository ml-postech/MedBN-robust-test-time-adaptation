import torch
import torch.utils.data

from robustbench.data import load_cifar10c, load_cifar100c


def prepare_test_data(cfg, corruption, severity):
    if cfg.CORRUPTION.DATASET == "cifar10":
        x_test, y_test = load_cifar10c(
            cfg.CORRUPTION.NUM_EX,
            severity,
            cfg.DATA_DIR,
            False,
            [corruption],
        )
        return x_test, y_test

    elif cfg.CORRUPTION.DATASET == "cifar100":
        x_test, y_test = load_cifar100c(
            cfg.CORRUPTION.NUM_EX,
            severity,
            cfg.DATA_DIR,
            False,
            [corruption],
        )
        return x_test, y_test

    else:
        print("ERROR: no valid datatset provided")
