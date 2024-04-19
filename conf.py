import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode
import wandb


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH = "Standard"

# Choice of (source, norm, tent, ...)
_C.MODEL.ADAPTATION = "source"
_C.MODEL.NORM = "bn"
_C.MODEL.BN_STAT = "mean"
_C.MODEL.CONTINUAL = False

_C.MODEL.CKPT_PATH = "."
_C.MODEL.EPS = 0.0

# ----------------------------- Hyp(erparameter) options -------------------------- #
_C.HYP = CfgNode()
# EATA
_C.HYP.FISHER_ALPHA = 1
_C.HYP.D_MARGIN = 0.0
_C.HYP.E_MARGIN = 0.0

# SoTTA
_C.HYP.MEM_SIZE = 64
_C.HYP.UPDATE_EVERY_X = 64
_C.HYP.USE_LEARNED_STATS = True
_C.HYP.TEMPERATURE = 1.0
_C.HYP.HIGH_THRESHOLD = 0.99
# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = "cifar10"

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
_C.CORRUPTION.SEVERITY = [3]

# Number of examples to evaluate (10000 for all samples in CIFAR-10)
_C.CORRUPTION.NUM_EX = 10000

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = "Adam"

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0
_C.OPTIM.TEMP = 1.0

# Set up in the config file (config.py)
_C.OPTIM.ADAPT = "ent"
_C.OPTIM.ADAPTIVE = False
_C.OPTIM.TBN = True
_C.OPTIM.UPDATE = True


# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 128
_C.TEST.NUM_CLASS = 10

# Statistics
_C.TEST.EMA = False
_C.TEST.BN_MOMENTUM = 0.2


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# ------------------------------- Attacking options --------------------------- #
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

_C.ATTACK = CfgNode()

# Attack methods
_C.ATTACK.METHOD = "PGD"

# The number of malicious sampes
_C.ATTACK.SOURCE = 10

# Targeted Number
_C.ATTACK.TARGET = 1

# L_inf bound
_C.ATTACK.EPS = 1.0

# attack update rate
_C.ATTACK.ALPHA = 0.00392157

# attack steps
_C.ATTACK.STEPS = 500

# attack white box, model is known
_C.ATTACK.WHITE = True

# the attack is targeted or not
_C.ATTACK.TARGETED = False

# the initialization point is random
_C.ATTACK.RAND = False

_C.ATTACK.OPTION = False

_C.ATTACK.WEIGHT_P = 0.0
_C.ATTACK.DFPIROR = 0.0
_C.ATTACK.DFTESTPIROR = 0.0
_C.ATTACK.FLayer = 0

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# ---------------------------------- Misc options --------------------------- #

# Optional description of a config
_C.DESC = ""

# Note that non-determinism is still present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Output directory
_C.SAVE_DIR = "./outputs"

# Data directory
_C.DATA_DIR = "./data"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ""


# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--cfg", dest="cfg_file", type=str, required=True, help="Config file location"
    )

    # wandb
    parser.add_argument(
        "--wandb", action="store_true", help="Track the result using wandb"
    )
    parser.add_argument(
        "--project", type=str, default="tta-defense", help="wandb project name"
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="See conf.py for all options",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    merge_from_file(args.cfg_file)
    print(args.opts)
    cfg.merge_from_list(args.opts)

    cfg.wandb = args.wandb
    cfg.project = args.project

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace(".yaml", "_{}.txt".format(current_time))

    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler(),
        ],
    )

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)
