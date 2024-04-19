import os
import torch

import random
import copy
import numpy as np
import math

from conf import cfg, load_cfg_fom_args
from utils import memory
from utils.data_utils import prepare_test_data
from utils.train_utils import get_model, get_log_name

from attack_adaptive import test_attack_adaptive

torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.enabled = True


def evaluate(description):
    load_cfg_fom_args(description)
    if cfg.wandb:
        import wandb

        wandb.login()
        wandb.init(
            project=cfg.project,
            config=cfg,
        )

        log_name = get_log_name(cfg)
        print(log_name)
        wandb.run.name = log_name
        wandb.config.update(cfg)

        wandb.define_metric("iter")
        wandb.define_metric("batch")
        wandb.define_metric("corruption")
        # define which metrics will be plotted against it
        wandb.define_metric("loss/generate attack loss", step_metric="iter")
        wandb.define_metric("loss/tta loss of model", step_metric="batch")
        wandb.define_metric("results_all/acc_clean_all", step_metric="corruption")

    num_classes_dic = {
        "cifar10": 10,
        "cifar100": 100,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = get_model(cfg, device)

    sotta_mem = None

    corruption, batch_counter = 0, 0
    for _, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        (tb, ta, sb, sa, bb, ba, clean, adv) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for _, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            if cfg.MODEL.ADAPTATION == "sotta":
                sotta_mem = memory.HUS(
                    capacity=cfg.HYP.MEM_SIZE, threshold=cfg.HYP.HIGH_THRESHOLD
                )

            num_classes = num_classes_dic[cfg.CORRUPTION.DATASET]

            x_test, y_test = prepare_test_data(cfg, corruption_type, severity)
            x_test, y_test = x_test.cuda(), y_test.cuda()
            y_test = y_test.type(torch.LongTensor)

            print(f"meta test begin on {cfg.CORRUPTION.DATASET}-C!")
            # reset the model
            net_test = copy.deepcopy(net)

            (
                acc_target_be_all,
                acc_target_af_all,
                acc_clean_all,
                acc_adv_all,
                acc_source_be_all,
                acc_source_af_all,
                acc_benign_be_all,
                acc_benign_af_all,
                batch_counter,
            ) = test_attack_adaptive(
                net_test,
                device,
                x_test,
                y_test,
                cfg.TEST.BATCH_SIZE,
                cfg.OPTIM.STEPS,
                use_test_bn=cfg.OPTIM.TBN,
                num_classes=num_classes,
                update=cfg.OPTIM.UPDATE,
                batch_counter=batch_counter,
                sotta_mem=sotta_mem,
            )

            if cfg.wandb:
                num_mal = cfg.ATTACK.SOURCE
                n_batches = math.ceil(x_test.shape[0] / cfg.TEST.BATCH_SIZE)

                wandb.log(
                    {
                        "results_all/acc_target_before_all": acc_target_be_all.item()
                        / n_batches,
                        "results_all/acc_target_after_all": acc_target_af_all.item()
                        / n_batches,
                        "results_all/acc_source_before_all": acc_source_be_all.item()
                        / (num_mal * n_batches),
                        "results_all/acc_source_after_all": acc_source_af_all.item()
                        / (num_mal * n_batches),
                        "results_all/acc_benign_before_all": acc_benign_be_all.item()
                        / ((cfg.TEST.BATCH_SIZE - num_mal) * n_batches),
                        "results_all/acc_benign_after_all": acc_benign_af_all.item()
                        / ((cfg.TEST.BATCH_SIZE - num_mal) * n_batches),
                        "results_all/acc_clean_all": acc_clean_all / x_test.shape[0],
                        "results_all/acc_adv_all": acc_adv_all / x_test.shape[0],
                        "corruption": corruption,
                    }
                )
                #####
                tb += acc_target_be_all.item() / n_batches
                ta += acc_target_af_all.item() / n_batches
                sb += acc_source_be_all.item() / (num_mal * n_batches)
                sa += acc_source_af_all.item() / (num_mal * n_batches)
                bb += acc_benign_be_all.item() / (
                    (cfg.TEST.BATCH_SIZE - num_mal) * n_batches
                )
                ba += acc_benign_af_all.item() / (
                    (cfg.TEST.BATCH_SIZE - num_mal) * n_batches
                )
                clean += acc_clean_all / x_test.shape[0]
                adv += acc_adv_all / x_test.shape[0]
                #####
            corruption += 1

        if cfg.ATTACK.METHOD != None and cfg.wandb:
            wandb.log(
                {
                    "results_avg/acc_target_before": tb / len(cfg.CORRUPTION.TYPE),
                    "results_avg/acc_target_after": ta / len(cfg.CORRUPTION.TYPE),
                    "results_avg/acc_source_before": sb / len(cfg.CORRUPTION.TYPE),
                    "results_avg/acc_source_after": sa / len(cfg.CORRUPTION.TYPE),
                    "results_avg/acc_benign_before": bb / len(cfg.CORRUPTION.TYPE),
                    "results_avg/acc_benign_after": ba / len(cfg.CORRUPTION.TYPE),
                    "results_avg/acc_clean": clean / len(cfg.CORRUPTION.TYPE),
                    "results_avg/acc_adv": adv / len(cfg.CORRUPTION.TYPE),
                }
            )


if __name__ == "__main__":
    evaluate("TTA evaluation.")
