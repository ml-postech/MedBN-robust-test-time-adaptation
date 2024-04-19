import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from conf import cfg

import copy
import math

import methods.bn as bn
from methods.sotta import filter_add_batch
import methods.tent as tent
from methods.tent import (
    copy_model_and_optimizer,
    load_model_and_optimizer,
    softmax_entropy,
)
from methods.eata import update_model_probs
from methods.sar import update_ema

from attacks import ATTACK

from robustbench.data import load_cifar10, load_cifar100
from utils.sam_optimizer import SAM, sam_collect_params

import wandb


def setup_optimizer(params, lr_test=None):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """

    if lr_test is None:
        lr_test = cfg.OPTIM.LR

    if cfg.OPTIM.ADAPT == "sar":
        if cfg.OPTIM.METHOD == "Adam":
            return SAM(
                params,
                torch.optim.Adam,
                rho=0.05,
                lr=lr_test,
                weight_decay=cfg.OPTIM.WD,
            )
        elif cfg.OPTIM.METHOD == "SGD":
            return SAM(params, torch.optim.SGD, lr=lr_test, momentum=cfg.OPTIM.MOMENTUM)
    elif cfg.OPTIM.ADAPT == "sotta":
        return SAM(
            params,
            torch.optim.Adam,
            rho=0.05,
            lr=cfg.OPTIM.LR,
            weight_decay=0,
        )
    else:
        if cfg.OPTIM.METHOD == "Adam":
            return optim.Adam(
                params,
                lr=lr_test,
                betas=(cfg.OPTIM.BETA, 0.999),
                weight_decay=cfg.OPTIM.WD,
            )
        elif cfg.OPTIM.METHOD == "SGD":
            return optim.SGD(
                params,
                lr=lr_test,
                momentum=cfg.OPTIM.MOMENTUM,
                dampening=cfg.OPTIM.DAMPENING,
                weight_decay=cfg.OPTIM.WD,
                nesterov=cfg.OPTIM.NESTEROV,
            )

def test_attack_adaptive(
    model,
    device,
    x_test,
    y_test,
    batch_size,
    n_inner_iter=1,
    use_test_bn=True,
    num_classes=10,
    update=True,
    batch_counter=0,
    sotta_mem=None,
):
    if use_test_bn:
        model = tent.configure_model(cfg, model)
    else:
        model = tent.configure_model_eval(model)
    
    if cfg.MODEL.ADAPTATION == "RBN":
        model = bn.adapt_robustBN(model, cfg.ATTACK.DFPIROR, cfg.ATTACK.FLayer)
    if cfg.OPTIM.ADAPT == "sar":
        params, _ = sam_collect_params(model, freeze_top=True)
        ema = None
    elif cfg.OPTIM.ADAPT == "sotta":
        params, _ = sam_collect_params(model, freeze_top=True)
    else:
        params, _ = tent.collect_params(model)

    # optimizer
    inner_opt = setup_optimizer(params)

    n_batches = math.ceil(x_test.shape[0] / batch_size)
    (
        acc_target_be_all,
        acc_target_af_all,
        acc_clean_all,
        acc_adv_all,
        acc_source_be_all,
        acc_source_af_all,
        acc_benign_be_all,
        acc_benign_af_all,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if cfg.OPTIM.ADAPT in ["eata", "eta"]:
        current_model_probs, current_victim_model_probs = None, None
        fishers = None
        ids1, ids2 = None, None

        if cfg.OPTIM.ADAPT == "eata":
            fisher_net = copy.deepcopy(model).cuda()
            source_x_test, source_y_test = (
                load_cifar10()
                if cfg.CORRUPTION.DATASET == "cifar10"
                else load_cifar100()
            )

            source_x_test, source_y_test = (
                source_x_test.cuda(),
                source_y_test.cuda(),
            )

            params, _ = tent.collect_params(fisher_net)
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().cuda()

            source_n_batches = math.ceil(source_x_test.shape[0] / batch_size)
            for iter_ in range(source_n_batches):
                x_fisher = source_x_test[
                    iter_ * batch_size : (iter_ + 1) * batch_size
                ].to(device)
                # y_fisher = source_y_test[
                #     iter_ * batch_size : (iter_ + 1) * batch_size
                # ].to(device)
                outputs = fisher_net(x_fisher)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in fisher_net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = (
                                param.grad.data.clone().detach() ** 2 + fishers[name][0]
                            )
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == source_n_batches:
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            del ewc_optimizer

    victim_model = copy.deepcopy(model).cuda()
    if cfg.OPTIM.ADAPT in ["sar", "sotta"]:
        params_victim, _ = sam_collect_params(victim_model, freeze_top=True)
    else:
        params_victim, _ = tent.collect_params(victim_model)
    inner_opt_victim = setup_optimizer(params_victim)

    if cfg.ATTACK.WHITE:
        sur_model = copy.deepcopy(model).cuda()
        if cfg.OPTIM.ADAPT in ["sar", "sotta"]:
            params_sur, _ = sam_collect_params(sur_model, freeze_top=True)
        else:
            params_sur, _ = tent.collect_params(sur_model)
        inner_opt_sur = setup_optimizer(params_sur)

    attack = ATTACK(
        cfg, source=cfg.ATTACK.SOURCE, target=cfg.ATTACK.TARGET, num_classes=num_classes
    )

    for counter in range(n_batches):
        x_curr = x_test[counter * batch_size : (counter + 1) * batch_size].to(device)
        y_curr = y_test[counter * batch_size : (counter + 1) * batch_size].to(device)

        if cfg.ATTACK.OPTION:
            model_state, optimizer_state = copy_model_and_optimizer(model, inner_opt)
            load_model_and_optimizer(
                victim_model, inner_opt_victim, model_state, optimizer_state
            )
        else:
            model_state, optimizer_state = copy_model_and_optimizer(model, inner_opt)
            load_model_and_optimizer(
                sur_model, inner_opt_sur, model_state, optimizer_state
            )
            load_model_and_optimizer(
                victim_model, inner_opt_victim, model_state, optimizer_state
            )

        ### for main model
        if update:
            for _ in range(n_inner_iter):
                if cfg.OPTIM.ADAPT == "sotta":
                    # backup prev mem
                    prev_sotta_mem_state = sotta_mem.save_state_dict()
                    # filter current batch and save it to sotta_mem
                    sotta_mem = filter_add_batch(model, sotta_mem, x_curr)
                    sotta_mem_state = sotta_mem.save_state_dict()
                    feats, _, _ = sotta_mem.get_memory()
                    if len(feats) == 0:
                        print("sotta mem has 0 data")
                        continue
                    filtered_x = torch.stack(feats)
                    outputs = model(filtered_x)
                else:
                    outputs = model(x_curr)
                outputs = outputs / cfg.OPTIM.TEMP

                tta_loss = 0
                if cfg.OPTIM.ADAPT == "ent":
                    tta_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                elif cfg.OPTIM.ADAPT in ["eta", "eata"]:
                    entropys = softmax_entropy(outputs)
                    # filter unreliable samples
                    filter_ids_1 = torch.where(entropys < cfg.HYP.E_MARGIN)
                    ids1 = filter_ids_1
                    ids2 = torch.where(ids1[0] > -0.1)
                    entropys = entropys[filter_ids_1]
                    # filter redundant samples
                    if current_model_probs is not None:
                        cosine_similarities = F.cosine_similarity(
                            current_model_probs.unsqueeze(dim=0),
                            outputs[filter_ids_1].softmax(1),
                            dim=1,
                        )
                        filter_ids_2 = torch.where(
                            torch.abs(cosine_similarities) < cfg.HYP.D_MARGIN
                        )
                        entropys = entropys[filter_ids_2]
                        ids2 = filter_ids_2
                        updated_probs = update_model_probs(
                            current_model_probs,
                            outputs[filter_ids_1][filter_ids_2].softmax(1),
                        )
                    else:
                        updated_probs = update_model_probs(
                            current_model_probs, outputs[filter_ids_1].softmax(1)
                        )
                    coeff = 1 / (
                        torch.exp(entropys.clone().detach() - cfg.HYP.E_MARGIN)
                    )
                    current_model_probs = updated_probs
                    # reweight entropy losses for diff. samples
                    tta_loss = entropys.mul(coeff)
                elif cfg.OPTIM.ADAPT == "sar":
                    tta_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                    filter_ids_1 = torch.where(tta_loss < cfg.HYP.E_MARGIN)
                    tta_loss = tta_loss[filter_ids_1]
                elif cfg.OPTIM.ADAPT == "sotta":
                    tta_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                else:
                    pass

                if len(tta_loss) > 0:
                    tta_loss = tta_loss.mean(0)
                    if cfg.OPTIM.ADAPT in ["eta", "eata"]:
                        if fishers is not None:
                            ewc_loss = 0
                            for name, param in model.named_parameters():
                                if name in fishers:
                                    ewc_loss += (
                                        cfg.HYP.FISHER_ALPHA
                                        * (
                                            fishers[name][0]
                                            * (param - fishers[name][1]) ** 2
                                        ).sum()
                                    )
                            tta_loss += ewc_loss
                        if x_curr[ids1][ids2].size(0) != 0:
                            tta_loss.backward()
                            inner_opt.step()
                    elif cfg.OPTIM.ADAPT == "sar":
                        inner_opt.zero_grad()
                        # first backward
                        tta_loss.backward()
                        inner_opt.first_step(zero_grad=True)
                        # second backward
                        outputs = model(x_curr)
                        second_loss = -(
                            outputs.softmax(1) * outputs.log_softmax(1)
                        ).sum(1)
                        filter_ids_2 = torch.where(second_loss < cfg.HYP.E_MARGIN)
                        second_loss = second_loss[filter_ids_2]
                        second_loss = second_loss.mean()
                        if not np.isnan(second_loss.item()):
                            # record moving average loss values for model recovery
                            ema = update_ema(ema, second_loss.item())
                        second_loss.backward()
                        inner_opt.second_step(zero_grad=True)
                    elif cfg.OPTIM.ADAPT == "sotta":
                        inner_opt.zero_grad()
                        # first backward
                        tta_loss.backward()
                        inner_opt.first_step(zero_grad=True)
                        # second forward
                        outputs = model(filtered_x)
                        second_loss = (
                            -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1).mean()
                        )
                        second_loss.backward()
                        inner_opt.second_step(zero_grad=True)
                    else:
                        tta_loss.requires_grad_(True)
                        tta_loss.backward()
                        inner_opt.step()
                    inner_opt.zero_grad()
                if cfg.wandb:
                    wandb.log(
                        {"loss/tta loss of model": tta_loss, "batch": batch_counter}
                    )

        with torch.no_grad():
            outputs_clean = model(x_curr)
        attack.update_target(outputs_clean, y_curr, counter)

        if cfg.ATTACK.METHOD == "PGD":
            x_adv = attack.generate_attacks(
                sur_model=sur_model,
                x=x_curr,
                y=y_curr,
                randomize=cfg.ATTACK.RAND,
                epsilon=cfg.ATTACK.EPS,
                alpha=cfg.ATTACK.ALPHA,
                num_iter=cfg.ATTACK.STEPS,
            )

        ### for victim model
        if update:
            for _ in range(n_inner_iter):
                victim_model.train()
                if cfg.OPTIM.ADAPT == "sotta":
                    # reset mem to previous state
                    sotta_mem.set_memory(prev_sotta_mem_state)
                    # filter current batch and save it to sotta_mem
                    sotta_mem = filter_add_batch(victim_model, sotta_mem, x_adv)
                    feats, _, _ = sotta_mem.get_memory()

                    # reset mem to current state
                    sotta_mem.set_memory(sotta_mem_state)
                    if len(feats) == 0:
                        print("(attacked) sotta mem has 0 data")
                        continue
                    filtered_x_adv = torch.stack(feats)
                    outputs = victim_model(filtered_x_adv)
                else:
                    outputs = victim_model(x_adv)
                outputs = outputs / cfg.OPTIM.TEMP

                tta_loss = 0
                if cfg.OPTIM.ADAPT == "ent":
                    tta_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                elif cfg.OPTIM.ADAPT in ["eta", "eata"]:
                    entropys = softmax_entropy(outputs)
                    # filter unreliable samples
                    filter_ids_1 = torch.where(entropys < cfg.HYP.E_MARGIN)
                    ids1 = filter_ids_1
                    ids2 = torch.where(ids1[0] > -0.1)
                    entropys = entropys[filter_ids_1]
                    # filter redundant samples
                    if current_victim_model_probs is not None:
                        cosine_similarities = F.cosine_similarity(
                            current_victim_model_probs.unsqueeze(dim=0),
                            outputs[filter_ids_1].softmax(1),
                            dim=1,
                        )
                        filter_ids_2 = torch.where(
                            torch.abs(cosine_similarities) < cfg.HYP.D_MARGIN
                        )
                        entropys = entropys[filter_ids_2]
                        ids2 = filter_ids_2
                        updated_probs = update_model_probs(
                            current_victim_model_probs,
                            outputs[filter_ids_1][filter_ids_2].softmax(1),
                        )
                    else:
                        updated_probs = update_model_probs(
                            current_victim_model_probs,
                            outputs[filter_ids_1].softmax(1),
                        )
                    coeff = 1 / (
                        torch.exp(entropys.clone().detach() - cfg.HYP.E_MARGIN)
                    )
                    current_victim_model_probs = updated_probs
                    # reweight entropy losses for diff. samples
                    tta_loss = entropys.mul(coeff)
                elif cfg.OPTIM.ADAPT == "sar":
                    tta_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                    filter_ids_1 = torch.where(tta_loss < cfg.HYP.E_MARGIN)
                    tta_loss = tta_loss[filter_ids_1]
                elif cfg.OPTIM.ADAPT == "sotta":
                    tta_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                else:
                    pass
                if len(tta_loss) > 0:
                    tta_loss = tta_loss.mean()
                    if cfg.OPTIM.ADAPT in ["eta", "eata"]:
                        if fishers is not None:
                            ewc_loss = 0
                            for name, param in victim_model.named_parameters():
                                if name in fishers:
                                    ewc_loss += (
                                        cfg.HYP.FISHER_ALPHA
                                        * (
                                            fishers[name][0]
                                            * (param - fishers[name][1]) ** 2
                                        ).sum()
                                    )
                            tta_loss += ewc_loss

                        if x_curr[ids1][ids2].size(0) != 0:
                            tta_loss.backward()
                            inner_opt_victim.step()
                    elif cfg.OPTIM.ADAPT == "sar":
                        inner_opt_victim.zero_grad()
                        # first backward
                        tta_loss.backward()
                        inner_opt_victim.first_step(zero_grad=True)
                        # second backward
                        outputs = victim_model(x_curr)
                        second_loss = -(
                            outputs.softmax(1) * outputs.log_softmax(1)
                        ).sum(1)
                        filter_ids_2 = torch.where(second_loss < cfg.HYP.E_MARGIN)
                        second_loss = second_loss[filter_ids_2]
                        second_loss = second_loss.mean()
                        if not np.isnan(second_loss.item()):
                            # record moving average loss values for model recovery
                            ema = update_ema(ema, second_loss.item())
                        second_loss.backward()
                        inner_opt_victim.second_step(zero_grad=True)
                    elif cfg.OPTIM.ADAPT == "sotta":
                        inner_opt_victim.zero_grad()
                        # first backward
                        tta_loss.backward()
                        inner_opt_victim.first_step(zero_grad=True)
                        # second forward
                        outputs = victim_model(filtered_x_adv)
                        second_loss = (
                            -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1).mean()
                        )
                        second_loss.backward()
                        inner_opt_victim.second_step(zero_grad=True)
                    else:
                        tta_loss.requires_grad_(True)
                        tta_loss.backward()
                        inner_opt_victim.step()
                    inner_opt_victim.zero_grad()

                if cfg.wandb:
                    wandb.log(
                        {
                            "loss/tta loss of victim model": tta_loss,
                            "batch": batch_counter,
                        }
                    )
        with torch.no_grad():
            outputs_adv = victim_model(x_adv)

        if cfg.MODEL.CONTINUAL:
            victim_model_state, victim_optimizer_state = copy_model_and_optimizer(
                victim_model, inner_opt_victim
            )
            load_model_and_optimizer(
                model, inner_opt, victim_model_state, victim_optimizer_state
            )

        (
            acc_target_be,  # adv before target accuracy
            acc_target_af,  # adv affter target accuracy
            acc_clean,  # accuracy before adv
            acc_adv,
            acc_source_be,  # adv before source accuracy
            acc_source_af,
            acc_benign_be,
            acc_benign_af,
        ) = attack.compute_acc(outputs_clean, outputs_adv, y_curr)
        num_mal = cfg.ATTACK.SOURCE
        batch_size = cfg.TEST.BATCH_SIZE
        
        if cfg.wandb:
            wandb.log(
                {
                    "results/acc_target_before": acc_target_be.item(),
                    "results/acc_target_after": acc_target_af.item(),
                    "results/acc_source_before": acc_source_be.item() / num_mal,
                    "results/acc_source_after": acc_source_af.item() / num_mal,
                    "results/acc_benign_before": acc_benign_be.item()
                    / (batch_size - num_mal),
                    "results/acc_benign_after": acc_benign_af.item()
                    / (batch_size - num_mal),
                    "results/acc_clean": acc_clean / batch_size,
                    "results/acc_adv": acc_adv / batch_size,
                    "batch": batch_counter,
                }
            )

        acc_target_be_all += acc_target_be
        acc_target_af_all += acc_target_af
        acc_clean_all += acc_clean
        acc_adv_all += acc_adv
        acc_source_be_all += acc_source_be
        acc_source_af_all += acc_source_af
        acc_benign_be_all += acc_benign_be
        acc_benign_af_all += acc_benign_af

        batch_counter += 1

    return (
        acc_target_be_all,
        acc_target_af_all,
        acc_clean_all,
        acc_adv_all,
        acc_source_be_all,
        acc_source_af_all,
        acc_benign_be_all,
        acc_benign_af_all,
        batch_counter,
    )
