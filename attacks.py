import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models import *
import wandb


class ATTACK:
    def __init__(self, cfg, source, target, num_classes):
        self.cfg = cfg
        self.source = source
        self.target = target
        self.num_classes = num_classes
        self.iter = 0

    def update_target(self, outputs_clean, y, counter):
        if self.cfg.ATTACK.TARGETED:
            self.target = 0
            self.target_label = (y[self.target] + 1) % self.num_classes
        # else:
        #     acc_target_be = (outputs_clean[target].argmax() == y[target]).float()
        #     while acc_target_be.item() == 0.:
        #         target += 1
        #         acc_target_be = (outputs_clean[target].argmax() == y[target]).float()
        #         if target > self.cfg.TEST.BATCH_SIZE - self.source - 1:
        #             target = 0
        #     self.target = target

        self.counter = counter

    def generate_attacks(
        self,
        sur_model,
        x,
        y,
        randomize=False,
        epsilon=16 / 255,
        alpha=2 / 255,
        num_iter=10,
    ):
        source = self.source
        target = self.target

        fixed = torch.zeros_like(
            x.clone()[:-source], requires_grad=False
        )  # benign samples # torch.Size([190, 3, 32, 32])

        if randomize:
            delta_0 = torch.rand_like(x[-source:])
        else:
            delta_0 = 127.5 / 255

        adv = (
            torch.zeros_like(x.clone()[-source:]) - x[-source:] + delta_0
        ).requires_grad_(
            True
        )  # malcious # torch.Size([10, 3, 32, 32])
        adv_pad = torch.cat((fixed, adv), 0)  # torch.Size([200, 3, 32, 32])

        if self.cfg.ATTACK.TARGETED:
            for t in tqdm(range(num_iter), disable=True):
                x_adv = x + adv_pad
                out = sur_model(x_adv)
                loss = nn.CrossEntropyLoss(reduction="none")(
                    out[target].reshape(1, -1), self.target_label.reshape(1)
                )
                loss.backward()

                if self.cfg.wandb:
                    wandb.log(
                        {"loss/generate attack loss": loss.item(), "iter": self.iter}
                    )
                self.iter += 1

                print(
                    "Learning Progress :%2.2f %% , loss1 : %f "
                    % ((t + 1) / num_iter * 100, loss.item()),
                    end="\r",
                )

                adv.data = (adv - alpha * adv.grad.detach().sign()).clamp(
                    -epsilon, epsilon
                )
                adv.data = (adv.data + x[-source:]).clamp(0, 1) - (x[-source:])
                adv_pad.data = torch.cat((fixed, adv), 0)
                adv.grad.zero_()
        else:
            for t in tqdm(range(num_iter), disable=True):
                x_adv = x + adv_pad  # benign + initialize malcious sample (127.5 / 255)
                out = sur_model(x_adv)
                loss = nn.CrossEntropyLoss(reduction="none")(
                    out[:-source], y[:-source]
                ).clamp(min=0, max=5)
                loss = loss.sum()

                loss.backward()
                if self.cfg.wandb:
                    wandb.log(
                        {"loss/generate attack loss": loss.item(), "iter": self.iter}
                    )
                self.iter += 1
                # if loss.item() > 1:
                #     break

                print(
                    "Learning Progress :%2.2f %% , loss1 : %f "
                    % ((t + 1) / num_iter * 100, loss.item()),
                    end="\r",
                )
        
                adv.data = (adv + alpha * adv.grad.detach().sign()).clamp(
                    -epsilon, epsilon
                )
                adv.data = (adv.data + x[-source:]).clamp(0, 1) - (x[-source:])
                adv_pad.data = torch.cat((fixed, adv), 0)
                adv.grad.zero_()

        print(loss.item())
        x_adv = x + adv_pad
        return x_adv

    def compute_acc(self, outputs_clean, outputs_adv, y):
        target = self.target
        source = self.source
        acc_target_be = (outputs_clean[target].argmax() == y[target]).float()
        acc_source_be = (outputs_clean.max(1)[1][-source:] == y[-source:]).float().sum()
        acc_clean = (outputs_clean.max(1)[1] == y).float().sum()
        acc_adv = (outputs_adv.max(1)[1] == y).float().sum()
        acc_target_af = (outputs_adv[target].argmax() == y[target]).float()
        acc_source_af = (outputs_adv.max(1)[1][-source:] == y[-source:]).float().sum()

        acc_benign_be = (outputs_clean.max(1)[1][:-source] == y[:-source]).float().sum()
        acc_benign_af = (outputs_adv.max(1)[1][:-source] == y[:-source]).float().sum()
        if self.cfg.ATTACK.TARGETED:
            acc_target_af = (outputs_adv[target].argmax() == self.target_label).float()

        return (
            acc_target_be,
            acc_target_af,
            acc_clean,
            acc_adv,
            acc_source_be,
            acc_source_af,
            acc_benign_be,
            acc_benign_af,
        )
