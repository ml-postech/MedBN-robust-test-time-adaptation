from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F




def filter_add_batch(model, mem, batch):
    training = model.training
    dummy_domain = 0
    with torch.no_grad():
        model.eval()
        for x in batch:
            logit = model(x.unsqueeze(0))
            pseudo_cls = logit.max(1, keepdim=False)[1][0].cpu().numpy()
            pseudo_conf = (
                F.softmax(logit, dim=1).max(1, keepdim=False)[0][0].cpu().numpy()
            )
            mem.add_instance([x, pseudo_cls, dummy_domain, pseudo_conf])
    if training:
        model.train()
    return mem
