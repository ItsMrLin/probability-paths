import torch
from torch import nn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[...,i]
            losses.append(torch.max(
                       (q-1) * errors, 
                          q * errors
                      ).unsqueeze(1))
        loss = torch.mean(torch.cat(losses, dim=1))
        return loss


class CoverageLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        i = 0
        j = len(self.quantiles) - 1
        loss = []
        while i < j:
            left, right = self.quantiles[i], self.quantiles[j]
            #print(preds[..., i].shape, target.shape)
            print(f"preds gradient: {preds.requires_grad}")
            #print( preds[..., i] < target)
            left_cover = torch.le(preds[..., i], target)
            print(f"preds gradient: {preds.requires_grad}")
            covered = torch.logical_and(torch.le(preds[..., i], target), 
                                        torch.le(target, preds[..., j]))

            loss.append((torch.mean(covered.float()) - (right - left)) ** 2)

            i += 1
            j -= 1
        return torch.stack(loss).mean()


