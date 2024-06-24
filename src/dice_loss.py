import torch


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(y_pred, y_true):
        y_pred = torch.flatten(y_pred)
        y_true = torch.flatten(y_true)

        counter = (y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum() + 1e-8

        dice = (2 * counter) / denominator

        return 1 - dice
