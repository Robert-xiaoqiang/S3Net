from torch import nn


class CEL(nn.Module):
    def __init__(self, reduction):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.reduction = reduction
        self.eps = 1e-6
    
    def forward(self, pred, target):
        batch_size = pred.shape[0]
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        total = numerator / (denominator + self.eps)
        if self.reduction == 'mean':
        	total /= batch_size
        return total
