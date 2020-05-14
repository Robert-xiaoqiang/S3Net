import torch.nn as nn
from torch.nn import functional as F

class KLL(nn.Module):
	def __init__(self, reduction):
		super().__init__()
		print("You are using `KLL`!")
		self.reduction = reduction
	def forward(self, input_logits, target_logits):
		assert input_logits.size() == target_logits.size()
		# print(input_logits.shape)
		input_log_softmax = F.log_softmax(input_logits, dim=1)
		target_log_softmax = F.log_softmax(target_logits, dim=1)
		return F.kl_div(input_log_softmax, target_log_softmax, reduction = self.reduction)
