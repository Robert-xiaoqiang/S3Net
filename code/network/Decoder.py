import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

class Decoder(nn.Module):
	def __init__(self, in_planes):
		super().__init__()
		self.in_planes = in_planes

	def forward(self, x):
		pass
