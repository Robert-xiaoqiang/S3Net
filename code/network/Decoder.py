import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

class Decoder(nn.Module):
	def __init__(self, in_planes, size, num_classes):
		super().__init__()
		self.in_planes = in_planes
		self.size = size
		self.num_classes = num_classes
		self.linear = nn.Sequential(
				nn.Linear(
					self.size * self.size * self.in_planes,
					self.size * self.size
				),
				# nn.BatchNorm1d(self.size * self.size),
				nn.Linear(
					self.size * self.size,
					self.num_classes
				)				
			)
	def forward(self, x):
		return self.linear(x.view(x.shape[0], -1))
