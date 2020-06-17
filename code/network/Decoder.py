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

class DownsamplingConv(nn.Module):
	def __init__(self, in_planes, planes):
		super().__init__()
		self.conv = nn.Conv2d(in_planes, planes, 3, stride = 2, padding = 1)
		self.bn = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace = True)
	def forward(self, x):
		return self.relu(self.bn(self.conv(x)))


class DenseDecoder(nn.Module):
	# target size
	def __init__(self, in_planes, size, num_classes):
		super().__init__()
		self.in_planes = in_planes
		self.num_classes = num_classes
		self.size = size

		self.conv2 = nn.Sequential(
			# DownsamplingConv(self.in_planes, 1),
			nn.MaxPool2d(3, stride = 16, padding = 1),
		)

		# self.conv4 = nn.Sequential(
		# 	DownsamplingConv(self.in_planes, 1),
		# 	DownsamplingConv(1, 1),
		# 	nn.MaxPool2d(3, stride = 2, padding = 1),
		# )

		# self.conv8 = nn.Sequential(
		# 	DownsamplingConv(self.in_planes, 1),
		# 	nn.MaxPool2d(3, stride = 2, padding = 1),
		# )

		# self.conv16 = DownsamplingConv(self.in_planes, 1)

		self.conv32 = nn.Conv2d(self.in_planes, 1, 1)

		self.linear = nn.Sequential(
				nn.Linear(
					self.size * self.size * 1,
					self.size * self.size
				),
				# nn.BatchNorm1d(self.size * self.size),
				nn.Linear(
					self.size * self.size,
					self.num_classes
				)				
			)
	# 128, 64, 32, 16, 8
	def forward(self, x2, x4, x8, x16, x32):
		x2 = self.conv2(x2)
		x2 = torch.max(x2, dim = 1, keepdim = True)[0]
		# x4 = self.conv4(x4)
		# x8 = self.conv8(x8)
		# x16 = self.conv16(x16)
		x32 = self.conv32(x32)
		# x = torch.cat([ x2, x4, x8, x16, x32 ], dim = 1) # channel concatenation/dense
		x = x2 + x32
		return self.linear(x.view(x.shape[0], -1))