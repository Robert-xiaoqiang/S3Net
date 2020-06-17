import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Squeezer:
	def __init__(self):
		pass
	def __call__(self, *args):
		ret = [ ]
		for f in args:
			avg = torch.mean(f, dim = 1)
			ret.append(torch.sigmoid(avg))

		return tuple(ret)
