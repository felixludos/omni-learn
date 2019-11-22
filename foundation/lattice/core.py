
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import util
from ..models import networks

class Lattice(nn.Module):
	def __init__(self, graph, dim=1, nonlin='sigmoid', use_bias=True):
		pass

class Node(nn.Module):
	def __init__(self, neighbors, dim=1, nonlin='sigmoid', use_bias=True):
		self.neighbors = neighbors
		
		self.nonlin = nonlin
		self.dim = dim
		self.use_bias = use_bias
		
		self.value = torch.zeros(dim)
		
	def reset(self):
		self.net = networks.BasicLinear(len(self.neighbors)*self.dim, self.dim, nonlinearity=self.nonlin, use_bias=self.use_bias)
		
	def forward(self):
		
		if len(self.neighbors):
			self.value = self.net(torch.cat([n.value for n in self.neighbors], -1))
		
		return self.value
		

class Expansion(nn.Module):
	
	def __init__(self, din, dout, terms=3, hidden_dim=1, merge='sum', coupling=None, learn_coupling=False, deftype='torch.FloatTensor'):
		super(Expansion, self).__init__()
		
		self.coupling = nn.Parameter(torch.FloatTensor([coupling])) if coupling is not None and learn_coupling else torch.FloatTensor([coupling]).type(deftype) if coupling is not None else None
		self.merge_type = merge
		self.terms = terms
		
		self.nets = nn.ModuleList([ networks.MLP(din, dout, hidden_dims=[hidden_dim]*i) for i in range(terms) ])
		
	def forward(self, x, combined=True):
		
		outputs = torch.stack([net(x) for net in self.nets])
		
		if self.coupling is not None:
			c = self.coupling.pow(torch.arange(0,self.terms).type_as(outputs)).unsqueeze(-1).unsqueeze(-1)
			outputs *= c
			
		if not combined:
			return outputs
		
		if self.merge_type == 'sum':
			return outputs.sum(0)
		elif self.merge_type == 'mean':
			return outputs.mean(0)
		elif self.merge_type == 'max':
			return outputs.max(0)[0]
		else:
			raise Exception('bad name: {}'.format(self.merge_type))
