from .imports import *
from ..abstract import AbstractModel
from ..core import Machine



class Model(Machine, AbstractModel, nn.Module):
	pass


