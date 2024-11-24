from .imports import *
from ..abstract import AbstractModel
from ..machines import Machine



class Model(Machine, nn.Module, AbstractModel):
	pass


