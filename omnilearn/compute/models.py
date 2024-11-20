from .imports import *
from ..abstract import AbstractModel
from ..core import ToolKit
from ..mixins import Prepared



class Model(Prepared, ToolKit, nn.Module, AbstractModel):
	pass


