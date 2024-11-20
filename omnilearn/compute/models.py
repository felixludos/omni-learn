from .imports import *
from ..abstract import AbstractModel
from ..core import ToolKit
from ..mixins import Prepared
from ..machines import Machine


class Model(Prepared, Machine, nn.Module, AbstractModel):
	pass


