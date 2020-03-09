from . import util
from .train import Component, AutoComponent, Modifier, AutoModifier
from . import framework
from . import data
# from . import sim

from . import models
from .framework import * # TODO: cleanup

try:
	from . import legacy
except:
	print('WARNING: failed to import legacy models')

