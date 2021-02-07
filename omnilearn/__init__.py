from . import util
from . import op
from . import data
from . import eval
# from . import sim

from . import models
# from .framework import * # TODO: cleanup
from .op.framework import FunctionBase, Model, Encodable, Decodable, Generative, Recordable, \
	Evaluatable, Visualizable, Function, Optimizable

from omnibelt import get_printer

prt = get_printer(__name__)

try:
	from . import legacy
except:
	prt.info('Failed to important legacy models')

import os
__info__ = {'__file__':os.path.join(os.path.abspath(os.path.dirname(__file__)), '_info.py')}
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), '_info.py'), 'r') as f:
	exec(f.read(), __info__)
del os
del __info__['__file__']
__author__ = __info__['author']
__version__ = __info__['version']
