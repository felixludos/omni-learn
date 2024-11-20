from .imports import *
from .core import ToolKit, Context, tool, gear, space, indicator
from .spaces import Scalar



def test_indicator():
	class Worker(ToolKit):
		@indicator('y')
		def compute(self, x):
			return x + 1

	ctx = Context(Worker())
	ctx['x'] = 10

	assert ctx['y'] == 11

	mech = ctx.mechanics()

	assert isinstance(mech['y'], Scalar)



def test_spaces():
	class Worker(ToolKit):
		@tool('y')
		def compute(self, x):
			return x + 1
		@compute.space
		def y_space(self, x):
			return f'{x} and 1'

	class Loader(ToolKit):
		@tool('x')
		def load(self):
			return 10
		@load.space
		def x_space(self):
			return 'x-space'

	w = Worker()
	l = Loader()

	ctx = Context(w, l)

	assert ctx['y'] == 11
	assert w.y_space == 'x-space and 1'

	ctx = Context()

	w = Worker()
	l = Loader()

	ctx.include(w).include(l)

	assert ctx['y'] == 11
	assert w.y_space == 'x-space and 1'





