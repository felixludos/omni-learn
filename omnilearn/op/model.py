
import omnifig as fig

import torch
from torch import nn

from .. import util
from . import framework as fm


def load_checkpoint(path):  # model parameters - TODO: maybe add more options
	try:
		return torch.load(path)
	except RuntimeError:
		return torch.load(path, map_location = "cpu")


@fig.Script('load-model', description='Creates/loads a model')
@fig.Component('model')
def load_model(A):
	'''
	Creates the model and possibly loads existing model parameters
	'''
	
	raw_type = A.pull('_type', silent=True)
	if raw_type == 'model':
		model_type = A.pull('_model_type', None, silent=True)
		if model_type is None:
			assert A.contains_nodefault('model')
			A = A.sub('model')
		else:
			A.push('_type', model_type, silent=True)
			A.push('_mod', A.pull('_model_mod', []), silent=True)
	
	util.Seed(A)
	
	# model = fig.create_component(A)
	model = A.pull_self()
	
	optim = model.set_optim(A) if isinstance(model, fm.Optimizable) else None
	
	try:
		device = model.get_device()
	except AttributeError:
		device = A.pull('device')
	model.to(device)
	
	print_model = A.pull('_print_model', False)
	if print_model:
		print(model)
		if A.pull('_print_optim', False):
			print(optim)
		print(f'Number of model parameters: {util.count_parameters(model)}')
	
	path = A.pull('_load-ckpt', None)
	if path is not None:
		model.load_checkpoint(path)
		if print_model:
			print(f'Loaded parameters from {path}')
	
	# origin_name = A.pull('__origin_key', None, silent=True)
	
	return model



fig.AutoComponent('criterion')(util.get_loss_type)
fig.AutoComponent('nonlin')(util.get_nonlinearity)
fig.AutoComponent('normalization')(util.get_normalization)
fig.AutoComponent('pooling')(util.get_pooling)
fig.AutoComponent('upsampling')(util.get_upsample)

@fig.AutoComponent('viz-criterion')
class Viz_Criterion(nn.Module):
	def __init__(self, criterion, arg_names=[], kwarg_names=[],
	             allow_grads=False):
		super().__init__()
		
		self.criterion = util.get_loss_type(criterion)
		self.arg_names = arg_names
		self.kwarg_names = kwarg_names
		self.allow_grads = allow_grads

	def forward(self, out):
		
		args = [out[key] for key in self.arg_names]
		kwargs = {key:out[key] for key in self.kwarg_names}
		
		if self.allow_grads:
			return self.criterion(*args, **kwargs)
		
		with torch.no_grad():
			return self.criterion(*args, **kwargs)

# @Component('stage') # TODO
class Stage_Model(fm.Model):
	def __init__(self, A):
		stages = A.pull('stages')

		din = A.pull('din')
		dout = A.pull('dout')

		criterion_info = A.pull('criterion', None)

		super().__init__(din, dout)

		self.stages = nn.ModuleList(self._process_stages(stages))

		self.criterion = None
		if criterion_info is not None:
			self.criterion = util.get_loss_type(criterion_info) \
				if isinstance(criterion_info, str) else util.get_loss_type(**criterion_info)  # TODO: automate



	def _create_stage(self, stage):
		return fig.create_component(stage)
	def _process_stages(self, stages):

		N = len(stages)

		assert N > 0, 'no stages provided'

		din = self.din

		sub = []
		for i, stage in enumerate(stages):

			stage.din = din
			stage = self._create_stage(stage)

			sub.append(stage)
			din = stage.dout

		assert din == self.dout, 'dins and douts not set correctly: {} vs {}'.format(din, self.dout)

		return sub

	def forward(self, x):
		q = x
		for stage in self.stages:
			q = stage(q)
		return q


	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		# compute loss

		x = batch[0]
		if len(batch) == 2:
			y = batch[1]
		else:
			assert len(batch) == 1, 'Cant figure out how the batch is setup (you will have to implement your own _step)'
			y = x

		out.x, out.y = x, y

		pred = self(x)
		out.pred = pred

		loss = self.criterion(pred, y)
		out.loss = loss

		if isinstance(self, fm.Regularizable):
			reg = self.regularize(out)
			loss += reg

		out.loss = loss

		if self.train_me():
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		return out




