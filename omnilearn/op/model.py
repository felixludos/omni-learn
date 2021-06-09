from pathlib import Path
from omnibelt import get_printer
import omnifig as fig

import torch
from torch import nn

from .. import util
from . import runs
from . import framework as fm

prt = get_printer(__file__)

def load_checkpoint(path):  # model parameters - TODO: maybe add more options
	try:
		return torch.load(path)
	except RuntimeError:
		return torch.load(path, map_location = "cpu")


@fig.Script('load-model', description='Creates/loads a model')
@fig.Component('model')
def load_model(A, silent=None):
	'''
	Creates the model and possibly loads existing model parameters
	'''
	
	if silent is None:
		silent = A.pull('silent', False, silent=True)
	
	model_config = A
	
	legacy = A.pull('legacy', False) # TODO: remove
	
	ckpt = A.pull('_load-ckpt', '<>load-model', '<>load', None)
	if ckpt is not None:
		try:
			path = runs.find_path(ckpt, A, silent=silent, allow_file=False)
		except runs.RunNotFoundError:
			prt.warning(f'Failed to find config from: {str(ckpt)}')
		else:
			model_config = fig.get_config(str(path))
			if legacy: # TODO: remove
				addr = A.pull('load-model', None)
				if addr is not None:
					model_config = fig.get_config(str(Path(ckpt).parents[1]/addr))
			src_config = model_config.pull('_loaded_model', None, silent=True, raw=True)
			if src_config is not None:
				model_config = src_config
			model_config.update(A.pull('model-override', {}, raw=True, silent=True))
			A.push('_loaded_model', model_config, silent=True, process=False)
	
	raw_type = model_config.pull('_type', None, silent=True)
	if raw_type is None:
		assert model_config.contains_nodefault('model'), 'No model found'
		model_config = model_config.sub('model')
		raw_type = model_config.pull('_type', None, silent=True)
	if raw_type == 'model':
		model_type = model_config.pull('_model_type', None, silent=True)
		if model_type is None:
			assert model_config.contains_nodefault('model')
			model_config = model_config.sub('model')
		else:
			model_config.push('_type', model_type, silent=True)
			model_config.push('_mod', model_config.pull('_model_mod', []), silent=True)
	
	util.Seed(model_config)
	
	# model = fig.create_component(A)
	model = model_config.pull_self()
	
	# TODO: check if model needs the dataset (eg. a batch) to initialize params before creating the optim
	
	optim = model.set_optim(model_config) if isinstance(model, fm.Optimizable) else None
	
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
	
	if ckpt is not None:
		ckpt = runs.find_ckpt_path(ckpt, A, silent=silent)
		model.load_checkpoint(ckpt)
		if print_model:
			print(f'Loaded parameters from {ckpt}')
	
	# origin_name = A.pull('__origin_key', None, silent=True)
	
	return model



# fig.AutoComponent('criterion')(util.get_loss_type)
# fig.AutoComponent('nonlin')(util.get_nonlinearity)
# fig.AutoComponent('normalization')(util.get_normalization)
# fig.AutoComponent('pooling')(util.get_pooling)
# fig.AutoComponent('upsampling')(util.get_upsample)

# @Component('stage') # TODO
class Stage_Model(fm.Model):
	def __init__(self, A):
		raise NotImplementedError
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




