
import omnifig as fig

import torch
from torch import nn

from .. import util
from .clock import CustomAlert
from .. import framework as fm


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
			assert A.contains_no_default('model')
			A = A.sub('model')
		else:
			A.push('_type', model_type, silent=True)
			A.push('_mod', A.pull('_model_mod', []), silent=True)
	
	seed = A.pull('seed')
	util.set_seed(seed)
	
	# model = fig.create_component(A)
	model = A.pull_self()
	
	optim = model.set_optim(A) if isinstance(model, fm.Optimizable) else None
	
	device = A.pull('device', 'cpu')
	if not torch.cuda.is_available():
		device = 'cpu'
	model.to(device)
	
	path = A.pull('_load_params', None)
	if path is not None:
		ckpt = load_checkpoint(path)
		
		params = ckpt['model_state']
		
		if 'optim' in params:
			load_optim = A.pull('load_optim_params', True)
			if not load_optim:
				del params['optim']
	
			elif 'scheduler' in params['optim']:
				load_scheduler = A.pull('load_scheduler_params', True)
				if not load_scheduler:
					del params['optim']['scheduler']
				
		strict_load_state = A.pull('strict_load_state', True)
		
		model.load_state_dict(params, strict=strict_load_state)
		
		print(f'Loaded parameters from {path}')
	
	origin_name = A.pull('__origin_key', None, silent=True)
	
	records = A.pull('records', None, ref=True)
	if records is not None:
		if isinstance(model, fm.Recordable):
			stats_fmt = A.pull('stats-fmt', None)
			records.register_stats_client(model.get_stats(), fmt=stats_fmt)

	clock = A.pull('clock', None, ref=True)
	if clock is not None:
		if isinstance(model, fm.Maintained):
			clock.register_alert(origin_name, CustomAlert(model.maintenance))
	
		if isinstance(optim, util.Schedulable) and optim.scheduler is not None:
			name = f'lr-{origin_name}' if origin_name is not None and origin_name != 'model' else 'lr'
			clock.register_alert(name, optim.scheduler)
	
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
		
		self.criterion = criterion
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
class Stage_Model(fm.Schedulable, fm.Trainable_Model):
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




