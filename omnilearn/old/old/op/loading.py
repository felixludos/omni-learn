import os
import pickle
from pathlib import Path
import numpy as np
import torch

from omnibelt import load_txt, save_txt, HierarchyPersistent, unspecified_argument
import omnifig as fig

from .. import util
from .runs import Run
# from .data impor

_config_root = Path(__file__).parents[0] / 'configs'

class CompatibilityUnpickler(pickle.Unpickler):
	def find_class(self, module, name):
		renamed_module = module.replace('foundation', 'omnilearn')
		return super().find_class(renamed_module, name)

CompatibilityUnpickler.Unpickler = CompatibilityUnpickler

# #@fig.AutoModifier('torch')
#@fig.Component('run')
class Torch_Run(Run):

	def _save_datafile(self, data, path, _save_fn=None, **kwargs):
		return super()._save_datafile(data, path, _save_fn=torch.save, **kwargs)


	def _load_datafile(self, path=None, name=None, ext='pth.tar', device=None, _load_fn=None, **kwargs):
		def _load_fn(p):
			special = {'map_location':device} if device is not None else {}
			try:
				return torch.load(p, **special)
			except ModuleNotFoundError:
				special['pickle_module'] = CompatibilityUnpickler
				return torch.load(p, **special)

		return super()._load_datafile(path, name=name, _load_fn=_load_fn, ext=ext, **kwargs)



def respect_config(A):
	device = A.push('device', 'cuda' if torch.cuda.is_available() else 'cpu',
					overwrite=not torch.cuda.is_available())
	
	cudnn_det = A.pull('cudnn_det', False)
	if 'cuda' in device:
		torch.backends.cudnn.deterministic = cudnn_det

	if A.pull('cuda-blocking', True):
		os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
	
	A.push('seed', util.gen_random_seed(), overwrite=False, silent=False)
	
#
# #@fig.Script('load_config')
# def load_config(A):
#
#
# 	# override = A.override if 'override' in A else {}
# 	override = A.pull('override', {})
#
# 	extend = A.pull('extend', None)
#
# 	raw_path, loading = get_raw_path(A)
# 	if raw_path is None:
# 		return A
#
# 	print(f'Loading Config: {raw_path}')
#
# 	path = find_config(raw_path)
#
# 	load_A = fig.get_config(path)
# 	if loading:
# 		load_A.update(A)
# 	else:
# 		A = load_A
# 	A.update(override)
#
# 	A.push('load' if loading else 'resume', raw_path, overwrite=True)
#
# 	if extend is not None:
# 		A.push('training.step_limit', extend)
#
# 	return A

#
# #@fig.Script('load_records')
# def load_records(A):
#
# 	last = A.pull('last', False)
#
# 	path = A.pull('path', None)
# 	loading = False
#
# 	if path is None:
# 		path, loading = get_raw_path(A)
# 	if path is None or loading:
# 		return setup_records(A)
#
# 	print(f'Loading Records: {path}')
#
# 	ckpt_path = find_checkpoint(path, last=last)
# 	A.push('model._load_params', ckpt_path, overwrite=True)
#
# 	path = find_records(path, last=last)
#
# 	with open(path, 'r') as f:
# 		records = yaml.safe_load(f)
# 	return records
#

# fig.register_config_dir(str(_config_root)) # WARNING: overrides debug
# fig.register_config('origin', str(_config_root/'origin.yaml'))

