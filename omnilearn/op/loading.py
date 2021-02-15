import pickle
from pathlib import Path
import torch

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

# @fig.AutoModifier('torch')
@fig.Component('run')
class Torch_Run(Run):
	
	def has_results(self, ident, path=None, ext=None):
		if ext is None:
			ext = 'pth.tar'
		return super().has_results(ident, path=path, ext=ext)
	
	def _save_results(self, data, path=None, name=None, ext='pth.tar'):
		if path is None:
			path = self.get_path()
		
		path = Path(path)
		if path.is_dir():
			assert name is not None, 'name is missing'
			path = path / f'{name}.{ext}'
		
		return torch.save(data, str(path))
	
	def _load_results(self, path=None, name=None, ext='pth.tar', device=None, **kwargs):
		if path is None:
			path = self.get_path()
		
		path = Path(path)
		if path.is_dir():
			assert name is not None, 'name is missing'
			path = path / f'{name}.{ext}'
			
		
		special = {'map_location':device} if device is not None else {}
		try:
			return torch.load(str(path), **special)
		except ModuleNotFoundError:
			special['pickle_module'] = CompatibilityUnpickler
			return torch.load(str(path), **special)

def respect_config(A):
	device = A.push('device', 'cuda' if torch.cuda.is_available() else 'cpu',
					overwrite=not torch.cuda.is_available())
	
	cudnn_det = A.pull('cudnn_det', False)
	if 'cuda' in device:
		torch.backends.cudnn.deterministic = cudnn_det
	
	A.push('seed', util.gen_random_seed(), overwrite=False, silent=False)
	
#
# @fig.Script('load_config')
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
# @fig.Script('load_records')
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
fig.register_config('origin', str(_config_root/'origin.yaml'))

