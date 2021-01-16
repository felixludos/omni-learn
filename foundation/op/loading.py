from pathlib import Path
import torch

import omnifig as fig

from .. import util
from foundation.op.runs import Run
# from .data import get_loaders

# @fig.AutoModifier('torch')
@fig.Component('run')
class TorchRun(Run):
	
	def _save_results(self, data, path=None, name=None, ext='pth.tar'):
		if path is None:
			path = self.get_path()
		
		path = Path(path)
		if path.is_dir():
			assert name is not None, 'name is missing'
			path = path / f'{name}.{ext}'
		
		return torch.save(data, str(path))
	
	def _load_results(self, path=None, name=None, ext='pth.tar'):
		if path is None:
			path = self.get_path()
		
		path = Path(path)
		if path.is_dir():
			assert name is not None, 'name is missing'
			path = path / f'{name}.{ext}'
			
		return torch.load(str(path))


def respect_config(A):
	device = A.push('device', 'cpu', overwrite=not torch.cuda.is_available())
	
	cudnn_det = A.pull('cudnn_det', False)
	if 'cuda' in device:
		torch.backends.cudnn.deterministic = cudnn_det
	
	A.push('seed', util.gen_random_seed(), overwrite=False, silent=True)
	
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
