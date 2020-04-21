
import sys, os, shutil
import numpy as np

from .registry import AutoScript

@AutoScript('sanitize')
def sanitize_runs(path=None, keep_ckpts='last', purge_tests=False,
                  skip_check=False):
	
	if path is None:
		assert 'FOUNDATION_SAVE_DIR' in os.environ
		path = os.environ['FOUNDATION_SAVE_DIR']
	
	# if isinstance(keep_ckpts, (list, tuple)):
	if keep_ckpts != 'last':
		raise NotImplementedError
	
	todo = []
	if keep_ckpts is not None:
		todo.append(f'keeping ckpts: {keep_ckpts}')
	if purge_tests:
		todo.append('purge tests')
	
	assert len(todo)

	if not skip_check:
		r = input('Confirm sanitization: {} - [y]/n? '.format(', '.join(todo)))
		if r in {'n', 'N', 'no', 'No'}:
			print('Quitting, nothing was changed.')
			return 1
	
	run_names = os.listdir(path)
	
	changes = 0
	
	for run_name in run_names:
		run_path = os.path.join(path, run_name)
		
		if os.path.isdir(run_path):
			change = False
			
			if purge_tests and run_name.startswith('test-'):
				print(f'Purging: {run_name}')
				shutil.rmtree(run_path)
				change = True
			else:
			
				ckpts = [name for name in os.listdir(run_path) if 'checkpoint' in name and '.pth.tar' in name]
				nums = np.array([int(name.split('.')[0].split('_')[1]) for name in ckpts])
				last = max(nums)
				
				if keep_ckpts == 'last':
					change = True
					for cname, num in zip(ckpts, nums):
						if num != last:
							cpath = os.path.join(run_path, cname)
							os.remove(cpath)
						else:
							print(f'Keeping {cname} in {run_name}')
	
			changes += int(change)
			
	print('Sanitation complete {} runs affected.')
	
	return 0