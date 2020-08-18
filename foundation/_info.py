

name = 'foundation'
long_name = 'foundation'

version = '0.2'
url = 'https://github.com/felixludos/foundation'

description = 'Powerful machine learning utilities for python'

author = 'Felix Leeb'
author_email = 'felixludos.info@gmail.com'

license = 'MIT'

readme = 'README.md'

packages = ['foundation']

import os
try:
	with open(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'requirements.txt'), 'r') as f:
		install_requires = f.readlines()
except:
	install_requires = ['pyyaml', 'c3linearize', 'numpy', 'matplotlib', 'torch', 'tensorflow',
	                    'gym', 'tabulate', 'ipdb', 'h5py', 'pyyaml', 'tqdm', 'pandas', 'humpack']
del os