

name = 'omnilearn'
long_name = 'omni-learn'

version = '0.5.12'
url = 'https://github.com/felixludos/omni-learn'

description = 'Powerful machine learning utilities for python'

author = 'Felix Leeb'
author_email = 'felixludos.info@gmail.com'

license = 'MIT'

readme = 'README.md'

installable_packages = ['omnilearn']

import os
try:
	with open(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'requirements.txt'), 'r') as f:
		install_requires = f.readlines()
except:
	# install_requires = ['humpack', 'omnibelt', 'omnifig', 'pyyaml', 'c3linearize', 'numpy', 'matplotlib',
	#                     'torch', 'tensorflow',
	#                     'gym', 'tabulate', 'ipdb', 'h5py', 'pyyaml', 'tqdm', 'pandas']
	install_requires = ['humpack', 'omnibelt',
'omnifig',
'numpy',
'matplotlib',
'torch',
'torchvision',
'tensorflow',
'gym',
'wget',
'opencv-python',
'tabulate',
	                    'ipython', 'networkx',
'ipdb',
'h5py',
'pyyaml',
'tqdm',
'pandas',
'scikit-learn',
'seaborn',
'moviepy']
del os