
import sys, os
import numpy as np
import torch

from .. import util
from .config import get_config


class Run(util.tdict):
	pass

class Run_Manager(object):

	def __init__(self, path, recursive=False):

		self.master_path = path
		self.recursive = False

		self.refresh()

	def refresh(self): # collects info from all runs

		self.full_info = []

		for name in os.listdir(self.master_path):

			run_path = os.path.join(self.master_path, name)
			if os.path.isdir(run_path) and 'config.yml' in os.listdir(run_path):
				run = Run(name=name, path=run_path)
				run.config = get_config(run_path)

				self.full_info.append(run)


		print('Found {} runs'.format(len(self.full_info)))

		self.active = []

	


