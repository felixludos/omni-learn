
import sys, os
import numpy as np
import torch

from .. import util
from .config import get_config
from .loading import find_checkpoint


class Run(util.tdict):
	pass

class Run_Manager(object):

	def __init__(self, path, recursive=False):

		self.master_path = path
		self.recursive = False

		self.refresh()

	def refresh(self, load_last=True): # collects info from all runs

		self.full_info = []

		for name in os.listdir(self.master_path):

			run_path = os.path.join(self.master_path, name)
			if os.path.isdir(run_path) and 'config.yml' in os.listdir(run_path):

				try:
					ckpt_path = find_checkpoint(run_path, load_last=load_last)
					run = Run(name=name, path=run_path)
					run.ckpt_path = ckpt_path
					run.config = get_config(run_path)
				except FileNotFoundError:
					pass
				else:
					self.full_info.append(run)


		print('Found {} runs'.format(len(self.full_info)))

		self.active = None
		self.clear_filters()

	def clear_filters(self):
		self.active = self.full_info.copy()

	def __getitem__(self, item):
		if self.active is None:
			print('From full')
			return self.full_info[item]
		return self.active[item]




