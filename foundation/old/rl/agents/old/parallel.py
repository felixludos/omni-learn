import sys, os
import torch
import torch.distributions as distrib
from torch.autograd import grad
from foundation.foundation import framework as fm
from foundation.foundation.models import networks as nets
from foundation.foundation import util
from .optim import Conjugate_Gradient, flatten_params
from .npg import NPG
import copy
import torch.multiprocessing as mp
from torch.utils.data.dataloader import ExceptionWrapper
import inspect

def _agent_training_backend(in_queue, out_queue):
	
	agent = None
	while True:
		
		paths = in_queue.get()
		if paths is None:
			break
		try:
			if isinstance(paths, fm.Agent):
				agent = paths # set agent
			else:
				out_queue.put(agent.train_step(paths))
		except Exception:
			out_queue.put(ExceptionWrapper(sys.exc_info()))

class Parallel_Agent(fm.Agent): # wrapper class for agents, executing train_step in a different process
	def __init__(self, agent, block=True):
		super(Parallel_Agent, self).__init__(agent.policy, agent.baseline, agent.discount)
		self.block = block # flag to block when training to guarantee training is complete before returning from train_step method
		# remove remote baseline
		agent.baseline = None
		
		self.inq = mp.Queue()
		self.outq = mp.Queue()
		
		self.backend = mp.Process(target=_agent_training_backend, args=(self.inq, self.outq))
		self.backend.daemon = True
		self.backend.start()
		
		self.inq.put(agent) # send agent to initialize backend
		
	def _dispatch_train_step(self, paths):
		
		if 'returns' not in paths:
			paths.returns = self.compute_returns(paths.rewards)
		if 'advantages' not in paths:
			paths.advantages = self.compute_advantages(paths.returns, paths.observations)
		
		collated_paths = util.NS()
		if isinstance(paths.returns, list):
			collated_paths.returns = torch.cat(paths.returns)
		if isinstance(paths.advantages, list):
			collated_paths.advantages = torch.cat(paths.advantages)
		if isinstance(paths.observations, list):
			collated_paths.observations = torch.cat(paths.observations)
		if isinstance(paths.actions, list):
			collated_paths.actions = torch.cat(paths.actions)
		
		# dispatch policy update
		self.last_stats = None
		self.baseline_stats = None
		self.inq.put(collated_paths)
		
		# train baseline while policy is learning
		if self.baseline is not None:
			self.baseline_stats = self.baseline.train_step(paths)
			
		if self.block: # wait until policy update is complete
			return self._get_stats()
		
		return self.baseline_stats
	
	def _get_stats(self):
		if self.last_stats is None:
			update_stats = self.outq.get(timeout=10)
			if isinstance(update_stats, ExceptionWrapper):
				raise update_stats.exc_type(update_stats.exc_msg)
			
			if self.baseline_stats is not None:
				update_stats.join(self.baseline_stats, prefix='bsln-')
			self.last_stats = update_stats
		return self.last_stats
	
	def __del__(self):
		self.inq.put(None) # shutdown backend
	
	def __getattribute__(self, item):
		if item == 'train_step': # only dispatch
			return self._dispatch_train_step
		return super(Parallel_Agent, self).__getattribute__(item)
		