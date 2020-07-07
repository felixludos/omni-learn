import torch
import torch.distributions as distrib
from torch.autograd import grad
from foundation.foundation import framework as fm
from foundation.foundation.models import networks as nets
from foundation.foundation import util
from .parallel import Parallel_Agent
from .optim import Conjugate_Gradient, flatten_params
from .npg import NPG
import copy

class Multi_NPG(fm.Manager):
	def __init__(self, sub_policy, num_sub, cmd_policy, parallel=True, cmd_baseline=None, sub_baseline=None, discount=0.99, unique_subs=True,
	             step_size=0.01, max_cg_iter=10, reg_coeff=1e-5, separate_stats=True):

		if unique_subs:
			sub_agents = [NPG(sub_policy, baseline=sub_baseline, discount=discount, step_size=step_size, max_cg_iter=max_cg_iter, reg_coeff=reg_coeff)
			              for _ in range(num_sub)]
		else:
			agent = NPG(sub_policy, baseline=sub_baseline, discount=discount, step_size=step_size,
			            max_cg_iter=max_cg_iter, reg_coeff=reg_coeff)
			sub_agents = [agent]*num_sub

		cmd_agent = NPG(cmd_policy, baseline=cmd_baseline, discount=discount, step_size=step_size, max_cg_iter=max_cg_iter, reg_coeff=reg_coeff)

		blocking = [False] + [not unique_subs]*num_sub
		parallels = [parallel]*2 + [parallel and unique_subs]*(num_sub-1)



		super(Multi_NPG, self).__init__(agents=[cmd_agent]+sub_agents, blocking=blocking, parallel=parallels, separate_stats=separate_stats)

# TODO: use sequential update module to parallelize hierarchical managers

class Global_Baseline_NPG(Multi_NPG): # requires all agents to receive the same reward
	def __init__(self, global_baseline, discount, **kwargs): # same args as Multi_NPG except baselines

		super(Global_Baseline_NPG, self).__init__(**kwargs)

		self.baseline = global_baseline
		self.discount = discount
		
	def train_step(self, paths): # list of not fully collated trajs (1 per agent)
	
		global_obs = [ torch.cat([ path.observations[i] for path in paths],1) for i in range(len(paths[0].observations)) ]
		
		#assert 'advantages' not in paths[0][0], 'Global baseline was not used to compute advantages'

		#print(paths)
		#print(paths[0].rewards)

		#print(paths[0].env_infos[0])
		#quit()

		returns = self.compute_returns(paths[0].rewards)
		advantages = self.compute_advantages(returns, global_obs)
		
		for agent_paths in paths:
			agent_paths.returns = returns
			agent_paths.advantages = advantages
		
		#results = [agent.train_step(p) for agent, p in zip(self.agents, paths)]  # dispatch/train

		results = []
		for agent, p in zip(self.agents, paths):
			#print(agent)
			results.append(agent.train_step(p))

		
		# train baseline
		baseline_stats = None
		if self.baseline is not None:
			baseline_stats = self.baseline.train_step(util.NS(observations=global_obs, returns=returns))
			
		joined_stats = self._collect_stats(results)
		
		if baseline_stats is not None:
			joined_stats.join(baseline_stats, prefix='bsln-')

		return joined_stats