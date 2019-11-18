import sys, os, time
import numpy as np
import torch
import torch.multiprocessing as mp
from .. import util

stack_collate = util.make_collate(stack=True)
cat_collate = util.make_collate(stack=False)

class _Generator_Iterator(object):
	def __init__(self, gen, step_threshold, track_stats_tau,
	             step_cutoff, episode_cutoff,
	             step_limit, episode_limit, ):
		self.gen = gen
		self.total_steps = 0
		self.total_episodes = 0

		self.step_threshold = step_threshold
		self.step_cutoff = step_cutoff
		self.episode_cutoff = episode_cutoff
		self.stats = util.StatsMeter('rewards', 'len', tau=track_stats_tau) if track_stats_tau is not None else None

		self.step_limit = step_limit
		self.episode_limit = episode_limit
		self.steps_left = step_limit
		self.episodes_left = episode_limit
		#assert not (self.step_limit is None and self.episode_limit is None), 'must specify limit for iterator'

	def steps_generated(self):
		return self.total_steps
	def episodes_generated(self):
		return self.total_episodes

	def __len__(self):
		l = self.episode_limit if self.episode_limit is not None else self.step_limit
		return 0 if l is None else l # can be infinite

	def __iter__(self):
		return self

	def __next__(self):
		if (self.episodes_left is not None and self.episodes_left <= 0) \
				or (self.steps_left is not None and self.steps_left <= 0):
			raise StopIteration

		step_cutoff = self.steps_left if self.step_cutoff is None else self.step_cutoff
		if self.step_cutoff is not None and self.steps_left is not None:
			step_cutoff = min(step_cutoff, self.step_cutoff)

		episode_cutoff = self.episodes_left if self.episode_cutoff is None else self.episode_cutoff
		if self.episode_cutoff is not None and self.episodes_left is not None:
			episode_cutoff = min(episode_cutoff, self.episode_cutoff)

		E, S, rollouts = self.gen._rollout(stats=self.stats,
		                                   step_threshold=self.step_threshold,
		                                   episode_cutoff=episode_cutoff,
		                                   step_cutoff=step_cutoff)
		self.total_episodes += E
		self.total_steps += S
		if self.steps_left is not None:
			self.steps_left -= S
		if self.episodes_left is not None:
			self.episodes_left -= E

		return rollouts

class Generator(object):
	def __init__(self, env, agent, track_stats_tau=0.1,
	             drop_last_state=False, max_episode_length=None,
	             episode_cutoff=None, step_cutoff=None, # hard cutoff per iteration
	             step_threshold=None, # soft cutoff per iteration
	             step_limit=None, episode_limit=None): # overall budget (optional)
		super().__init__()

		self.env = env
		self.agent = agent

		self.stats = util.StatsMeter('rewards', 'len', tau=track_stats_tau) if track_stats_tau is not None else None
		self.drop_last = drop_last_state
		self.step_threshold = step_threshold
		self.episode_cutoff = episode_cutoff
		self.step_cutoff = step_cutoff
		self.max_episode_length = max_episode_length

		self.total_steps = 0
		self.total_episodes = 0

		self.step_limit = step_limit
		self.episode_limit = episode_limit

		assert not (self.step_threshold is None
		            and self.step_cutoff is None
		            and self.episode_cutoff is None), 'must specify iteration size in some way'

	def steps_generated(self):
		return self.total_steps
	def episodes_generated(self):
		return self.total_episodes

	def _rollout(self, stats=None, step_threshold=None, episode_cutoff=None, step_cutoff=None):
		if stats is None:
			stats = self.stats
		if step_threshold is None:
			step_threshold = self.step_threshold
		if episode_cutoff is None:
			episode_cutoff = self.episode_cutoff
		if step_cutoff is None:
			step_cutoff = self.step_cutoff

		states = []
		actions = []
		rewards = []
		info = []

		steps_generated = 0
		episodes_generated = 0

		with torch.no_grad():

			while (episode_cutoff is None or episodes_generated < episode_cutoff) \
					and (step_cutoff is None or steps_generated < step_cutoff) \
					and (step_threshold is None or steps_generated < step_threshold):

				S = []
				A = []
				AI = []
				EI = []
				R = []

				S.append(self.env.reset().view(1, -1))
				episodes_generated += 1

				done = False
				steps = 0
				while not done and (self.max_episode_length is None or steps < self.max_episode_length)\
						and (step_cutoff is None or steps_generated < step_cutoff):
					a, ai = self.agent.gen_action(S[-1])
					A.append(a)
					AI.append(ai)

					s, r, done, *einfo = self.env.step(a)
					ei = einfo[0] if len(einfo) else {}
					
					steps += 1
					steps_generated += 1

					s = s.view(1, -1)
					EI.append(ei)
					S.append(s)
					R.append(r)

				if self.drop_last:
					S = S[:-1]

				S = torch.cat(S)
				A = torch.cat(A)
				R = torch.stack(R)
				AI = cat_collate(AI)
				EI = cat_collate(EI)

				if self.stats is not None:
					stats.update('len', A.size(0))
					stats.update('rewards', R.sum())

				states.append(S)
				actions.append(A)
				info.append(AI)
				info[-1].update(EI)  # combine all info
				rewards.append(R)

			info = stack_collate(info)
			rollout = {
				'states': stack_collate(states),
				'actions': stack_collate(actions),
				'rewards': stack_collate(rewards),
			}
			rollout.update(info)

		self.total_episodes += episodes_generated
		self.total_steps += steps_generated

		return episodes_generated, steps_generated, rollout

	def __iter__(self):
		return _Generator_Iterator(self, step_threshold=self.step_threshold,
		                           track_stats_tau=self.stats.tau if self.stats is not None else None,
	             step_cutoff=self.step_cutoff, episode_cutoff=self.episode_cutoff,
	             step_limit=self.step_limit, episode_limit=self.episode_limit, )

	def __call__(self, *args, **kwargs):
		return self._rollout(*args, **kwargs)[-1]