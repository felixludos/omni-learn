import ray
#import sequential_ray as ray
#print('using sequential')
import numpy as np
import os
import torch
from torch.autograd import grad
import torch.distributions as distrib
from foundation import util
from foundation.util import NS
from tabulate import tabulate
from foundation.agents.optim import Conjugate_Gradient
import time

@ray.remote(num_return_vals=1)
def save_checkpoint(save_path, args, policy, baseline, stats, episode, best):
	ckpt = {
		'policy': policy.state_dict(),
		'baseline': baseline,
		'stats': stats.export(),
		'args': args,
		'total-episodes': episode,
		'best': best,
	}
	torch.save(ckpt, save_path)

def load_checkpoint(path):
	if os.path.isdir(path):
		latest = os.listdir(path)[-1]
		if 'pth.tar' in latest:
			path = os.path.join(path, 'model_best.pth.tar')
		else:
			path = os.path.join(path, latest, 'model_best.pth.tar')

	return path, torch.load(path)

def print_stats(stats, time_stats=None, logger=None, verbose=False, total_episodes=0, itr=0, num_iter=1, best_perf=None):
	if verbose:
		# print stats TODO: using tabulate
		print('.' * 50)
		print('ITERATION {}/{} : episode={}'.format(itr + 1, num_iter, total_episodes))
		print('Stats:')
		print(tabulate(sorted([(name, stat.val, stat.avg) for name, stat in stats.items()])))
		if time_stats is not None:
			print('Timing:')
			print(tabulate(sorted([(name, stat.val, stat.avg) for name, stat in time_stats.items()])))
	else:  # condensed print
		print("[ {} ] {:}/{:} (ep={}) : {:5.3f} {:5.3f} {:5.3} ".format((time.strftime("%m-%d-%y %H:%M:%S")), itr + 1,
		                                                                num_iter, total_episodes,
		                                                                stats['optim-returns'].avg,
		                                                                stats['eval-reward'].avg,
		                                                                best_perf if best_perf is not None else ''))

	if logger is not None:
		info = {name: stats[name].avg for name in stats.keys()}
		info.update({'timing-' + name: time_stats[name].val for name in time_stats.keys()})
		for k, v in info.items():
			logger.scalar_summary(k, v, total_episodes)

# Generate trajectories

def generate_trajectory(policy, env, T=None, seed=None, include_last_state=False):
	# if env is None:
	# 	env = envs.get_environment(env_name)
	
	if seed is not None:
		env.seed(seed)
		util.set_seed(seed)

	T = min(T, env.horizon) if T is not None else env.horizon

	t = 0
	o = env.reset()

	#print(list(policy.parameters())[-1], o[0])

	path = NS(
		observations=[],
		actions=[],
		rewards=[],
		# agent_infos=[],
		# env_infos=[],
		#terminated=False,
	)

	done = False

	while t < T and not done:
		#a, agent_info = policy(o)
		a = policy(o)[0]
		next_o, r, done, env_info = env.step(a)
		path.observations.append(o)
		path.actions.append(a)
		path.rewards.append(r)
		# path.agent_infos.append(agent_info)
		# path.env_infos.append(env_info)
		o = next_o
		t += 1

	#path.terminated = done

	if include_last_state:
		path.observations.append(o)

	if type(path.observations[0]) == list:  # multi agent - stack second dim
		path.observations = [np.stack(obs) for obs in zip(*path.observations)]
		path.actions = [np.stack(act) for act in zip(*path.actions)]
		path.rewards = [np.stack(rew) for rew in zip(*path.rewards)]
	else:
		path.observations = np.stack(path.observations)
		path.actions = np.stack(path.actions)
		path.rewards = np.stack(path.rewards)

	return path

def compute_advantages(coeff, returns, obs, time_order=1, obs_order=1):
	'''

	:param returns: computed returns ([B] x T x 1)
	:param obs: ([B] x T x O)
	:param baseline: callable(T x O) - returns (T x 1)
	:return:
	'''
	
	if coeff is None:
		return returns
	# print(coeff.shape)
	# print(feature_extractor(obs, time_order=time_order, obs_order=obs_order).shape)
	return returns - feature_extractor(obs, time_order=time_order, obs_order=obs_order) @ coeff

def compute_returns(rewards, discount=None):  # rewards T x 1
	returns = []
	run_sum = 0
	for i in range(rewards.shape[0]-1,-1,-1):
		run_sum = rewards[i] + discount * run_sum
		returns.append(run_sum)
	return np.array(returns[::-1]).reshape(-1,1)

@ray.remote(num_return_vals=1)
def generate_path(policy, env, baseline=None, discount=0.99, seed=None, T=None, include_last_state=False, time_order=1, obs_order=1):

	path = generate_trajectory(policy, env, T=T, seed=seed, include_last_state=include_last_state)

	path.returns = compute_returns(path.rewards, discount=discount)
	if baseline is not None:
		path.advantages = compute_advantages(baseline, path.returns, path.observations, time_order=time_order, obs_order=obs_order)
	else:
		path.advantages = path.returns.copy()

	return path

def collate(paths):
	full = NS()
	for k in paths[0].keys():
		full[k] = np.concatenate([p[k] for p in paths])
	return full

# baseline
def feature_extractor(X, lim=None, ones=True, obs_order=1, time_order=1):
	T, O = X.shape # T x O
	
	if lim is not None:
		X = X.clip(-lim, lim)
	
	ts = np.arange(0, T).reshape(-1, 1) / 1000  # timestep
	
	terms = [X ** (n + 1) for n in range(obs_order)]
	terms.extend([ts ** (n + 1) for n in range(time_order)])
	
	if ones:
		terms.append(np.ones(ts.shape))
	
	return np.concatenate(terms, 1)

@ray.remote(num_return_vals=2)
def train_baseline(baseline, paths, reg_coeff=1e-8, time_order=1, obs_order=1):
	X = feature_extractor(paths.observations, time_order=time_order, obs_order=obs_order)
	Y = paths.returns.reshape(-1, 1)
	
	A = X.T @ X
	b = X.T @ Y
	rI = np.eye(X.shape[-1]) * reg_coeff
	
	stats = util.StatsMeter('error-before', 'error-final')
	if baseline is not None:
		stats.update('error-before', ((X @ baseline - Y) ** 2).mean())
	else:
		stats.update('error-before', (Y ** 2).mean())
	
	for _ in range(10):
		soln = np.linalg.lstsq(
			A + rI,
			b
		)[0]
		if np.isfinite(soln).all():
			break
		rI *= 10
	
	# set new coeffs
	baseline = soln
	
	stats.update('error-final', ((X @ baseline - Y) ** 2).mean())
	return baseline, stats

# policy update

def objective(pi, pi_old, actions, advantages):
	advantages = advantages / (advantages.abs().max() + 1e-8)
	#print('action',pi.log_prob(actions)[:5])
	#print('action', pi_old.log_prob(actions)[:5])
	surr = torch.mean( (pi.log_prob(actions) - pi_old.log_prob(actions)).exp() * advantages ) # CPI surrogate
	#print(surr)
	return surr

def flatten_params(params):
	return torch.cat([param.clone().view(-1) for param in params])

def FVP(v, grad_kl, params, reg_coeff):
	Fv = flatten_params(grad(grad_kl.clone() @ v, params, retain_graph=True))
	return Fv + reg_coeff * v

def cg_solve(apply_A, b, x0=None, res_tol=1e-10, nsteps=10):
	if x0 is None:
		x = torch.zeros(*b.size())
		r = b.clone()
	else:
		x = x0
		r = b - apply_A(x)
	
	p = r.clone()
	rdotr = r @ r
	for i in range(nsteps):
		Ap = apply_A(p)
		alpha = rdotr / (p @ Ap)
		x += alpha * p
		r -= alpha * Ap
		new_rdotr = r @ r
		beta = new_rdotr / rdotr
		p = r + beta * p
		rdotr = new_rdotr
		if rdotr < res_tol:
			break
	return x

@ray.remote(num_return_vals=2)
def old_npg_update(policy, paths, def_type='torch.FloatTensor', reg_coeff=1e-5, step_size=0.01, residual_tol=1e-10, max_cg_iter=10):

	stats = util.StatsMeter('returns', 'objective', 'mean-kl', 'npg-norm', 'vpg-norm', 'alpha')
	
	optim = Conjugate_Gradient(policy.parameters(), step_size=step_size, nsteps=max_cg_iter, residual_tol=residual_tol, ret_stats=True)

	advantages = torch.from_numpy(paths.advantages).type(def_type).view(-1)
	observations = torch.from_numpy(paths.observations).type(def_type)
	actions = torch.from_numpy(paths.actions).type(def_type)
	
	#print(actions.size())
	#print(advantages.size())
	#print(observations.size())

	stats.update('returns', paths.returns.mean())

	# evaluate objective
	pi, pi_old = policy.get_pi(observations, include_old=True)

	eta = objective(pi, pi_old, actions, advantages)
	stats.update('objective', eta.item())
	
	optim.zero_grad()
	eta.backward(retain_graph=True)
	
	grad_kl = flatten_params(
		grad(distrib.kl_divergence(pi, pi_old).mean(),
		     policy.parameters(), create_graph=True))

	optim_stats = optim.step(lambda v: FVP(v, grad_kl, policy, reg_coeff))
	stats.join(optim_stats)

	stats.update('mean-kl', distrib.kl_divergence(*policy.get_pi(observations, include_old=True)).mean().item())
	
	#print('old', list(policy.old_model.parameters())[-1])
	#print('new', list(policy.model.parameters())[-1])
	
	#pi, pi_old = policy.get_pi(observations, include_old=True)
	
	#new_eta = objective(pi, pi_old, actions, advantages)
	#print(new_eta.item(), eta.item(), (new_eta - eta).item())
	
	policy.update_old_model()
	
	#print('adv', advantages.max(), advantages.min())
	
	#print(stats)
	
	#quit()
	
	return policy, stats

@ray.remote(num_return_vals=2)
def npg_update(policy, paths, def_type='torch.FloatTensor', reg_coeff=1e-5, step_size=0.01, residual_tol=1e-10,
               max_cg_iter=10):
	stats = util.StatsMeter('returns', 'objective', 'mean-kl', 'npg-norm', 'vpg-norm', 'alpha')
	
	advantages = torch.from_numpy(paths.advantages).type(def_type)
	observations = torch.from_numpy(paths.observations).type(def_type)
	actions = torch.from_numpy(paths.actions).type(def_type)
	
	stats.update('returns', paths.returns.mean())
	
	# evaluate objective
	pi, pi_old = policy.get_pi(observations, include_old=True)
	
	eta = objective(pi, pi_old, actions, advantages)
	stats.update('objective', eta.item())
	
	grads = grad(eta, policy.parameters(), retain_graph=True)
	vpg = torch.cat([p.view(-1) for p in grads], 0)  # flat vpg
	stats.update('vpg-norm', vpg.norm().item())
	
	grad_kl = flatten_params(
		grad(distrib.kl_divergence(pi, pi_old).mean(),
		     policy.parameters(), create_graph=True))
	
	npg = cg_solve(lambda v: FVP(v, grad_kl, policy.parameters(), reg_coeff), vpg, res_tol=residual_tol,
	               nsteps=max_cg_iter)
	stats.update('npg-norm', npg.norm().item())
	
	alpha = (2 * step_size / (vpg @ npg)).sqrt()
	stats.update('alpha', alpha.item())
	
	update = alpha * npg
	
	i = 0
	for param in policy.parameters():
		l = param.numel()
		param.data.add_(update[i:i + l].view(*param.size()))  # gradient ascent
		i += l
	
	stats.update('mean-kl', distrib.kl_divergence(*policy.get_pi(observations, include_old=True)).mean().item())
	
	policy.update_old_model()
	
	return policy, stats

@ray.remote(num_return_vals=1)
def evaluate_policy(policy, env, seed=None, T=None):
	if seed is not None:
		env.seed(seed)
		util.set_seed(seed)

	T = min(T, env.horizon) if T is not None else env.horizon

	t = 0
	o = env.reset()
	done = False
	
	stats = util.StatsMeter('reward', 'len')
	reward = 0

	while t < T and not done:
		#a, agent_info = policy(o)
		a = policy(o)[0]
		next_o, r, done, env_info = env.step(a)
		reward += r
		o = next_o
		t += 1
	
	stats.update('reward', reward)
	stats.update('len', t)
	
	return stats

