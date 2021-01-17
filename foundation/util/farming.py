
from humpack.farming import Farmer, make_ghost, Replicator, Parallelizer, replicate, Cloner

# import sys, os
# import torch
# import numpy as np
# import torch.multiprocessing as mp
# import itertools
# import traceback

# class ExceptionWrapper(object):
# 	r"""Wraps an exception plus traceback to communicate across threads"""
# 	def __init__(self, exc_info):
# 		# It is important that we don't store exc_info, see
# 		# NOTE [ Python Traceback Reference Cycle Problem ]
# 		self.exc_type = exc_info[0]
# 		self.exc_msg = "".join(traceback.format_exception(*exc_info))
#
# def _worker_loop(fn, private_args, in_queue, out_queue, unique_args={}, init_fn=None):
# 	torch.set_num_threads(1)
#
# 	output_args = None
# 	if init_fn is not None:
# 		args = private_args.copy()
# 		args.update(unique_args)
# 		try:
# 			output_args = init_fn(**args)
# 		except Exception:
# 			print('failed')
# 			out_queue.put(ExceptionWrapper(sys.exc_info()))
# 			return
#
# 	while True:
# 		all_args = private_args.copy()
# 		all_args.update(unique_args)
# 		if output_args is not None:
# 			all_args.update(output_args)
#
# 		args = in_queue.get()
# 		if args is None:
# 			break
# 		try:
# 			shared_args, volatile_args = args
# 			all_args.update(shared_args)
# 			all_args.update(volatile_args)
# 			output = fn(**all_args)
# 		except Exception:
# 			out_queue.put(ExceptionWrapper(sys.exc_info()))
# 		else:
# 			out_queue.put(output)
#
# class Farmer(object):
# 	'''
# 	Farms computation (of functions with both private and shared args) to other processes
# 	4 types of arguments (all of which are dicts):
# 	- shared_args = arguments to be sent to workers when dispatched from master process
# 	- private_args = arguments that all workers own privately (eg. data you dont want to pass in each dispatch)
# 	- unique_worker_args = arguments that each worker owns privately (eg. random seeds)
# 	- volatile_args = arguments generated and sent for each dispatch
#
# 	'''
# 	def __init__(self, fn, shared_args={}, private_args={}, unique_worker_args=None, volatile_gen=None,
# 	             init_fn=None, num_workers=0, timeout=20, waiting=None, auto_dispatch=True):
# 		'''
#
# 		:param fn:
# 		:param shared_args:
# 		:param private_args:
# 		:param unique_worker_args:
# 		:param volatile_gen:
# 		:param init_fn:
# 		:param num_workers:
# 		:param timeout:
# 		:param waiting:
# 		:param auto_dispatch:
# 		'''
#
# 		if unique_worker_args is not None:
# 			try:
# 				num_workers = len(unique_worker_args)
# 			except TypeError:
# 				pass
#
# 		self.num_workers = num_workers
# 		self.shared_args = shared_args
# 		self.volatile_gen = volatile_gen
# 		self.timeout = timeout
# 		self.workers = None
#
# 		self.in_queue = mp.Queue()
# 		self.outstanding = 0
# 		self.auto_dispatch = auto_dispatch
#
# 		self.dispatch_num = 1
#
# 		if num_workers > 0:
# 			if unique_worker_args is not None:  # list of dicts
# 				assert len(unique_worker_args) == num_workers
# 			else:
# 				unique_worker_args = [{}] * num_workers
#
# 			self.out_queue = mp.Queue()
# 			self.workers = [
# 				mp.Process(target=_worker_loop, args=(fn, private_args, self.in_queue, self.out_queue, u, init_fn))
# 				for i, u in enumerate(unique_worker_args)]
#
# 			for w in self.workers:
# 				w.daemon = True  # ensure that the worker exits on process exit
# 				w.start()
#
# 			if waiting is None:
# 				waiting = num_workers if auto_dispatch else 0
# 			self._dispatch(waiting)
#
# 		else:
# 			self.fn = fn
# 			self.args = private_args.copy()
# 			if init_fn is not None:
# 				output_args = init_fn(**private_args)
# 				if output_args is not None:
# 					self.args.update(output_args)
#
# 	def _get_volatile_args(self):
# 		if self.volatile_gen is not None:
# 			return next(self.volatile_gen)
# 		return {}
#
# 	def _dispatch(self, n=1, args=None):
# 		if args is None:
# 			args = self.shared_args
# 		for _ in range(n):
# 			try:
# 				self.in_queue.put((args, self._get_volatile_args()))
# 				self.outstanding += 1
# 			except StopIteration:
# 				pass
#
# 	def dispatch(self, **kwargs):
# 		self._dispatch(self.dispatch_num, args=kwargs.update(self.shared_args))
#
# 	def __len__(self):
# 		return self.outstanding
#
# 	def __iter__(self):
# 		return self
#
# 	def __next__(self):
# 		if self.auto_dispatch:
# 			self._dispatch()
# 		if self.outstanding == 0:
# 			raise StopIteration
# 		if self.workers is None: # apply fn in this process
# 			args = self.args.copy()
# 			shared_args, volatile_args = self.in_queue.get()
# 			self.outstanding -= 1
# 			args.update(shared_args)
# 			args.update(volatile_args)
# 			return self.fn(**args)
# 		output = self.out_queue.get(timeout=self.timeout)
# 		self.outstanding -= 1
# 		if isinstance(output, ExceptionWrapper):
# 			try:
# 				raise output.exc_type(output.exc_msg)
# 			except:
# 				print('***ERROR: Exception of type {} occurred'.format(output.exc_type))
# 				raise Exception(output.exc_msg)
# 				#quit()
# 		return output
#
# 	def __del__(self):
# 		if self.workers is not None:
# 			for _ in self.workers:
# 				self.in_queue.put(None)
#
#
# def make_ghost(source, execute=None):
# 	'''
# 	upon anything done to a ghost instance it will check the 'source' for the correct behavior
# 	and will call 'execute' with the function and args
#
# 	:param execute: should be callable with signature: execute(fn, args=[], kwargs={})
# 	:param source: class which contains the functionality
# 	:return: Ghost object which can be used to execute functions from 'source' in 'parent'
# 	'''
#
# 	if execute is None:
# 		def execute(fn, args=[], kwargs={}):
# 			return fn(*args, **kwargs)
#
# 	def make_listener(fn):
# 		def listener(*args, **kwargs):
# 			return execute(fn, args, kwargs)
# 		return listener
#
# 	class Ghost(object):
# 		def __getattr__(ignore, item):
# 			#print('rpl', item)
# 			if hasattr(source, item):  # any method in obj_type
# 				return make_listener(getattr(source, item))
#
# 			return execute(getattr, args=[item])
#
# 		# NOTE: use r.__len__()
# 		#def __len__(ignore):
# 		#	raise Exception('Python error: len(r) doesn\'t work on replicas, use r.__len__() instead')
#
# 		def __setattr__(self, key, value):
# 			execute(getattr(source, '__setattr__'), args=[key, value])
#
# 		def __delattr__(self, item):
# 			execute(getattr(source, '__delattr__'), args=[item])
#
# 		def __getitem__(ignore, item):
# 			return execute(getattr(source, '__getitem__'), args=[item])
#
# 		def __iter__(self):
# 			return itertools.zip_longest(*execute(getattr(source, '__iter__')))
#
# 		def __add__(ignore, other):
# 			return execute(getattr(source, '__add__'), args=[other])
#
# 	return Ghost()
#
#
# # Init fn for any replica (used by Replicator, Parallelizer, and Cloner) - creates instance of replicated object
# def _replica_init(obj_type, init_args, init_kwargs, unique_init_kwargs={}):
# 	init_kwargs.update(unique_init_kwargs)
# 	return {'obj': obj_type(*init_args, **init_kwargs)}
#
# # Run fn for any replica (used by Replicator, Parallelizer, and Cloner) - applies function to be executed to instance
# def _replica_run(obj, fn, args, kwargs, **other_args):
# 	try:
# 		return fn(obj, *args, **kwargs)
# 	except Exception as e:
# 		return e
#
# class Replicator(object):  # see 'replicate' function below
# 	def __init__(self, obj_type, replicas=None, unique_init_kwargs=None, init_args=[], init_kwargs={},
# 	             timeout=20, collate=True):
#
# 		assert replicas is not None or unique_init_kwargs is not None, 'not sure how many replicas to make'
# 		if replicas is None:
# 			replicas = len(unique_init_kwargs)
# 		self.replicas = replicas
#
# 		replica_init_args = {
# 			'obj_type': obj_type,
# 			'init_args': init_args,
# 			'init_kwargs': init_kwargs,
# 		}
#
# 		if replicas > 0:
# 			if unique_init_kwargs is not None:  # list of dicts
# 				assert len(unique_init_kwargs) == replicas
# 			else:
# 				unique_init_kwargs = [{}] * replicas
#
# 			self.in_queues = np.array([mp.Queue() for _ in range(replicas)])
# 			self.out_queues = np.array([mp.Queue() for _ in range(replicas)])
# 			self.workers = [
# 				mp.Process(target=_worker_loop, args=(
# 				_replica_run, replica_init_args, in_q, out_q, {'unique_init_kwargs': unique}, _replica_init))
# 				for i, (unique, in_q, out_q) in enumerate(zip(unique_init_kwargs, self.in_queues, self.out_queues))]
#
# 			for w in self.workers:
# 				w.daemon = True  # ensure that the worker exits on process exit
# 				w.start()
#
# 		else:  # creates an invisible wrapper
# 			assert unique_init_kwargs is None
# 			self.obj = _replica_init(**replica_init_args)['obj']
#
# 		self.obj_type = obj_type
# 		self.collate = collate
# 		self._idx = None
# 		self.timeout = timeout
#
# 	def __len__(self):
# 		return self.replicas
#
# 	def _idx_execute(self, sel):
#
# 		if len(sel) == 0:
# 			return make_ghost(self.obj_type, self._execute)
#
# 		options = np.arange(self.replicas)
#
# 		idx = []
# 		for s in sel:
# 			new = options[s]
# 			try:
# 				idx.extend(new)
# 			except:
# 				idx.append(new)
#
# 		def execute_idx(fn=None, args=[], kwargs={}):
# 			return self._execute(fn, args, kwargs, idx=idx)
# 		return make_ghost(self.obj_type, execute_idx)
#
# 	def __call__(self, *sel):
# 		return self._idx_execute(sel)
#
# 	def __getitem__(self, sel):
# 		if isinstance(sel, (int, slice)):
# 			sel = [sel]
# 		return self._idx_execute(sel)
#
# 	def _execute(self, fn=None, args=[], kwargs={}, idx=None):
#
# 		shared_args = {
# 			'args': args,
# 			'kwargs': kwargs,
# 			'fn': fn,
# 		}
#
# 		# dispatch job
# 		if self.replicas == 0:
# 			#print(fn, args, kwargs)
# 			return fn(self.obj, *args, **kwargs)
#
# 		in_queues = self.in_queues
# 		out_queues = self.out_queues
#
# 		if idx is not None:
# 			in_queues = in_queues[idx]
# 			out_queues = out_queues[idx]
#
# 		for in_q in in_queues:
# 			in_q.put((shared_args, {}))
#
# 		output = []
# 		for out_q in out_queues:
# 			output.append(out_q.get(timeout=self.timeout))
# 			if isinstance(output[-1], ExceptionWrapper):
# 				raise output.exc_type(output.exc_msg)
#
# 		# collate output
# 		if self.collate and isinstance(output[0], tuple):
# 			output = ([o[i] for o in output] for i in range(len(output[0])))
#
# 		return output
#
# class Parallelizer(Replicator): # see 'replicate' function below
# 	#def __init__(self, *args, **kwargs):
# 	#	super(Parallelizer, self).__init__(*args, **kwargs)
#
# 	def _execute(self, fn=None, args=[], kwargs={}, idx=None):
#
# 		# dispatch job
# 		if self.replicas == 0:
# 			# print(fn, args, kwargs)
# 			return fn(self.obj, *args, **kwargs)
#
# 		in_queues = self.in_queues
# 		out_queues = self.out_queues
#
# 		if idx is not None:
# 			in_queues = in_queues[idx]
# 			out_queues = out_queues[idx]
#
# 		jlen = len(in_queues)
#
# 		if len(args) > 0:
# 			assert len(args[0]) == jlen
# 			args = zip(*args)
# 		else:
# 			args = [[]] * jlen
#
# 		if len(kwargs) > 0:
# 			# uncollate kwargs
# 			kwargs = [{k: v[i] for k, v in kwargs.items()} for i in range(jlen)]
# 		else:
# 			kwargs = [{}] * jlen
#
# 		for in_q, a, k in zip(in_queues, args, kwargs):
# 			in_q.put(({'args': a, 'kwargs': k, 'fn': fn}, {}))
#
# 		output = []
# 		for out_q in out_queues:
# 			output.append(out_q.get(timeout=self.timeout))
# 			if isinstance(output[-1], ExceptionWrapper):
# 				raise output.exc_type(output.exc_msg)
#
# 		# collate output
# 		if self.collate and isinstance(output[0], tuple):
# 			output = ([o[i] for o in output] for i in range(len(output[0])))
#
# 		return output
#
# def replicate(*args, separate_args=False, ghost=False, **kwargs):
# 	'''
# 	Creates replica objects for multiprocessing. Each process contains a unique instance of the same class ('obj_type').
#
# 	There are 2 types of managers: Replicators and Parallelizers (chosen using 'separate_args')
# 	Replicator - all replicas take the same arguments when called
# 	Parallelizer - each replica takes different arguments (passed in as a list)
#
# 	Subsets of replicas may be addressed by indexing or calling the manager. If 'ghost' is true a ghost object will be
# 	returned in a addition to the manager which will apply anything done to that object to all replicas
#
# 	import time
# 	class Example:
# 		def f(self, i):
# 			time.sleep(0.5)
# 			print(i)
# 			return i+1
#
# 	replicator, ghost = replicate(Example, replicas=8, ghost=True)
#
# 	replicator(4).f(5) # executes f on replica 4
# 	replicator[0,2:6].f(5) # executes f on replicas 0 and 2:6
# 	replicator[0,0].f(5) # executes f on replica 0 twice
# 	replicator().f(5) # executes f on all replicas
# 	ghost.f(5) # executes f on all replicas
#
# 	# parallelizers have the same behavior except for each arg and kwarg a list the same length as the number of replicas to be executed must be passed in.
#
# 	:param args: for manager (check Replicator)
# 	:param separate_args: if true, then for each argument in the source function, a list of arguments must be passed one for each replica
# 	:param kwargs: for manager (check Replicator
# 	:return: manager, replicas (essentially a voodoo doll)
# 	'''
#
# 	Manager = Parallelizer if separate_args else Replicator
#
# 	manager = Manager(*args, **kwargs)
#
# 	if not ghost:
# 		return manager
#
# 	replicas = make_ghost(manager.obj_type, manager._execute)
#
# 	return manager, replicas # anything done to the replicas object will be applied in parallel to all replicas, replicator holds info about replicas
#
#
# class Cloner(Farmer):
# 	'''
# 	executes any method in 'obj_type' N (default=num_workers) times with the same args using a pool of 'num_workers' workers
#
# 	unique init args can be passed to each worker, but args to all workers for each dispatch will be the same
#
# 	example:
#
# 	import time
# 	class Example:
# 		def f(self, i):
# 			time.sleep(0.5)
# 			print(i)
# 			return i+1
#
# 	clones = Cloner(Example, num_workers=4)
#
# 	out = clones(8).f(5)
# 	# takes about 1 sec and prints "5" 8 times
# 	# 'out' now contains [6, 6, 6, 6, 6, 6, 6, 6]
#
# 	'''
# 	def __init__(self, obj_type, default_N=None, init_args=[], init_kwargs={}, unique_worker_args=None,
# 	             num_workers=0, collate=True, timeout=20):
#
# 		assert num_workers is not None or unique_worker_args is not None, 'must specify how many workers to use'
#
# 		if unique_worker_args is not None:
# 			unique_worker_args = [{'unique_init_kwargs': unique} for unique in unique_worker_args]
# 			num_workers = len(unique_worker_args)
#
# 		worker_init_args = {
# 			'obj_type': obj_type,
# 			'init_args': init_args,
# 			'init_kwargs': init_kwargs,
# 		}
#
# 		super(Cloner, self).__init__(fn=_replica_run, init_fn=_replica_init,
# 		                             private_args=worker_init_args, unique_worker_args=unique_worker_args,
# 		                             num_workers=num_workers, auto_dispatch=False, timeout=timeout)
# 		self.default_N = max(num_workers, 1) if default_N is None else default_N
# 		self.obj_type = obj_type
# 		self.collate = collate
#
# 		#self.clones = make_ghost(self.obj_type, self._execute)
#
# 	def __call__(self, N=None):
# 		def execute_N(fn=None, args=[], kwargs={}):
# 			return self._execute(fn, args, kwargs, N)
# 		return make_ghost(self.obj_type, execute_N)
#
# 	def _execute(self, fn=None, args=[], kwargs={}, N=None): # execution should not change the state of the clone
#
# 		self.shared_args = {
# 			'args': args,
# 			'kwargs': kwargs,
# 			'fn': fn,
# 		}
#
# 		if N is None:
# 			N = self.default_N
#
# 		# dispatch job
# 		self._dispatch(N)
#
# 		# collect responses
# 		output = [out for out in self]
#
# 		# collate output
# 		if self.collate and isinstance(output[0], tuple):
# 			output = ([o[i] for o in output] for i in range(len(output[0])))
#
# 		return output

