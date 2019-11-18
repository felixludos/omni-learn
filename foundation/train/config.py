
import sys, os
import toml

from .. import util

_config_registry = {}
def register_config(name, path):
	_config_registry[name] = path
def recover_path(name):
	if name in _config_registry:
		return _config_registry[name]
	assert os.path.isfile(name), 'invalid path: {}'.format(name)
	return name


def configurize(data):
	if isinstance(data, dict):
		return Config({k:configurize(v) for k,v in data.items()})
	if isinstance(data, list):
		return util.tlist(configurize(x) for x in data)
	return data


def load_config(path, process=True, parents=None):

	path = recover_path(path)
	data = toml.load(path)

	if process:
		data = configurize(data)

		if parents is not None and 'parents' in data:
			todo = []
			for parent in data.parents: # prep new parents
				ppath = _config_registry[parent] if parent in _config_registry else parent
				if ppath not in parents:
					todo.append(ppath)
					parents[ppath] = None
			for ppath in todo: # load parents
				parents[ppath] = load_config(ppath, parents=parents)

	return data


def _add_default_parent(C):
	for child in C.values():
		if isinstance(child, Config):
			child._parent_obj_for_defaults = C
			_add_default_parent(child)

def _clean_up_reserved(C):
	bad = []
	for k,v in C.items():
		if v is '_x_':
			bad.append(k)
		elif isinstance(v, Config):
			_clean_up_reserved(v)
	for k in bad:
		del C[k]




def get_config(path, parent_defaults=True):

	parents = {}

	root = load_config(path, parents=parents)

	if len(parents): # topo sort parents

		def _get_parents(n):
			if 'parents' in n:
				return [parents[recover_path(p)] for p in n.parents]
			return []
			if n not in parents:
				n = recover_path(n)
			return parents[n].parents if 'parents' in parents[n] else []

		order = util.toposort(root, _get_parents)

		root = order.pop()
		while len(order):
			root.update(order.pop())

	_clean_up_reserved(root)

	if parent_defaults:
		_add_default_parent(root)

	return root


class Config(util.NS):
	def __init__(self, *args, _parent_obj_for_defaults=None, **kwargs):
		self.__dict__['_parent_obj_for_defaults'] = _parent_obj_for_defaults
		super().__init__(*args, **kwargs)

	def update(self, other):
		if not isinstance(other, Config):
			return super().update(other)

		for k, v in other.items():
			if k in self and v is '_x_': # reserved for deleting settings in parents
				del self[k]
			elif k in self and isinstance(v, Config) and isinstance(self[k], Config):
				self[k].update(v)
			else:
				self[k] = v

	def __getitem__(self, item):
		if item not in self and item[0] != '_' and self._parent_obj_for_defaults is not None:
			return self._parent_obj_for_defaults[item]
		return super().__getitem__(item)

	def __contains__(self, item):
		if self._parent_obj_for_defaults is not None and item[0] != '_' and not super().__contains__(item):
			return item in self._parent_obj_for_defaults
		return super().__contains__(item)
