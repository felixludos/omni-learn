
import sys, os
import yaml
import json

from .model import create_component, MissingConfigError
from .. import util

_config_registry = {}
def register_config(name, path):
	assert os.path.isfile(path), 'Cant find config file: {}'.format(path)
	_config_registry[name] = path
def register_config_dir(path):
	assert os.path.isdir(path)
	for fname in os.listdir(path):
		if os.path.isfile(os.path.join(path, fname)):
			name = fname.split('.')[0]
			register_config(name, os.path.join(path, fname))
def recover_path(name):
	if name in _config_registry:
		return _config_registry[name]
	assert os.path.isfile(name), 'invalid path: {}'.format(name)
	return name


nones = {'None', '_none', '_None', 'null', 'nil', }

def configurize(data):
	if isinstance(data, dict):
		return Config({k:configurize(v) for k,v in data.items()})
	if isinstance(data, list):
		return [configurize(x) for x in data]
	if isinstance(data, str) and data in nones:
		return None
		# return util.tlist(configurize(x) for x in data)
	return data

class YamlifyError(Exception):
	def __init__(self, obj):
		super().__init__('Unable to yamlify: {} (type={})'.format(obj, type(obj)))
		self.obj = obj

def yamlify(data):
	if data is None:
		return '_None'
	if isinstance(data, dict):
		return {k:yamlify(v) for k,v in data.items()}
	if isinstance(data, (list, tuple, set)):
		return [yamlify(x) for x in data]
	if isinstance(data, util.primitives):
		return data

	raise YamlifyError(data)

def load_single_config(path, process=True, parents=None):

	path = recover_path(path)
	with open(path,'r') as f:
		data = yaml.load(f)

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
				parents[ppath] = load_single_config(ppath, parents=parents)

	return data


def get_config(path, parent_defaults=True): # Top level function

	parents = {}

	root = load_single_config(path, parents=parents)

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
			root.update(order.pop(), parent_defaults=parent_defaults)

	return root

# def _nested_set(obj, keys, value):
# 	for key in keys[:-1]:
# 		obj = obj[key]
# 	obj[keys[-1]] = value

class ParsingError(Exception):
	pass
class NoConfigFound(Exception):
	def __init__(self):
		super().__init__('Either provide a config name/path as the first argument, or set your $FOUNDATION_CONFIG environment variable to a config name/path')

def parse_cmd_args(argv=None, parent_defaults=True):

	if argv is None:
		argv = sys.argv

	argv = argv[1:]

	if len(argv) == 0 or (argv[0] not in _config_registry and not os.path.isfile(argv[0])):

		if 'FOUNDATION_CONFIG' not in os.environ:
			print('WARNING: No parent config found (using command line args only)')
			parent = None
			# raise NoConfigFound() # TODO possibly allow no specifying a config

		else:
			parent = os.environ['FOUNDATION_CONFIG']
			assert os.path.isfile(parent), 'Invalid setting of $FOUNDATION_CONFIG: {}'.format(parent)

	else:
		parent = argv[0]

	if parent is not None:
		argv = argv[1:]
		parent = get_config(parent, parent_defaults=parent_defaults)

	if len(argv) == 0:
		return parent

	terms = iter(argv)

	root = Config()

	term = next(terms)
	if term[:2] != '--':
		raise ParsingError(term)
	done = False
	while not done:
		keys = term[2:].split('.')
		values = []
		try:
			val = next(terms)
			while val[:2] != '--':
				try:
					values.append(configurize(json.loads(val)))
				except json.JSONDecodeError:
					values.append(val)
				val = next(terms)
			term = val
		except StopIteration:
			done = True

		if len(values) == 0:
			values = [True]
		if len(values) == 1:
			values = values[0]
		root[keys] = values

	if parent is not None:
		parent.update(root, parent_defaults=parent_defaults)
	else:
		return root

	return parent

def _add_default_parent(C):
	for k, child in C.items():
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


class Config(util.NS): # TODO: allow adding aliases
	def __init__(self, *args, _parent_obj_for_defaults=None, **kwargs):
		self.__dict__['_parent_obj_for_defaults'] = _parent_obj_for_defaults
		# self.__dict__['_aliases'] = {}
		super().__init__(*args, **kwargs)

	# def register_alias(self, attr, alias):
	# 	if attr not in self._aliases:
	# 		self._aliases[attr] = set()
	# 	self._aliases[attr].add(alias)

	def update(self, other, parent_defaults=True):
		if not isinstance(other, Config):
			return super().update(other)

		for k, v in other.items():
			if k in self and v is '_x_': # reserved for deleting settings in parents
				del self[k]
			elif k in self and isinstance(v, Config) and isinstance(self[k], Config):
				self[k].update(v)
			else:
				self[k] = v

		_clean_up_reserved(self)

		if parent_defaults:
			_add_default_parent(self)
			
		x = 1+1

	def _single_get(self, item):
		if not self.contains_nodefault(item) and item[0] != '_' and self._parent_obj_for_defaults is not None:
			return self._parent_obj_for_defaults[item]
		return super().__getitem__(item)

	def __getitem__(self, item):

		if '.' in item:
			item = item.split('.')

		if isinstance(item, (list, tuple)):
			return self._single_get(item[0])[item[1:]]
		return self._single_get(item)

	def __setitem__(self, key, value):
		if '.' in key:
			key = key.split('.')

		if isinstance(key, (list, tuple)):
			if len(key) == 1:
				return super().__setitem__(key[0], value)
			return super().__getitem__(key[0]).__setitem__(key[1:], value)

		return super().__setitem__(key, value)


	def get_nodefault(self, item):
		return super().__getitem__(item)

	def contains_nodefault(self, item):
		return super().__contains__(item)

	def pull(self, item, *defaults): # pull for each arg should only be called once!

		defaulted = item not in self
		if defaulted:
			if len(defaults) == 0:
				raise MissingConfigError(item)
			val = defaults[0]
			defaults = defaults[1:]
		else:
			val = self[item]

		if isinstance(val, dict) and '_type' in val:  # WARNING: using pull will automatically create registered sub components
			print('Creating sub-component: {} (type={})'.format(item, val['_type']))
			val = create_component(val)

		elif isinstance(val, str) and val[:2] == '<>':  # alias
			alias = val[2:]
			val = self.pull(alias, *defaults)
			print('{} is an alias for {}'.format(item, alias))

		else:
			print('{}: {}'.format(item, val))


		if defaulted:
			print('{} default: {}'.format(item, val))

		return val

	def export(self, path=None):

		data = yamlify(self)

		if path is not None:
			if os.path.isdir(path):
				path = os.path.join(path, 'config.yaml')
			with open(path, 'w') as f:
				yaml.dump(data, f)
			return path

		return data

	def __contains__(self, item):
		if self._parent_obj_for_defaults is not None and item[0] != '_' and not super().__contains__(item):
			return item in self._parent_obj_for_defaults
		return super().__contains__(item)
