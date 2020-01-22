
import sys, os
import yaml
import json

from .model import create_component, MissingConfigError
from .. import util

_config_registry = {}
def register_config(name, path):
	assert os.path.isfile(path), 'Cant find config file: {}'.format(path)
	_config_registry[name] = path
def register_config_dir(path, recursive=False, prefix=None, joiner=''):
	assert os.path.isdir(path)
	for fname in os.listdir(path):
		parts = fname.split('.')
		candidate = os.path.join(path, fname)
		if os.path.isfile(candidate) and len(parts) > 1 and parts[-1] in {'yml', 'yaml'}:
			name = parts[0]
			if prefix is not None:
				name = joiner.join([prefix, name])
			register_config(name, os.path.join(path, fname))
		elif recursive and os.path.isdir(candidate):
			prefix = fname if prefix is None else os.path.join(prefix, fname)
			register_config_dir(candidate, recursive=recursive, prefix=prefix, joiner=os.sep)

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

def load_config_from_path(path, process=True):
	path = recover_path(path)
	with open(path, 'r') as f:
		data = yaml.load(f)

	if process:
		return configurize(data)
	return data

def load_single_config(data, process=True, parents=None): # data can either be an existing config or a path to a config

	if isinstance(data, str):
		data = load_config_from_path(data, process=process)

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


def get_config(path=None, parent_defaults=True): # Top level function

	if path is None:
		return Config()

	parents = {}

	root = load_single_config(path, parents=parents)

	if len(parents): # topo sort parents
		def _get_parents(n):
			if 'parents' in n:
				return [parents[recover_path(p)] for p in n.parents]
			return []

		order = util.toposort(root, _get_parents)

		# for analysis, record the history of all loaded parents
		pnames = []
		for p in order:
			if 'parents' in p:
				for prt in p.parents:
					if len(pnames) == 0 or prt != pnames[-1]:
						pnames.append(prt)

		root = order.pop()
		while len(order):
			root.update(order.pop(), parent_defaults=parent_defaults)

		root.info.history = pnames
		if 'parents' in root:
			del root.parents
	else: # TODO: clean up
		root.update(Config()) # update to connect parents and children in tree and remove reversed - see Config.update

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

def parse_config(argv=None, parent_defaults=True):
	# WARNING: 'argv' should not be equivalent to sys.argv here (no script name in element 0)

	if argv is None:
		argv = sys.argv[1:]

	# argv = argv[1:]

	root = Config() # from argv

	parents = []
	for term in argv:
		if len(term) >= 2 and term[:2] == '--':
			break
		else:
			assert term in _config_registry or os.path.isfile(term), 'invalid config name/path: {}'.format(term)
			parents.append(term)
	root.parents = parents

	argv = argv[len(parents):]

	if len(argv):

		terms = iter(argv)

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

	root = get_config(root, parent_defaults=parent_defaults)

	return root

def _add_default_parent(C):
	for k, child in C.items():
		if isinstance(child, Config):
			child._parent_obj_for_defaults = C
			_add_default_parent(child)
		elif isinstance(child, (tuple, list, set)):
			for c in child:
				if isinstance(c, Config):
					c._parent_obj_for_defaults = C
					_add_default_parent(c)

def _clean_up_reserved(C):
	bad = []
	for k,v in C.items():
		if '_x_' == v:
			bad.append(k)
		elif isinstance(v, Config):
			_clean_up_reserved(v)
	for k in bad:
		del C[k]

_print_waiting = False
_print_indent = 0
def _print_with_indent(s):
	return s if _print_waiting else ''.join(['  '*_print_indent, s])

'''
Features:

Keys:
'_{}' = protected - not visible to children
({1}, {2}, ...) = [{1}][{2}]...
'{1}.{2}' = ['{1}']['{2}']
if {} not found: first check parent (if exists) otherwise create self[{}] = Config(parent=self)

Values:
'<>{}' = alias to key '{}'
'_x_' = (only when merging) remove this key locally, if exists
'__x__' = dont default this key and behaves as though it doesnt exist (except on iteration)

'''


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
			if self.contains_nodefault(k) and '_x_' == v: # reserved for deleting settings in parents
				del self[k]
			elif self.contains_nodefault(k) and isinstance(v, Config) and isinstance(self[k], Config):
				self[k].update(v)
			else:
				self[k] = v

		_clean_up_reserved(self)

		if parent_defaults:
			_add_default_parent(self)


	def _single_get(self, item):

		if not self.contains_nodefault(item) \
				and self._parent_obj_for_defaults is not None \
				and item[0] != '_':
			return self._parent_obj_for_defaults[item]

		return self.get_nodefault(item)

	def __getitem__(self, item):

		if '.' in item:
			item = item.split('.')

		if isinstance(item, (list, tuple)):
			if len(item) == 1:
				item = item[0]
			else:
				return self._single_get(item[0])[item[1:]]
		return self._single_get(item)

	def __setitem__(self, key, value):
		if '.' in key:
			key = key.split('.')

		if isinstance(key, (list, tuple)):
			if len(key) == 1:
				return self.__setitem__(key[0], value)
			return self.__getitem__(key[0]).__setitem__(key[1:], value)

		return super().__setitem__(key, value)

	def __contains__(self, item):
		if '.' in item:
			item = item.split('.')

		if isinstance(item, (tuple, list)):
			if len(item) == 1:
				item = item[0]
			else:
				return item[0] in self and item[1:] in self[item[0]]

		return self.contains_nodefault(item) \
			or (not super().__contains__(item)
				and self._parent_obj_for_defaults is not None
				and item[0] != '_'
			    and item in self._parent_obj_for_defaults)

	def get_nodefault(self, item):
		val = super().__getitem__(item)
		if val == '__x__':
			raise KeyError(item)
		return val

	def contains_nodefault(self, item):
		# if isinstance(item, (tuple, list)):
		# 	if len(item) == 1:
		# 		item = item[0]
		# 	else:
		# 		return item[0] in self and item[1:] in self
		if super().__contains__(item):
			return super().__getitem__(item) != '__x__'
		return False

	def _process_val(self, item, val, *defaults, defaulted=False, byparent=False):
		global _print_indent, _print_waiting

		if isinstance(val, dict) and '_type' in val:
			# WARNING: using pull will automatically create registered sub components
			assert not byparent, 'Pulling a sub-component from a parent is not supported (yet): {}'.format(item)
			print(_print_with_indent('{} (type={}): '.format(item, val['_type'])))
			_print_indent += 1
			if val['_type'] == 'list':
				terms = []
				for i, v in enumerate(val._elements): # WARNING: elements must be listed with '_elements' key
					terms.append(self._process_val('({})'.format(i), v))
				val = tuple(terms)
			else:
				val = create_component(val)
			_print_indent -= 1

		elif isinstance(val, str) and val[:2] == '<>':  # alias
			alias = val[2:]
			assert not byparent, 'Using an alias from a parent is not supported: {} {}'.format(item, alias)

			print(_print_with_indent('{} --> '.format(item)), end='')
			_print_waiting = True
			val = self.pull(alias, *defaults)
			_print_waiting = False

		else:
			print(_print_with_indent('{}: {}{}'.format(item, val, ' (by default)' if defaulted else (' (by parent)' if byparent else ''))))

		return val


	def pull(self, item, *defaults): # pull for each arg should only be called once!

		defaulted = item not in self
		byparent = False
		if defaulted:
			if len(defaults) == 0:
				raise MissingConfigError(item)
			val, *defaults = defaults
		else:
			byparent = not self.contains_nodefault(item)
			val = self[item]

		val = self._process_val(item, val, *defaults, defaulted=defaulted, byparent=byparent)

		if isinstance(val, list):
			val = tuple(val)

		return val

	def __str__(self):
		return super().__repr__()

	def export(self, path=None):

		data = yamlify(self)

		if path is not None:
			if os.path.isdir(path):
				path = os.path.join(path, 'config.yaml')
			with open(path, 'w') as f:
				yaml.dump(data, f)
			return path

		return data

