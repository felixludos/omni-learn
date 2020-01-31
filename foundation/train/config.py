
import sys, os
import yaml
import json

from .registry import find_config_path, create_component, MissingConfigError, _reserved_names, _appendable_keys
from .. import util


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
	path = find_config_path(path)
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
			# ppath = _config_registry[parent] if parent in _config_registry else parent
			ppath = find_config_path(parent)
			if ppath not in parents:
				todo.append(ppath)
				parents[ppath] = None
		for ppath in todo: # load parents
			parents[ppath] = load_single_config(ppath, parents=parents)

	return data

def _check_for_load(config, parent_defaults=True):

	if 'load' in config:
		lparents = {}
		load = load_single_config(config.load, parents=lparents)
		assert len(lparents) == 0, 'Loaded configs are not allowed to have parents.'
		load.update(config, parent_defaults=parent_defaults)
		config = load

	return config

def merge_configs(configs, parent_defaults=True):
	'''
	configs should be ordered from oldest to newest (ie. parents first, children last)
	also configs can contain "load"
	'''

	if not len(configs):
		return Config()

	child = configs.pop()
	merged = merge_configs(configs, parent_defaults=parent_defaults)

	load = child.load if 'load' in child else None
	merged.update(child)

	if load is not None:
		lparents = {}
		load = load_single_config(load, parents=lparents)
		assert len(lparents) == 0, 'Loaded configs are not allowed to have parents.'
		load.update(merged, parent_defaults=parent_defaults)
		merged = load

	return merged


def get_config(path=None, parent_defaults=True, include_load_history=False): # Top level function

	if path is None:
		return Config()

	parents = {}

	root = load_single_config(path, parents=parents)

	pnames = []
	if len(parents): # topo sort parents
		def _get_parents(n):
			if 'parents' in n:
				return [parents[find_config_path(p)] for p in n.parents]
			return []

		order = util.toposort(root, _get_parents)

		# for analysis, record the history of all loaded parents

		for p in order:
			if 'parents' in p:
				for prt in p.parents:
					if len(pnames) == 0 or prt != pnames[-1]:
						pnames.append(prt)

		# root = merge_configs(order, parent_defaults=parent_defaults)
		# root = _check_for_load(order.pop(), parent_defaults=parent_defaults)
		# while len(order):
		#
		# 	child = order.pop()
		# 	will_load = None
		# 	if 'load' in child:
		# 		will_load = child.load
		#
		#
		#
		# 	root.update()
		#
		# 	root.update(_check_for_load(order.pop(), parent_defaults=parent_defaults),
		# 	            parent_defaults=parent_defaults)

		order = list(reversed(order))


	else: # TODO: clean up
		order = [root]

	root = merge_configs(order, parent_defaults=parent_defaults) # update to connect parents and children in tree and remove reversed - see Config.update

	if include_load_history:
		root._load_history = pnames
	if 'parents' in root:
		del root.parents

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

def parse_config(argv=None, parent_defaults=True, include_load_history=False):
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
			# assert term in _config_registry or os.path.isfile(term), 'invalid config name/path: {}'.format(term)
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

	root = get_config(root, parent_defaults=parent_defaults, include_load_history=include_load_history)

	return root

_reserved_names.update({'_x_'})

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
		if v == '_x_': # maybe include other _reserved_names
			bad.append(k)
		elif isinstance(v, Config):
			_clean_up_reserved(v)
	for k in bad:
		del C[k]

# TODO: find a way to avoid this ... probably not easy
_print_waiting = False
_print_indent = 0
def _print_with_indent(s):
	return s if _print_waiting else ''.join(['  '*_print_indent, s])



class Config(util.NS): # TODO: allow adding aliases
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
	(for values of "appendable" keys)
	"+{}" = '{}' gets appended to preexisting value if if it exists
		(otherwise, the "+" is removed and the value is turned into a list with itself as the only element)

	Also, this is Transactionable, so when creating subcomponents, the same instance is returned when pulling the same
	sub component again.

	NOTE: avoid setting '__obj' keys (unless you really know what you are doing)

	'''
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
			super().update(other)
		else:
			for k, v in other.items():
				if self.contains_nodefault(k) and '_x_' == v: # reserved for deleting settings in parents
					del self[k]
				elif self.contains_nodefault(k) and isinstance(v, Config) and isinstance(self[k], Config):
					self[k].update(v)

				elif k in _appendable_keys and v[0] == '+':
					# values of appendable keys can be appended instead of overwritten,
					# only when the new value starts with "+"
					vs = []
					if self.contains_nodefault(k):
						prev = self[k]
						if not isinstance(prev, list):
							prev = [prev]
						vs = prev
					vs.append(v[1:])
					self[k] = vs
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

	def _process_val(self, item, val, *defaults, silent=False, force_new=False, defaulted=False, byparent=False):
		global _print_indent, _print_waiting

		if isinstance(val, dict) and '_type' in val:
			# WARNING: using pull will automatically create registered sub components

			# no longer an issue
			# assert not byparent, 'Pulling a sub-component from a parent is not supported (yet): {}'.format(item)

			assert not defaulted


			# TODO: should probably be deprecated - just register a "list" component separately
			# if val['_type'] == 'list':
			# 	print(_print_with_indent('{} (type={}): '.format(item, val['_type'])))
			# 	terms = []
			# 	for i, v in enumerate(val._elements): # WARNING: elements must be listed with '_elements' key
			# 		terms.append(self._process_val('({})'.format(i), v))
			# 	val = tuple(terms)
			# else:

			type_name = val['_type']
			mod_info = ''
			if '_mod' in val:
				mods = val['_mod']
				if not isinstance(mods, (list, tuple)):
					mods = mods,

				mod_info = ' (mods=[{}])'.format(', '.join(m for m in mods)) if len(mods) > 1 \
					else ' (mod={})'.format(mods[0])

			cmpn = None
			if self.in_transaction() and '__obj' in val and not force_new:
				print('WARNING: would usually reuse {} now, but instead creating a new one!!')
				# cmpn = val['__obj']

			creation_note = 'Creating ' if cmpn is None else 'Reusing '

			if not silent:
				print(_print_with_indent('{}{} (type={}){}{}'.format(creation_note, item, type_name, mod_info,
				                                                     ' (in parent)' if byparent else '')))

			if cmpn is None:
				_print_indent += 1
				cmpn = create_component(val)
				_print_indent -= 1

			if self.in_transaction():
				self[item]['__obj'] = cmpn
			else:
				print('WARNING: this Config is NOT currently in a transaction, so all subcomponents will be created '
				      'again everytime they are pulled')

			val = cmpn


		elif isinstance(val, str) and val[:2] == '<>':  # alias
			alias = val[2:]
			assert not byparent, 'Using an alias from a parent is not supported: {} {}'.format(item, alias)

			if not silent:
				print(_print_with_indent('{} --> '.format(item)), end='')
			_print_waiting = True
			val = self.pull(alias, *defaults, silent=silent)
			_print_waiting = False

		else:
			if not silent:
				print(_print_with_indent('{}: {}{}'.format(item, val, ' (by default)' if defaulted
						else (' (in parent)' if byparent else ''))))

		return val


	def pull(self, item, *defaults, silent=False, force_new=False, _byparent=False, _defaulted=False):
		# TODO: change defaults to be a keyword argument providing *1* default, and have item*s* instead,
		#  which are the keys to be checked in order

		defaulted = item not in self
		byparent = False
		if defaulted:
			if len(defaults) == 0:
				raise MissingConfigError(item)
			val, *defaults = defaults
		else:
			byparent = not self.contains_nodefault(item)
			val = self[item]

		if byparent: # if the object was found
			val = self.__dict__['_parent_obj_for_defaults'].pull(item, silent=silent, force_new=force_new, _byparent=True)
		else:
			val = self._process_val(item, val, *defaults, silent=silent, defaulted=defaulted or _defaulted,
			                        byparent=byparent or _byparent, force_new=force_new, )

		if type(val) == list: # TODO: a little heavy handed
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

