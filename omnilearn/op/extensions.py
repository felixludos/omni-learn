import math
import omnifig as fig

from omnifig.config import ConfigList, ConfigIter


@fig.Component('pull')
def _config_pull(A):
	terms = A.pull('_terms', '<>_term', silent=True)
	args = A.pull('_args', {}, silent=True)
	return A.pull(*terms, **args)


def _process_single(term, A):
	op, val = None, term

	if isinstance(term, (tuple, list)):
		assert len(term) == 2, f'unknown term: {term}'
		op, val = term

	if isinstance(val, (tuple, list)):
		val = A.pull(*val, silent=True)
	elif isinstance(val, str):
		val = A.pull(val, silent=True)
	if op is None or op in {0, 1, 'id', 'i', 'x', ''}:
		return val

	if op in {'minv', '/', 'div'}:
		return 1 / val
	elif op in {'ainv', '-', 'sub'}:
		return -val
	raise Exception(f'unknown op: {op}')

@fig.Component('expr')
def _config_expression(A):  # TODO: boolean ops

	red = A.pull('_reduce', '<>_op', '+', silent=True)

	terms = A.pull('_terms', '<>_term', silent=True)

	if not isinstance(terms, (list, tuple)):
		terms = terms,

	vals = [_process_single(term, A) for term in terms]

	if red in {'+', 'add', 'sum'}:
		out = sum(vals)
	elif red in {'avg', 'average', 'mean'}:
		out = sum(vals) / len(vals)
	elif red in {'*', 'product', 'mul'}:
		out = math.prod(vals)
	elif red in {'%', 'mod'}:
		assert len(vals) == 2, f'bad red: {vals}'
		out = vals[0] % vals[1]
	elif red in {'//', 'idiv'}:
		assert len(vals) == 2, f'bad red: {vals}'
		out = vals[0] // vals[1]
	else:
		raise Exception(f'unknown reduction {red}')

	caste = A.pull('_caste', None, silent=True)

	if caste is None:
		return out
	elif caste == 'int':
		return int(out)
	elif caste == 'str':
		return str(out)
	elif caste == 'float':
		return float(out)
	raise Exception(f'unkonwn caste: {caste}')


@fig.Component('iter/repeat')
class Repeat_Iter(ConfigIter):
	def __init__(self, A):
		num = A.pull('_num', '<>_len')
		unpack = A.pull('_unpack', False)

		element = A['_element']

		auto = A.pull('auto_pull', True)
		include_key = A.pull('include_key', None)

		super().__init__(A, elements=ConfigList(data=num*[*element] if unpack else num*[element], parent=A),
		                 auto_pull=auto, include_key=include_key)


@fig.Component('copy')
def copy_config(A):

	src = A.pull('_src', None)

	mods = A.pull('_src_mod', None)

	if src is not None:
		src = A.pull(src, raw=True)

		A.push('_type', '_x_', silent=True)
		A.push('_src', '_x_', silent=True)
		A.push('_src_mod', '_x_', silent=True)

		updates = A.pull_self(raw=True)

		A.update(src)
		A.update(updates)

		if mods is not None:
			A.push('_mod', mods)

	return A.pull_self()


