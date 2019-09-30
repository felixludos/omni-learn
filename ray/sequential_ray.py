
_store = {}
_ID = 0
class ObjectID(object):
	def __init__(self):
		global _ID
		self.ID = _ID
		_ID += 1
	
	def __hash__(self):
		return self.ID
	
	def __repr__(self):
		return 'ID({})'.format(self.ID)

def init(*args, **kwargs):
	print('Sequential Ray initialized.')

class Remote_Fn(object):
	def __init__(self, fn, num_return_vals=None):
		self.fn = fn
		self.num_return_vals = num_return_vals
	def remote(self, *args, **kwargs):
		args = [get(arg) if isinstance(arg, ObjectID) else arg for arg in args]
		kwargs = {key: (get(value) if isinstance(value, ObjectID) else value) for key, value in kwargs.items()}
		result = self.fn(*args, **kwargs)
		if self.num_return_vals is None or self.num_return_vals == 1:
			ID = ObjectID()
			_store[ID] = result
			return ID
		ids = [ObjectID() for _ in range(self.num_return_vals)]
		for ID, obj in zip(ids, result):
			_store[ID] = obj
		return ids

def remote(num_return_vals=None):
	def remote_num(fn):
		return Remote_Fn(fn, num_return_vals)
	return remote_num

# def remote(fn):
# 	return Remote_Fn(fn)


def get(ID):
	if isinstance(ID, list):
		r = []
		for i in ID:
			r.append(_store[i])
			del _store[i]
		return r
	r = _store[ID]
	del _store[ID]
	return r


def put(obj):
	ID = ObjectID()
	_store[ID] = obj
	return ID
