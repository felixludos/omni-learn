
import numpy as np
import torch

# TODO: multi dim objects (3D rotations/quaternions)
# TODO: connect kinematics: x -> v -> a
# TODO: units
# TODO: allow different representations - eg. theta -> sin/cos
# TODO: automatically format states in sim._observe

class Dim_Description(object):
	def __init__(self, name, **props):
		self.__dict__[name] = name
		self.__dict__['properties'] = props

	def sample(self, N=None):
		raise NotImplementedError

	def __getattr__(self, item):
		if item in self.properties:
			return self.properties[item]
		return super().__getattribute__(item)
	def __setattr__(self, key, value):
		if 'properties' in self.__dict__ and key == 'properties':
			raise Exception('{} is a reserved name and cannot be set'.format('properties'))
		if key in self.properties:
			self.properties[key] = value
			return
		return super().__setattr__(key, value)
	def __delattr__(self, item):
		assert item != 'properties', 'cannot be deleted'
		if item in self.properties:
			del self.properties[item]
		return super().__delattr__(item)

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		return '{}[{}]({})'.format(self.__class__.__name__, self.name, ', '.join('{}={}'.format(k,v) for k,v in self.properties.items() if k[0] != '_'))


class Typed_Dim(Dim_Description):
	def __init__(self, type, **props):
		props['type'] = type
		super().__init__(**props)

	def get_type(self):
		return self.type

class Velocity_Dim(Typed_Dim):
	def __init__(self, scale=1., **props):
		if 'type' not in props:
			props['type'] = 'vel'
		super().__init__(**props)
		self._scale = scale

	def sample(self, N=None):
		shape = (1,) if N is None else (N, 1)
		return self._scale * torch.randn(*shape)

class Ranged_Dim(Typed_Dim):
	def __init__(self, min, max, **props):
		props['min'] = min
		props['max'] = max
		if 'type' not in props:
			props['type'] = 'ranged'
		super().__init__(**props)

	def sample(self, N=None):
		shape = (1,) if N is None else (N,1)
		return torch.rand(*shape) * (self.max - self.min) + self.min

class Periodic_Dim(Ranged_Dim):
	def __init__(self, period, **props):
		props['period'] = period
		props['max'] = period
		props['min'] = 0.
		if 'type' not in props:
			props['type'] = 'periodic'
		super().__init__(**props)

	def get_period(self):
		return self.period

class Angle_Dim(Periodic_Dim):
	def __init__(self, **props):
		props['period'] = 2*np.pi
		props['type'] = 'angle'
		super().__init__(**props)

class Discrete_Dim(Typed_Dim):
	def __init__(self, cardinality, **props):
		props['cardinality'] = cardinality
		if 'type' not in props:
			props['type'] = 'discrete'
		super().__init__(**props)

	def sample(self, N=None):
		return torch.randint(0, self.cardinality, size=N)
