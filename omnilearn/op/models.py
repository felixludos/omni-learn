from .imports import *
from .common import Machine

from ..compute import Model as _Model, MLP as _MLP, SGD as _SGD, Adam as _Adam



class Model(Machine, _Model):
	pass



class MLP(Model, _MLP):
	def __init__(self, hidden: Optional[Iterable[int]] = None, *,
				 nonlin: str = 'elu', output_nonlin: Optional[str] = None,
				 #input_dim: Optional[int] = None, output_dim: Optional[int] = None,
				 **kwargs):
		super().__init__(hidden=hidden, nonlin=nonlin, output_nonlin=output_nonlin,
						 # input_dim=input_dim, output_dim=output_dim,
						 **kwargs)



class SGD(Machine, _SGD):
	def __init__(self, lr: float, momentum: float = 0., dampening: float = 0.,
				 weight_decay: float = 0., nesterov: bool = False, **kwargs):
		super().__init__(lr=lr, momentum=momentum, dampening=dampening,
						 weight_decay=weight_decay, nesterov=nesterov, **kwargs)



class Adam(Machine, _Adam):
	def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
				 eps: float = 1e-8, weight_decay: float = 0., amsgrad: bool = False,
				 **kwargs):
		super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay,
						 amsgrad=amsgrad, **kwargs)


