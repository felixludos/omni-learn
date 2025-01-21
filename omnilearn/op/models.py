from .imports import *
from .common import Machine

from ..compute import Model as _Model, MLP as _MLP, SGD as _SGD, Adam as _Adam, Linear as _Linear



class Model(Machine, _Model):
	pass



class MLP(Model, _MLP):
	def __init__(self, hidden: Optional[Iterable[int]] = None, *,
				 nonlin: str = 'elu', output_nonlin: Optional[str] = None,
				 norm: Optional[str] = None, dropout: Optional[float] = None,
				 output_norm: Optional[str] = None, output_dropout: Optional[float] = None,
				 input_dim: Optional[int] = None, output_dim: Optional[int] = None,
				 **kwargs):
		super().__init__(hidden=hidden, nonlin=nonlin, output_nonlin=output_nonlin,
				   norm=norm, dropout=dropout, output_norm=output_norm, output_dropout=output_dropout,
						 input_dim=input_dim, output_dim=output_dim,
						 **kwargs)


class Linear(Model, _Linear):
	def __init__(self, in_features: int = None, out_features: int = None, *, bias: bool = True, **kwargs):
		super().__init__(in_features=in_features, out_features=out_features, bias=bias, **kwargs)


class SGD(Machine, _SGD):
	def __init__(self, lr: float, momentum: float = 0., dampening: float = 0.,
				 weight_decay: float = 0., nesterov: bool = False, 
				 objective: str = 'loss', maximize: bool = False, **kwargs):
		super().__init__(lr=lr, momentum=momentum, dampening=dampening,
						 weight_decay=weight_decay, nesterov=nesterov,
						  objective=objective, maximize=maximize, **kwargs)



class Adam(Machine, _Adam):
	def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
				 eps: float = 1e-8, weight_decay: float = 0., amsgrad: bool = False, 
				 objective: str = 'loss', maximize: bool = False, 
				 **kwargs):
		super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay,
						 amsgrad=amsgrad,
						  objective=objective, maximize=maximize, **kwargs)


