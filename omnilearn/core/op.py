from .imports import *
import omnifig as fig

from .datasets import DatasetBase
from .models import MLP as MLPBase, ModelBase
from .optimizers import Adam as AdamBase, SGD as SGDBase

# configurable versions of the top level functions



class Dataset(fig.Configurable, DatasetBase):
    pass



class Model(fig.Configurable, ModelBase):
    pass



class MLP(fig.Configurable, MLPBase):
    def __init__(self, hidden: Optional[Iterable[int]] = None, *,
                 nonlin: str = 'elu', output_nonlin: Optional[str] = None,
                 input_dim: Optional[int] = None, output_dim: Optional[int] = None,
                 **kwargs):
        super().__init__(hidden=hidden, nonlin=nonlin, output_nonlin=output_nonlin, input_dim=input_dim, output_dim=output_dim, **kwargs)



class SGD(fig.Configurable, SGDBase):
    def __init__(self, lr: float, momentum: float = 0., dampening: float = 0., 
                 weight_decay: float = 0., nesterov: bool = False, **kwargs):
        super().__init__(lr=lr, momentum=momentum, dampening=dampening, 
                         weight_decay=weight_decay, nesterov=nesterov, **kwargs)



class Adam(fig.Configurable, AdamBase):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0., amsgrad: bool = False, 
                 **kwargs):
        super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, 
                         amsgrad=amsgrad, **kwargs)








