import omnifig as fig



# trainer
from .op import Trainer
fig.component('trainer')(Trainer)



# models
from .op import MLP
fig.component('mlp')(MLP)



# optimizers
from .op import Adam, SGD
fig.component('adam')(Adam)
fig.component('sgd')(SGD)


