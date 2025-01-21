import omnifig as fig

from .op import Mechanism
fig.component('mechanism')(Mechanism)


# trainer
from .op import Trainer, Planner, Reporter, Checkpointer, WandB_Monitor, Evaluator
fig.component('trainer')(Trainer)
fig.component('planner')(Planner)
fig.component('reporter')(Reporter)
fig.component('checkpointer')(Checkpointer)
fig.component('wandb')(WandB_Monitor)
fig.component('evaluator')(Evaluator)



# models
from .op import MLP, Linear
fig.component('mlp')(MLP)
fig.component('linear')(Linear)



# optimizers
from .op import Adam, SGD
fig.component('adam')(Adam)
fig.component('sgd')(SGD)


