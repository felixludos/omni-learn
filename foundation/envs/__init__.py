#from .general import *
#from .test_env import Sum_Env

from .wrappers import *

# from gym.envs.registration import register

from .test_env import Sum_Env, Sel_Env
from .multi_agent.control import Walking
from .arm import Arm
from .MNIST_envs import MNIST_Walker


# register(
#     id='DRL_PointMass-v0',
#     entry_point='foundation.envs.point_mass:PointMassEnv',
#     max_episode_steps=25,
# )
#
# register(
#     id='DRL_Swimmer-v0',
#     entry_point='foundation.envs.swimmer:SwimmerEnv',
#     max_episode_steps=500,
# )
#
# register(
#     id='DRL_HalfCheetah-v0',
#     entry_point='foundation.envs.half_cheetah:HalfCheetahEnv',
#     max_episode_steps=500,
# )
#
# register(
#     id='DRL_Ant-v0',
#     entry_point='foundation.envs.ant:AntEnv',
#     max_episode_steps=500,
# )
#
# register(
#     id='SimpleAnt-v0',
#     entry_point='foundation.envs.ant:SimpleAntEnv',
#     max_episode_steps=500,
# )
#
# register(
#     id='Cartpole-v0',
#     entry_point='foundation.envs:Mujoco_Pendulum_Env',
#     max_episode_steps=500,
# )