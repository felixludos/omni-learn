# Foundation - RL and beyond

This repo contains a framework for deep RL using primarily pytorch (>=0.4).

## Installation

Run `pip install .` while in this dir to install package.

## Execution

Run `python main_pg.py -c config/test-npg.yaml --name test` (optionally with -v flag) to test policy gradient training.

To visualize an example trained policy, run `python main_viz.py example_results/npg -n 2`.

## Todo

### Short Term

- GPU support
- No old model when using NPG
- Merge gaussian/cat policies - just use Normal_MultiCat_Policy

### Long Term

(Roughly by priority)

- Multi Agent Control Envs
  - Mass Balance
  - Walking
  - Electric Trap
- Multi Agent Managers - training regimes, including commanders
- Utility algs - mixture of experts, genetic, search, line search for npg, proper GAE
- Multi Agent Envs - tracking, foraging, toy, assembly line, tracking, formations
- Doc - proper doc and clean code
- Integrate notes
- Interactive scripts - jupyter
- Utility tests - systematic testing framework
- Integrate particle envs
- Q Learning
  - DDPG, DQN, A3C
  - Tabular
- Adv pg - TRPO, PPO
- Exploration modules
- Imitation Learning
- Trajectory Optim/Guided Policy Search
- Transition models - including E2C
- MCTS
- control - MPPI, iLQG
- proper optim framework - GN, Newton
- Representation Learning - conv nets, AEs, VAEs, GANs
- Continuous time - diff eq solvers, integrators
- Simulation - MD, MC, psi4, integrate Ion framework
- Stat Mech - cellular automata, phase behavior
- Physics engines - Mujoco, pymunk
- 3D Visualization - vpython
- Julia? - flux