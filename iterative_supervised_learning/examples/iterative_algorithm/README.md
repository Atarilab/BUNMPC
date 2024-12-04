# Safe Learning of Locomotion Skills from MPC - Experiments

## Variables
Here is a list of common variables we will deal with in the scripts.
| Variables | Comment | Structure |
| --- | --- | --- |
| pd_target robot action | Output of NN. Target robot joint positions | array of 12 values for each leg joint |
| robot state | partial input of NN |[robot velocity v + n_eff*(x,y position of eef to robot base) + robot configuration q without robot base com x,y] = [18 + 8 + 17] = array of 43 values |
| vc goal | velocity conditioned goal | [gait phase + vx + vy + w + gait type] = array of 5 values |
| cc goal | contact conditioned goal | [goal_horizon * n_eff * (time to next contact + base position to next contact)] = [? * 4 * 4]|
| base_history | robot base absolute position | [episode_length, 3] |
| contact schedule | list of when and where each eef makes contact with the ground | [n_eff x number of contact events x (time, x, y, z)] |
| cnt_plan | contact plan. list of position of each eef for all time steps and whether if they are in contact with the ground | [planning horizon x n_eff x (in contact?, x, y, z)] |

## Files overview
### Behavior Cloning
|  | Script | Config | Comment |
| --- | --- | --- | --- |
| **Data Collection**| [data_collection.py](./data_collection.py) | [data_collection_config.yaml](./cfgs/data_collection_config.yaml) | Generate data for behavior cloning |

### Helper scripts
| Script | Comment |
| --- | --- |
| [contact_planner.py](./contact_planner.py) | Plan contact locations with Raibert contact planner |
| [database.py](./database.py) | Database class |
| [networks.py](./networks.py) | Policy network class |
| [simulation.py](./simulation.py) | Simulation class. Contains all simulation functions to rollout robot |
| [utils.py](./utils.py) | Other useful functions |

## Useful Files
pybullet functions can be found in [../envs/pybullet_env.py](../envs/pybullet_env.py)

solo12 gait parameters can be found in [../motions/cyclic](../motions/cyclic)

BiConMP cyclic functions can be found in [../mpc/abstract_cyclic_gen.py](../mpc/abstract_cyclic_gen.py)

[back to main page](../../../README.md)