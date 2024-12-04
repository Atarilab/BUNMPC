from simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from omegaconf import OmegaConf
from contact_planner import ContactPlanner
# matplotlib.use('TkAgg')

# recreate hydra-like config file with the most basic parameters
cfg = OmegaConf.create({
    'gait': 'trot',
    'action_type': "pd_target",
    'sim_dt': 0.001,
    'random_joint_init_enabled': False,
    'meas_vec_len': 30,
    'n_state': 43,
    'n_action': 12,
    'goal_horizon': 1,
    'dt_meas_history': 1,
    'history_size': 1,
    'kp': 2.0,
    'kd': 0.1,
})

def get_end_effector_contact(cnt_index):
    out = []        
    for j in range(len(cnt_index)):
        if cnt_index[j] == 1.:
            out = np.hstack((out, j))
    return out

def get_switches(cnt_plan, i_replan):
    out = []
    pre_contact = get_end_effector_contact(cnt_plan[0, :, 0])
    for i in range(1, len(cnt_plan[:, 0, 0])):
        contact = get_end_effector_contact(cnt_plan[i, :, 0])
        for ee in contact:
            if not ee in pre_contact:
                cnt_time = 50*i_replan + i*0.05/0.001
                cnt = np.hstack((ee, cnt_time))
                cnt = np.hstack((cnt, cnt_plan[i, int(ee), 1:4]))
                cnt[-1] = 1e-3
                if len(out) == 0:
                    out = cnt
                else:
                    out = np.vstack((out, cnt))
        pre_contact = contact
    return out


# simulation parameters
episode_length = 3000  # sim steps
v_des = np.array([0.2, 0.0, 0.0])
w_des = 0.0
start_time = 0.0

### create simulation object
sim = Simulation(cfg=cfg)

# show GUI?
# show_visualization = False
show_visualization = True

# initialize Pybullet environment
sim.init_pybullet_env(display_simu=show_visualization)


# Run mpc simulation, which returns data
state, action, vcgoal, ccgoal, base, frames = sim.rollout_mpc(episode_length, start_time, v_des, w_des, cfg.gait, nominal=True, save_video=True)

# kill pybullet environment
sim.kill_pybullet_env()



