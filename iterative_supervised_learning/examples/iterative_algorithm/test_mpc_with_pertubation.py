from simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pinocchio as pin
from utils import rotate_jacobian
from omegaconf import OmegaConf
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

# simulation parameters
episode_length = 3000  # sim steps = 6*0.5/0.001
v_des = [0.2, 0.0, 0.0]
w_des = 0.0
start_time = 0.0

### perturbation mean and variance
mu_base_pos, sigma_base_pos = 0, 0.1 # base position
mu_joint_pos, sigma_joint_pos = 0, 0.2 # joint position
mu_base_ori, sigma_base_ori = 0, 0.7 # base orientation
mu_vel, sigma_vel = 0, 0.2 # joint velocity

### create simulation object
sim = Simulation(cfg=cfg)

# show GUI?
show_visualization = False

# initialize Pybullet environment
sim.init_pybullet_env(display_simu=show_visualization)


# Run mpc simulation, which returns data
state, measurement, action, goal, base = sim.rollout_mpc(episode_length, start_time, v_des, w_des, cfg.gait, nominal=True)

nominal_pos, nominal_vel = sim.q_nominal, sim.v_nominal

num_replanning = int(sim.gait_params.gait_period/sim.plan_freq)

contact_plan = sim.gg.cnt_plan

num_random_initial = 1

# try replanning
for j in range(num_replanning):
    start_time = j * sim.plan_freq
    new_q0 = nominal_pos[int(start_time/sim.plan_freq)]
    new_v0 = nominal_vel[int(start_time/sim.plan_freq)]
    sim.pin_robot.computeJointJacobians(new_q0)
    sim.pin_robot.framesForwardKinematics(new_q0)

    ### find end-effectors in contact
    ee_in_contact = []
    for ee in range(len(sim.gg.eff_names)):
        if contact_plan[j][ee][0] == 1:
            ee_in_contact.append(sim.gg.eff_names[ee])
    cnt_jac = np.zeros((3*len(ee_in_contact), len(new_v0)))
    cnt_jac_dot = np.zeros((3*len(ee_in_contact), len(new_v0)))

    ### compute Jacobian of end-effectors in contact and its derivative
    for ee_cnt in range(len(ee_in_contact)):
        jac = pin.getFrameJacobian(sim.pin_robot.model,\
            sim.pin_robot.data,\
            sim.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]),\
            pin.ReferenceFrame.LOCAL)
        cnt_jac[3*ee_cnt:3*(ee_cnt+1),] = rotate_jacobian(sim, jac,\
            sim.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]))[0:3,]
        jac_dot = pin.getFrameJacobianTimeVariation(sim.pin_robot.model,\
            sim.pin_robot.data,\
            sim.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]),\
            pin.ReferenceFrame.LOCAL)
        cnt_jac_dot[3*ee_cnt:3*(ee_cnt+1),] = rotate_jacobian(sim, jac_dot,\
            sim.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]))[0:3,]

    ### perturb around the state at current time
    for k in range(num_random_initial):
        min_ee_height = .0
        while min_ee_height >= 0:
            ### constraint-consistent perturbation
            perturbation_pos = np.concatenate((np.random.normal(mu_base_pos, sigma_base_pos, 3),\
                            np.random.normal(mu_base_ori, sigma_base_ori, 3), \
                            np.random.normal(mu_joint_pos, sigma_joint_pos, len(new_v0)-6)))
            perturbation_vel = np.random.normal(mu_vel, sigma_vel, len(new_v0))
            if ee_in_contact == []:
                random_pos_vec = perturbation_pos
                random_vel_vec = perturbation_vel
            else:
                random_pos_vec = (np.identity(len(new_v0)) - np.linalg.pinv(cnt_jac)@\
                            cnt_jac) @ perturbation_pos
                jac_vel = cnt_jac_dot * perturbation_pos + cnt_jac * perturbation_vel
                random_vel_vec = (np.identity(len(new_v0)) - np.linalg.pinv(jac_vel)@\
                            jac_vel) @ perturbation_pos

            ### add perturbation to nominal trajectory
            new_v0 = nominal_vel[int(start_time/sim.plan_freq)] + random_vel_vec
            new_q0 = pin.integrate(sim.pin_robot.model, \
                nominal_pos[int(start_time/sim.plan_freq)], random_pos_vec)

            ### check if the swing foot is below the ground
            sim.pin_robot.framesForwardKinematics(new_q0)
            ee_below_ground = []
            for e in range(len(sim.gg.eff_names)):
                frame_id = sim.pin_robot.model.getFrameId(sim.gg.eff_names[e])
                if sim.pin_robot.data.oMf[frame_id].translation[2] < 0.:
                    ee_below_ground.append(sim.gg.eff_names[e])
            if ee_below_ground == []:
                min_ee_height = -1.

        ### perform rollout
        print("====== rollout with pertubation - nominal traj. position ", str(j), " number ", str(k), " =========")
        state, measurement, action, goal, base, frames = sim.rollout_mpc(episode_length, start_time, v_des, w_des, cfg.gait, 
                                                                 nominal=True, q0=new_q0, v0=new_v0, save_video=True)

# kill pybullet environment
sim.kill_pybullet_env()
