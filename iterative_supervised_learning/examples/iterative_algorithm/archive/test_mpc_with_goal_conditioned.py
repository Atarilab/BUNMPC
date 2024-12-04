from simulation_mpc_gc import SimulationGC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from datetime import datetime
import pybullet
from contact_planner import ContactPlanner
from robot_properties_solo.solo12wrapper import Solo12Config
import pinocchio as pin
from plotting import plot_goal
from utils import quaternion_to_euler_angle, get_plan, construct_goals, get_estimated_com
current_date = datetime.today().strftime("_%b_%d_%Y_")
current_time = datetime.now().strftime("%H_%M_%S")

### Select gait
gait = "trot"

### parameters
episode_length = 3000 #ms
start_time = 0.
v_des = np.array([0.2, 0.0, 0.])
w_des = np.array(0.0)
action_types = ["pd_target"]
goal_horizon = 1
sim_dt = 0.001

# Initialize robot
pin_robot = Solo12Config.buildRobotWrapper()
urdf_path = Solo12Config.urdf_path
q0 = np.array(Solo12Config.initial_configuration)
q0[0:2] = 0.0
v0 = np.zeros(len(q0)-1)
ee = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
pin.forwardKinematics(pin_robot.model, pin_robot.data, q0, np.zeros(pin_robot.model.nv))
pin.updateFramePlacements(pin_robot.model, pin_robot.data)


# Create desired goal aka. contact plan
plan = get_plan(gait)
cp = ContactPlanner(plan)
des_cnt_schedule, des_cnt_plan, des_swing_time = cp.get_contact_schedule(pin_robot, urdf_path, q0, v0, v_des, w_des, episode_length)

estimated_com = get_estimated_com(pin_robot, q0, v0, v_des, episode_length, sim_dt)

end_time = episode_length
for i in range(4):
    max_time = np.sort(des_cnt_schedule[i, :, 0])[-goal_horizon] 
    end_time = int(min(end_time, max_time))

desired_goal_history = np.zeros((end_time, 4*goal_horizon*4))
for i in range(end_time):
    desired_goal_history[i, :] = construct_goals(i, estimated_com[i, :], des_cnt_schedule, 4, goal_horizon)

# FL = desired_goal[:, 0:4]
# plt.plot(FL[:, 1])

### create simulation object
sim = SimulationGC(gait, des_cnt_plan, des_swing_time)

### disturbance specifications
n_directions = 1
min_force = 10
max_force = 50
# sim.f_ext = [0., 0., 0.]
# sim.m_ext = [0., 0., 0.]
# sim.t_dist = [1500., 1600.] #ms

#   

force = np.zeros((n_directions+1, 2))
# for i in range(n_directions):
#     for j in range(10, max_force):
#         sim.f_ext = [j*np.cos(2*i*np.pi/n_directions), j*np.sin(2*i*np.pi/n_directions), 0.]
#         state, measurement, action = sim.rollout(episode_length, start_time, v_des, w_des,\
#          action_types, uneven_terrain = False, use_estimator = True)
#         if state == []:
#             force[i, :] = np.array([2*i*np.pi/n_directions,j-1])
#             print("\n\n failed for push in direction "+str(360*i/n_directions)+" wtih force "+str(j)+"\n\n")
#             break
#         else:
#             print("\n\n didn't fall for push in direction "+str(360*i/n_directions)+" wtih force "+str(j)+"\n\n")


state, measurement, action, goal_history = sim.rollout(episode_length, start_time, v_des, w_des,\
        action_types, save_video = False, uneven_terrain = False, use_estimator = False, add_noise = False, server=pybullet.GUI)
# pybullet.DIRECT

# FL = goal_history[:, 0:4]
# plt.plot(FL[:, 1])
# plt.show()
#plot_goal(goal_history, desired_goal_history)

state = state[:, 4:-1]  # exclude phase %, des v,w and gait type
action = action[action_types[0]]

desired_goal_history = desired_goal_history[:len(goal_history)]

goal_time_error = 0
goal_pos_error = 0
for i in range(len(goal_history)):
    for j in range(4):
        for k in range(goal_horizon):
            goal_time_error += np.linalg.norm([goal_history[i, 4*k*4 + 4*(j + 0)] - desired_goal_history[i, 4*k*4 + 4*(j + 0)]], 1)
            goal_pos_error += np.linalg.norm(goal_history[i, 4*k*4 + 4*(j + 1) : 4*k*4 + 4*(j + 4)] - desired_goal_history[i, 4*k*4 + 4*(j + 1) : 4*k*4 + 4*(j + 4)], 1)

print(goal_time_error)
print(goal_pos_error)

total_goal_error = (goal_time_error + goal_pos_error) / len(goal_history)
print(total_goal_error)



# force = np.load("tests_results/max_force_trot_12_19_18.npy")

### roll pitch yaw
# rpy = np.zeros((len(state[:,0]), 3))
# for i in range(len(state[:,0])):
#     rpy[i,:] = quaternion_to_euler_angle(state[i, 31], state[i, 32], state[i, 33], state[i, 34])

# plt.figure()
# plt.plot(state[:, 4])
# plt.figure()
# plt.plot(state[:, 5])
# plt.figure()
# plt.plot(state[:, 12+18])
# plt.figure()
# plt.plot(rpy[:, 0])
# plt.figure()
# plt.plot(rpy[:, 1])
# plt.figure()
# plt.plot(rpy[:, 2])
# plt.show()
#
# np.save("state_mpc_"+gait+".npy", state)
# np.save("rpy_mpc_"+gait+".npy", rpy)

# force[n_directions, :] = np.array([2*np.pi, force[0,1]])
# # np.save("/home/mkhadiv/my_codes/tests_results/max_force_mpc_"+gait+"_"+current_time+".npy", force)
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(force[:,0], force[:,1])
# ax.set_rmax(max(force[:,1]))
# ax.set_rticks([int(max(force[:,1])/4), int(max(force[:,1])/2),\
#  int(3*max(force[:,1])/4), max(force[:,1])])  # Less radial ticks
# ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
# ax.grid(True)

# ax.set_title("Max external disturbance (N)", va='bottom')
# plt.show()
