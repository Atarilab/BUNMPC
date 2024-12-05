import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from simulation import Simulation
from contact_planner import ContactPlanner
from utils import get_plan, get_des_velocities, get_estimated_com, \
                    construct_cc_goal, rotate_jacobian
import pinocchio as pin
from database import Database

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import random
import hydra
import os
from tqdm import tqdm
from datetime import datetime
import h5py
import pickle
import sys
import time
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

# set random seet for reproducability
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class TestTrainedPolicy():  

    def __init__(self, cfg):        
        # configuration file (containing the hyper/parameters)
        self.cfg = cfg
            
        # Simulation rollout properties
        self.episode_length = 10000
        self.sim_dt = cfg.sim_dt
        
        # MPC rollout pertubations
        self.mu_base_pos, self.sigma_base_pos = cfg.mu_base_pos, cfg.sigma_base_pos # base position
        self.mu_joint_pos, self.sigma_joint_pos = cfg.mu_joint_pos, cfg.sigma_joint_pos # joint position
        self.mu_base_ori, self.sigma_base_ori = cfg.mu_base_ori, cfg.sigma_base_ori # base orientation
        self.mu_vel, self.sigma_vel = cfg.mu_vel, cfg.sigma_vel # joint velocity
        
        # Model Parameters
        self.action_type = cfg.action_type
        
        # Data related parameters 
        self.n_state = cfg.n_state
        self.n_action = cfg.n_action
        self.history_size = cfg.history_size
        self.goal_horizon = cfg.goal_horizon
        
        # Desired Motion Parameters
        self.gaits = cfg.gaits
        self.vx_des_min, self.vx_des_max = cfg.vx_des_min, cfg.vx_des_max
        self.vy_des_min, self.vy_des_max = cfg.vy_des_min, cfg.vy_des_max
        self.w_des_min, self.w_des_max = cfg.w_des_min, cfg.w_des_max
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # init simulation class
        self.simulation = Simulation(cfg=self.cfg)
        
        self.num_pertubations_per_replanning = cfg.num_pertubations_per_replanning
        
    def load_network(self, filename=None):
        """
        load policy network and determine input and output sizes
        """    
        if filename is None: 
            Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
            filename = askopenfilename(initialdir=self.cfg.data_save_path) # show an "Open" dialog box and return the path to the selected file
        if len(filename) == 0:
            raise FileNotFoundError()
        file = torch.load(filename, map_location=self.device)
        self.network = file['network'].to(self.device)
        self.network.eval()
        self.policy_input_parameters = file['norm_policy_input']

        print("Policy Network loaded from: " + filename)
        if self.policy_input_parameters is None:
            print('Policy Input will NOT be normalized')
        else:
            print('Policy Input will be normalized')
    
    
    def create_desired_contact_schedule(self, pin_robot, urdf_path, q0, v0, v_des, w_des, gait, start_time):
        """
        create contact schedule for a desired robot velocity and gait.

        Args:
            pin_robot (robot): pinocchio robot model
            urdf_path (path): robot urdf path
            q0 (np.array): robot initial configuration
            v0 (np.array): robot initial velocity
            v_des (np.array): desired translational velocity of robot com
            w_des (float): desired yaw velocity of robot
            gait (str, optional): gait to simulate. Defaults to None.

        Returns:
            contact_schedule (np.array): [n_eff x number of contact events x (time, x, y, z)]
            cnt_plan (np.array): [planning horizon x n_eff x (in contact?, x, y, z)]
        """        

        plan = get_plan(gait)
        cp = ContactPlanner(plan)
        contact_schedule, cnt_plan = cp.get_contact_schedule(pin_robot, urdf_path, q0, v0, v_des, w_des, self.episode_length, start_time)
        return contact_schedule, cnt_plan
        
        
    def run(self):   
        '''
        Run the iterative algorithm
        '''

        # Initialize policy network
        self.load_network()
        
        # Initialize Robot model
        n_eef = len(self.simulation.f_arr)
        
        # Activate matplotlib interactive plot for non-blocking plotting
        plt.ion()
        # self.fig_goal_error, self.ax_goal_error = plt.subplots()
        self.fig_goal, self.ax_goal = plt.subplots(4, 3)
        self.fig_action, self.ax_action = plt.subplots(4, 3)
        self.fig_vel, self.ax_vel = plt.subplots(1, 1)
        plt.show()
    
            
        # condition on which iterations to show GUI for Pybullet    
        # display_simu = False
        display_simu = True
        
        # init env for if no pybullet server is active
        if self.simulation.currently_displaying_gui is None:
            self.simulation.init_pybullet_env(display_simu=display_simu)
        # change pybullet environment between with/without display, depending on condition
        elif display_simu != self.simulation.currently_displaying_gui:
            self.simulation.kill_pybullet_env()
            self.simulation.init_pybullet_env(display_simu=display_simu)
            
            
        # NOTE: Sampling of Velocity and Gait (currently not needed but still implemented)
        # randomly decide which gait to simulate
        gait = random.choice(self.gaits)

        # get desired velocities from probability distribution
        v_des, w_des = get_des_velocities(self.vx_des_max, self.vx_des_min, self.vy_des_max, self.vy_des_min, 
                                        self.w_des_max, self.w_des_min, gait, dist='uniform')
        
        # print selected gait and desired velocities
        print(f"-> gait: {gait} | v_des: {v_des} | w_des: {w_des}")
        
        
        # NOTE: Rollout benchmark MPC
        # set simulation start time to 0.0
        start_time = 0.0
        
        print("=== Benchmark MPC Rollout ===")
        
        # rollout mpc
        mpc_state, mpc_action, mpc_goal, mpc_base = \
            self.simulation.rollout_mpc(self.episode_length, start_time, v_des, w_des, gait, nominal=True)
        
        # collect position and velocity of nominal trajectory
        nominal_pos, nominal_vel = self.simulation.q_nominal, self.simulation.v_nominal
        
        # get contact plan of benchmark mpc
        contact_plan = self.simulation.gg.cnt_plan
                
                
        # NOTE: Create desired contact schedule and desired Goal History for Policy rollout depending on pertubation
        pin_robot, urdf_path = self.simulation.pin_robot, self.simulation.urdf_path
        
        new_q0, new_v0 = self.simulation.q0, self.simulation.v0
        
        # Create desired contact schedule with chosen gait and desired velocity
        desired_contact_schedule, _ = self.create_desired_contact_schedule(pin_robot, urdf_path, new_q0, new_v0, v_des, w_des, gait, start_time)

        # Calculate estimated center of mass of robot given the desired velocity
        estimated_com = get_estimated_com(pin_robot, new_q0, new_v0, v_des, self.episode_length + 1, self.sim_dt, get_plan(gait))
        
        # Construct desired goal
        desired_goal = construct_cc_goal(self.episode_length + 1, n_eef, desired_contact_schedule, estimated_com, 
                                        goal_horizon=self.goal_horizon, sim_dt=self.sim_dt)
            
    
        # NOTE: Policy Rollout
        # if iterations already exceeded defined warm up iterations, start rolling out policy net

        print("=== Policy Rollout ===")
        policy_state, policy_action, policy_goal, _, commanded_goal = self.simulation.rollout_policy(self.episode_length, start_time, v_des, w_des, gait, 
                                                                                        self.network, desired_goal, q0=new_q0, v0=new_v0,
                                                                                        norm_policy_input=self.policy_input_parameters)
        print('Policy rollout completed. No. Datapoints: ' + str(len(policy_goal)))

        

        # NOTE: Compute Goal Reaching Error
        print("=== Computing Goal Reaching Errors ===")
        
        # Handle if both mpc and policy failed   
        if len(mpc_state) == 0 and len(policy_state) == 0:
            print("Both MPC and Policy failed. Continuing to next iteration...")
        
        # Plot desired goal
        for y in range(n_eef):
            for z in range(3):
                self.ax_goal[y, z].clear()
                self.ax_goal[y, z].plot(desired_goal[:, 3*y+z], label='des')
                self.ax_goal[y, z].grid()
                
        self.ax_goal[0, 0].set_title('time')
        self.ax_goal[0, 1].set_title('x')
        self.ax_goal[0, 2].set_title('y')
        # self.ax_goal[0, 3].set_title('z')
        self.fig_goal.canvas.draw()
        self.fig_goal.canvas.flush_events()
        
        # Compute MPC goal reaching error
        mpc_goal_reaching_error = np.nan
        if len(mpc_goal) != 0:
            print('-> Compute MPC goal reaching error')
            # mpc_goal_reaching_error = compute_goal_reaching_error(desired_goal, mpc_goal, self.goal_horizon, self.sim_dt, n_eef)
            
            # Plot mpc goal and action
            for y in range(4):
                for z in range(3):
                    self.ax_goal[y, z].plot(mpc_goal[:, 3*y+z], label='mpc')
                    self.ax_goal[y, z].grid()
                    self.ax_action[y, z].clear()
                    self.ax_action[y, z].plot(mpc_action[:, 3*y+z], label='mpc')
                    self.ax_action[y,z].grid()
            self.fig_goal.canvas.draw()
            self.fig_goal.canvas.flush_events()
            self.fig_action.canvas.draw()
            self.fig_action.canvas.flush_events()
        else:
            print("-> mpc rollout failed")                

        # Compute Policy Goal reaching error
        policy_goal_reaching_error = np.nan
        if len(policy_goal) != 0:
            print('-> Compute Policy goal reaching error')
            policy_goal_reaching_error = compute_goal_reaching_error(desired_goal, policy_goal, self.goal_horizon, self.sim_dt, n_eef)
            
            # plot policy goal and action
            for y in range(4):
                for z in range(3):
                    self.ax_goal[y, z].plot(policy_goal[:, 3*y+z], label='policy')
                    self.ax_goal[y, z].grid()
                    self.ax_goal[y, z].legend()
                    self.ax_action[y, z].plot(policy_action[:, 3*y+z], label='policy')
                    self.ax_action[y, z].grid()
                    self.ax_action[y, z].legend()
            self.fig_goal.canvas.draw()
            self.fig_goal.canvas.flush_events()
            self.fig_action.canvas.draw()
            self.fig_action.canvas.flush_events()
        else:
            print("-> policy rollout failed")
        
        
        # Velocity Tracking
        self.ax_vel.axhline(v_des[0], label='desired')
        self.ax_vel.plot(mpc_state[:,0], label='mpc')
        self.ax_vel.plot(policy_state[:,0], label='policy')
        self.ax_vel.grid()
        self.ax_vel.legend()
        self.fig_vel.canvas.draw()
        self.fig_vel.canvas.flush_events()
        
        
        plt.show(block=True)
                
        
@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    icc = TestTrainedPolicy(cfg) 
    icc.run() 

if __name__ == '__main__':
    main()
    
        


    


