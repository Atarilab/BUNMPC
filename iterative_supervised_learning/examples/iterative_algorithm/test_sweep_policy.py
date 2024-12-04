import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from simulation import Simulation
from contact_planner import ContactPlanner
from utils import get_plan, get_des_velocities, get_estimated_com, \
                    construct_goal, compute_goal_reaching_error, rotate_jacobian
import pinocchio as pin
from database import Database

import numpy as np
# import matplotlib.pyplot as plt
import random
import hydra
import os
from tqdm import tqdm
from datetime import datetime
import h5py
import pickle
import sys
import time
import tkinter as tk
from tkinter.filedialog import asksaveasfilename
import wandb

# set random seet for reproducability
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class TestSweepPolicy():  

    def __init__(self, cfg):        
        # configuration file (containing the hyper/parameters)
        self.cfg = cfg
        
        # Data related parameters 
        self.n_state = cfg.n_state
        self.n_action = cfg.n_action
        self.goal_horizon = cfg.goal_horizon
        
        # Dataset parameters
        self.max_replay_buffer_size = cfg.max_replay_buffer_size
        self.size_dataset_per_training_iter = cfg.size_dataset_per_training_iter
        
        assert self.size_dataset_per_training_iter <= self.max_replay_buffer_size, 'size of dataset per training iter larger than buffer size!'
        
        # Policy Network Properties
        # WATCHOUT: phi, vdes, wdes and gait type is excluded from state!
        self.input_size = (self.n_state) + (self.goal_horizon * 3 * 4)  # goal is goal_horizon * (time + xy) * n_eff.
        self.output_size = self.n_action
        self.network = None
        self.criterion = nn.L1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Nvidia GPU availability is ' + str(torch.cuda.is_available()))
        
        # Training properties
        self.n_train_frac = 0.9
        
        ### Rollout Evaluation ###
        self.episode_length = 10000
        self.sim_dt = cfg.sim_dt
        
        # Model Parameters
        self.action_type = cfg.action_type
        
        # Desired Motion Parameters
        self.gaits = cfg.gaits
        self.vx_des_min, self.vx_des_max = cfg.vx_des_min, cfg.vx_des_max
        self.vy_des_min, self.vy_des_max = cfg.vy_des_min, cfg.vy_des_max
        self.w_des_min, self.w_des_max = cfg.w_des_min, cfg.w_des_max
        
        # init simulation class
        self.simulation = Simulation(cfg=self.cfg)
    
    
    def initialize_network(self, num_hidden_layer=4, hidden_dim=256, batch_norm=False):
        """
        load policy network and determine input and output sizes
        """        
        from networks import GoalConditionedPolicyNet
        
        self.network = GoalConditionedPolicyNet(self.input_size, self.output_size, num_hidden_layer=num_hidden_layer, 
                                                hidden_dim=hidden_dim, batch_norm=batch_norm).to(self.device)
        print("Policy Network initialized")
        
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
    
    def train_network(self, batch_size=256, learning_rate=0.001, n_epoch=150):
        """
        Train and validate the policy network with samples from the current dataset

        Args:
            dataset (Pytorch Dataset): the sampled and splitted dataset for training
            current_iter (int): current simulation step (Not Time!)
            plot_loss (bool, optional): (Non-Blocking) Plot the Training and validation 
            loss. Defaults to False.

        Returns:
            loss_history (np.array): array of training loss
            test_loss_history (np.array): array of validation loss
        """        
        
        # get the training dataset size (use whole dataset)
        train_set_size = len(self.database)
        # train_set_size = min(len(self.database), self.size_dataset_per_training_iter)
        print("Dataset size: " + str(train_set_size))
        

        # define training and test set size
        n_train = int(self.n_train_frac*train_set_size)
        n_test = train_set_size - n_train
        n_batches = n_train/batch_size
        n_batches_test = n_test/batch_size
        
        # Split data into training and validation
        train_data, test_data = torch.utils.data.random_split(self.database, [n_train, n_test])
        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size, shuffle=True)
        
        # define training optimizer
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
            
        tepoch = tqdm(range(n_epoch))
        
        # main training loop
        for epoch in tepoch:
            # set network to training mode
            self.network.train()
            
            # training loss
            training_running_loss = 0
            
            # train network
            for x, y in train_loader:
                self.optimizer.zero_grad()
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_pred = self.network(x)
                loss = self.criterion(y_pred, y)
                
                loss.backward()
                self.optimizer.step()
                training_running_loss += loss.item()
            

            # test network
            test_running_loss = 0
            self.network.eval()
            for z, w in test_loader:
                z, w = z.to(self.device).float(), w.to(self.device).float()
                w_pred = self.network(z)
                test_loss = self.criterion(w_pred, w)
                test_running_loss += test_loss.item()
                
            tepoch.set_postfix({'training loss': training_running_loss/n_batches,
                                'validation loss': test_running_loss/n_batches_test})
            
            # wandb log
            wandb.log({'Training Loss': training_running_loss/n_batches,
                       'Validation Loss': test_running_loss/n_batches_test})  
        
    def run(self, sweep_cfg=None):   
        
        # Initialize policy network
        self.initialize_network(num_hidden_layer=sweep_cfg.num_hidden_layer, hidden_dim=sweep_cfg.hidden_dim,
                                batch_norm=sweep_cfg.batch_norm)
        
        # Declare Database
        print('input normalization: ' + str(sweep_cfg.norm_input))
        self.database = Database(limit=self.cfg.max_replay_buffer_size, norm_input=sweep_cfg.norm_input)
        
        # load saved database
        self.database.load_saved_database(filename='/home/atari_ws/data/goal_cond_iterative_algorithm/trot/Apr_18_2024_07_42_59/dataset/database_0.hdf5')
        
        
        # NOTE: Train Policy
        print('=== Training Policy ===')
        
        # train network
        self.train_network(batch_size=sweep_cfg.batch_size, learning_rate=sweep_cfg.learning_rate, n_epoch=sweep_cfg.epochs)
        
        
        # NOTE: Evaluate Trained Policy
        print('=== Evaluate Policy ===')
        n_eef = len(self.simulation.f_arr)
        
        # init pybullet environment
        self.simulation.init_pybullet_env(display_simu=False)
        
        # randomly decide which gait to simulate
        gait = random.choice(self.gaits)

        # get desired velocities from probability distribution
        v_des, w_des = get_des_velocities(self.vx_des_max, self.vx_des_min, self.vy_des_max, self.vy_des_min, 
                                        self.w_des_max, self.w_des_min, gait, dist='uniform')
        
        # wandb.log({'des_vx': v_des[0], 'des_vy': v_des[1], 'des_w': w_des})
        
        # print selected gait and desired velocities
        print(f"-> gait: {gait} | v_des: {v_des} | w_des: {w_des}")        
                
        
        # NOTE: Create desired contact schedule and desired Goal History for Policy rollout depending on pertubation
        pin_robot, urdf_path = self.simulation.pin_robot, self.simulation.urdf_path
        new_q0, new_v0 = self.simulation.q0, self.simulation.v0
        start_time = 0.0
        
        # Create desired contact schedule with chosen gait and desired velocity
        desired_contact_schedule, _ = self.create_desired_contact_schedule(pin_robot, urdf_path, new_q0, new_v0, v_des, w_des, gait, start_time)

        # Calculate estimated center of mass of robot given the desired velocity
        estimated_com = get_estimated_com(pin_robot, new_q0, new_v0, v_des, self.episode_length + 1, self.sim_dt, get_plan(gait))
        
        # Construct desired goal
        desired_goal = construct_goal(self.episode_length + 1, n_eef, desired_contact_schedule, estimated_com, 
                                        goal_horizon=self.goal_horizon, sim_dt=self.sim_dt)
        
        # for a in range(len(desired_goal)):
        #     for b in range(len(desired_goal[0])):
        #         key = 'desired_goal_' + str(b)
        #         wandb.log({key: desired_goal[a, b]})
                
        # NOTE: Policy Rollout
        # if iterations already exceeded defined warm up iterations, start rolling out policy net

        print("=== Policy Rollout ===")
        policy_state, policy_action, policy_goal, policy_base, _ = self.simulation.rollout_policy(self.episode_length, start_time, v_des, w_des, gait, 
                                                                                        self.network, desired_goal, q0=new_q0, v0=new_v0,
                                                                                        norm_policy_input=self.database.get_database_mean_std())
        print('Policy rollout completed. No. Datapoints: ' + str(len(policy_goal)))
        
        # only log policies which succeed
        if len(policy_goal) > self.episode_length * 2 / 3:
            # log wandb
            for a in range(len(policy_state)):
                for b in range(len(policy_state[0])):
                    key = 'policy_state_' + str(b)
                    wandb.log({key: policy_state[a, b]})
                    
                for b in range(len(policy_action[0])):
                    key = 'policy_action_' + str(b)
                    wandb.log({key: policy_action[a, b]})
                    
                for b in range(len(policy_goal[0])):
                    key = 'policy_goal_' + str(b)
                    wandb.log({key: policy_goal[a, b]})
                    
                for b in range(len(policy_base[0])):
                    key = 'policy_base_' + str(b)
                    wandb.log({key: policy_base[a, b]})
                    
                    
            # NOTE: Compute Goal Reaching Error
            print("=== Computing Goal Reaching Error ===")          

            # Compute Policy Goal reaching error
            if len(policy_goal) != 0:
                print('-> Compute Policy goal reaching error')
                policy_time_error, policy_pos_error = compute_goal_reaching_error(desired_goal, policy_goal, self.goal_horizon, self.sim_dt, n_eef)
                wandb.log({'policy_time_error': policy_time_error, 'policy_pos_error': policy_pos_error})
            else:
                print("-> policy rollout failed")
        
        else:
            print('Policy has too little datapoints. consider as failed')
        
        # Kill pybullet environment
        self.simulation.kill_pybullet_env()
        
        print('run completed')
                
        
@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    wandb.init(project=project_name)
    icc = TestSweepPolicy(cfg)
    icc.run(wandb.config) 
    
    
# NOTE: Sweep settings
# sweep_configuration = {
#     "method": "random",
#     "metric": {"goal": "minimize", "name": "policy_pos_error"},
#     "parameters": {
#         "learning_rate": {'distribution': 'uniform', "max": 0.01, "min": 0.0001},
#         "batch_size": {"values": [64, 128, 256]},
#         "epochs": {"values": [100, 150, 200]},
#         "num_hidden_layer": {"values": [3, 4, 5]},
#         "hidden_dim": {"values": [128, 256, 512]},
#         "batch_norm": {"values": [True, False]},
#         "norm_input": {"values": [True, False]},
#     },
# }

project_name = 'pure_imitiation_learning_vx0_2_trot_solo12'

if __name__ == '__main__':
    main()
    
    # wandb.login()
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    # # sweep_id = 'v6pdrhqe'
    # wandb.agent(sweep_id, function=main, count=50)
    
        


    


