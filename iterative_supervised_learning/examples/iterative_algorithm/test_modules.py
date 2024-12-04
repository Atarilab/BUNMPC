import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from simulation import Simulation
import utils
import pinocchio as pin
from database import Database

import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
import random
import hydra
import os
from tqdm import tqdm
from datetime import datetime
import h5py
import pickle
import json
import sys
import time
import wandb

# set random seed for reproducability
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# # login to wandb
# wandb.login()
# project_name = 'test_modules'

# # Initialize WandB with your project name
# wandb.init(project=project_name, config={"learning_rate": 0.001, "batch_size": 32})

# specify safety bounds. If want default, set to None
safety_bounds_dict = {
    'z_height': [0.15, 1.0],
    'body_angle': 25,
    'HAA_L': [-0.5, 1.0],
    'HAA_R': [-1.0, 0.5],
    'HFE_F': [0.0, 1.6],                
    'HFE_B': [-1.6, 0.0],
    'KFE_F': [-2.8, 0.0],
    'KFE_B': [0.0, 2.8]
}
print(safety_bounds_dict)

class LocoSafeDagger():
    def __init__(self,cfg):
        # configuration file (containing the hyper/parameters)
        self.cfg = cfg
            
        # Simulation rollout properties
        self.sim_dt = cfg.sim_dt
        
        # MPC rollout pertubations
        self.mu_base_pos, self.sigma_base_pos = cfg.mu_base_pos, cfg.sigma_base_pos # base position
        self.mu_joint_pos, self.sigma_joint_pos = cfg.mu_joint_pos, cfg.sigma_joint_pos # joint position
        self.mu_base_ori, self.sigma_base_ori = cfg.mu_base_ori, cfg.sigma_base_ori # base orientation
        self.mu_vel, self.sigma_vel = cfg.mu_vel, cfg.sigma_vel # joint velocity
        
        # Model Parameters
        self.action_type = cfg.action_type
        self.normalize_policy_input = cfg.normalize_policy_input
        
        # Warmup
        self.num_rollouts_warmup = cfg.num_rollouts_warmup
        self.num_pertubations_per_replanning_warmup = cfg.num_pertubations_per_replanning_warmup
        self.episode_length_warmup = cfg.episode_length_warmup
        
        # Evaluation
        self.num_rollouts_eval = cfg.num_rollouts_eval
        self.num_pertubations_per_replanning_eval = cfg.num_pertubations_per_replanning_eval
        self.episode_length_eval = cfg.episode_length_eval
        
        # Data Collection
        self.num_iterations_safedagger = cfg.num_iterations_safedagger
        self.num_rollouts_per_iteration_data = cfg.num_rollouts_per_iteration_data
        self.num_replannings_on_nom_traj_data = cfg.num_replannings_on_nom_traj_data
        self.num_pertubations_per_replanning_data = cfg.num_pertubations_per_replanning_data
        self.num_steps_to_block_under_safety = cfg.num_steps_to_block_under_safety
        self.ending_mpc_rollout_episode_length = cfg.ending_mpc_rollout_episode_length
        self.episode_length_data = cfg.episode_length_data
        
        # Desired Motion Parameters
        self.gaits = cfg.gaits
        self.vx_des_min, self.vx_des_max = cfg.vx_des_min, cfg.vx_des_max
        self.vy_des_min, self.vy_des_max = cfg.vy_des_min, cfg.vy_des_max
        self.w_des_min, self.w_des_max = cfg.w_des_min, cfg.w_des_max
        
        # init simulation class
        self.simulation = Simulation(cfg=self.cfg)
        
        # Data related parameters 
        self.n_state = cfg.n_state
        self.n_action = cfg.n_action
        self.goal_horizon = cfg.goal_horizon
        
        self.criterion = nn.L1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Nvidia GPU availability is ' + str(torch.cuda.is_available()))
        
        # Training properties
        self.n_epoch_data = cfg.n_epoch_data
        self.batch_size = cfg.batch_size
        self.n_train_frac = cfg.n_train_frac
        self.learning_rate = cfg.learning_rate
        self.n_epoch_warmup = cfg.n_epoch_warmup
        
        # Initialize Network
        # self.cc_input_size = self.n_state + (self.goal_horizon * 3 * 4)
        self.vc_input_size = self.n_state + 5  # phi, vx, vy, w
        
        self.output_size = self.n_action
        
        # Initialize policy network
        # self.vc_network = self.initialize_network(input_size=self.vc_input_size, output_size=self.output_size, 
        #                                             num_hidden_layer=self.cfg.num_hidden_layer, hidden_dim=self.cfg.hidden_dim,
        #                                             batch_norm=True)
        
        # define log file name
        str_gaits = ''
        for gait in self.gaits:
            str_gaits = str_gaits + gait
        self.str_gaits = str_gaits
        
        current_date = datetime.today().strftime("%b_%d_%Y_")
        current_time = datetime.now().strftime("%H_%M_%S")

        save_path_base = "/safedagger/" + str_gaits
        if cfg.suffix != '':
            save_path_base += "_"+cfg.suffix
        save_path_base += "/" +  current_date + current_time
        
        self.data_save_path = self.cfg.data_save_path + save_path_base
        self.dataset_savepath = self.data_save_path + '/dataset'
        self.network_savepath = self.data_save_path + '/network'
        
        # Declare Database
        self.database = Database(limit=cfg.database_size, goal_type='vc') # goal type and normalize input does not need to be set, as no training is done here

@hydra.main(config_path='cfgs', config_name='safedagger_modified_config')        
def main(cfg):
    icc = LocoSafeDagger(cfg)
    
if __name__ == '__main__':
    main()  