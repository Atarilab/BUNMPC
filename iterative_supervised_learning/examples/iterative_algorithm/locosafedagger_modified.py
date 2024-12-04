# This file is to implement Majid's algorithm on the HUAWEI-MIRMI Safeman proposal based on the locosafedagger work from Xun Pua's master thesis time.

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

# login to wandb
wandb.login()
project_name = 'locosafedagger_modified'

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
        self.vc_network = self.initialize_network(input_size=self.vc_input_size, output_size=self.output_size, 
                                                    num_hidden_layer=self.cfg.num_hidden_layer, hidden_dim=self.cfg.hidden_dim,
                                                    batch_norm=True)
        
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
    
    def initialize_network(self, input_size=0, output_size=0, num_hidden_layer=3, hidden_dim=512, batch_norm=True):
        """initilize policy network

        Args:
            input_size (int, optional): input dimension of network (state + goal). Defaults to 0.
            output_size (int, optional): output dimention of network (action). Defaults to 0.
            num_hidden_layer (int, optional): number of hidden layers. Defaults to 3.
            hidden_dim (int, optional): number of nodes per hidden layer. Defaults to 512.
            batch_norm (bool, optional): if 1D batch normalization should be performed. Defaults to True.

        Returns:
            network: returns initialized pytorch network
        """              
        
        from networks import GoalConditionedPolicyNet
        
        network = GoalConditionedPolicyNet(input_size, output_size, num_hidden_layer=num_hidden_layer, 
                                                hidden_dim=hidden_dim, batch_norm=batch_norm).to(self.device)
        print("Policy Network initialized")
        return network
    
    def train_network(self, network, batch_size=256, learning_rate=0.002, n_epoch=150):
        """Train and validate the policy network with samples from the current dataset

        Args:
            network (_type_): policy network to train
            batch_size (int, optional): batch size. Defaults to 256.
            learning_rate (float, optional): learning rate. Defaults to 0.002.
            n_epoch (int, optional): number of epochs to train. Defaults to 150.

        Returns:
            network: trained network
        """             
        
        # get the training dataset size (use whole dataset)
        database_size = len(self.database)

        print("Dataset size: " + str(database_size))
        print(f'Batch size: {batch_size}')
        print(f'learning rate: {learning_rate}')
        print(f'num of epochs: {n_epoch}')

        # define training and test set size
        n_train = int(self.n_train_frac*database_size)
        n_test = database_size - n_train
        
        print(f'training data size: {n_train}')
        print(f'validation data size: {n_test}')
        
        # Split data into training and validation
        train_data, test_data = torch.utils.data.random_split(self.database, [n_train, n_test])
        train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size, shuffle=True, drop_last=True)
        
        # define training optimizer
        
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
            
        tepoch = tqdm(range(n_epoch))
        
        # main training loop
        for epoch in tepoch:
            # set network to training mode
            network.train()
            
            # training loss
            train_loss, valid_loss = [], []
            
            # train network
            for x, y in train_loader:
                self.optimizer.zero_grad()
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_pred = network(x)
                loss = self.criterion(y_pred, y)
                
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            

            # test network
            test_running_loss = 0
            network.eval()
            for z, w in test_loader:
                z, w = z.to(self.device).float(), w.to(self.device).float()
                w_pred = network(z)
                test_loss = self.criterion(w_pred, w)
                valid_loss.append(test_loss.item())
                
            train_loss_avg = np.mean(train_loss)
            valid_loss_avg = np.mean(valid_loss)
            tepoch.set_postfix({'training loss': train_loss_avg,
                                'validation loss': valid_loss_avg})
            
            # wandb log
            wandb.log({'Training Loss': train_loss_avg,
                       'Validation Loss': valid_loss_avg})
            
        return network
    
    def save_network(self, network, name='policy'):
        """save trained network

        Args:
            network (_type_): network to save
            name (str, optional): name of network to save. Defaults to 'policy'.
        """        
        os.makedirs(self.network_savepath, exist_ok=True)
        
        savepath =  self.network_savepath + "/"+name+".pth"
        
        payload = {'network': network,
                #    'optimizer': self.optimizer,
                #    'scheduler': self.scheduler,
                   'norm_policy_input': None}
        
        if self.normalize_policy_input:
            payload['norm_policy_input'] = self.database.get_database_mean_std()
        
        torch.save(payload, savepath)
        print('Network Snapshot saved')
    
    def save_dataset(self, iter):
        """save current database

        Args:
            iter (_type_): database iteration (just a number)
        """
           
        print("saving dataset for iteration " + str(iter))
        
        # make directory
        os.makedirs(self.dataset_savepath, exist_ok=True)
        
        # Get data len from dataset class
        data_len = len(self.database)
        
        # save numpy datasets
        with h5py.File(self.dataset_savepath + "/database_" + str(iter) + ".hdf5", 'w') as hf:
            hf.create_dataset('states', data=self.database.states[:data_len])
            hf.create_dataset('vc_goals', data=self.database.vc_goals[:data_len])
            # hf.create_dataset('cc_goals', data=self.database.cc_goals[:data_len])
            hf.create_dataset('actions', data=self.database.actions[:data_len]) 
        
        # save config as pickle only once
        if os.path.exists(self.dataset_savepath + "/config.json") == False:
            # convert hydra cfg to config dict
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)
            
            f = open(self.dataset_savepath + "/config.json", "w")
            json.dump(config_dict, f, indent=4)
            f.close()
        
        print("saved dataset at iteration " + str(iter))
    
    def warmup(self):
        pass
    
    def _construct_desired_goal(self, v_des, w_des, gait):
        """Construct a desired goal array."""
        desired_goal = np.zeros((self.episode_length_eval, 5))
        for t in range(self.episode_length_eval):
            desired_goal[t, 0] = utils.get_phase_percentage(t, self.sim_dt, gait)
        desired_goal[:, 1] = v_des[0]
        desired_goal[:, 2] = v_des[1]
        desired_goal[:, 3] = w_des
        desired_goal[:, 4] = utils.get_vc_gait_value(gait)
        return desired_goal

    def run(self):
        """Run the modified locosafedagger algorithm
        """
        for i in range(self.num_iterations_locosafedagger):
            print(f"============ Iteration {i+1} ==============")

            # sample goals
            gait = random.choice(self.gaits)
            v_des,w_des = utils.get_des_velocities(
                self.vx_des_max,self.vx_des_min,
                self.vy_des_max,self.vy_des_min,
                self.w_des_max,self.w_des_min,gait,dis="uniform"
            )
            
            #TODO: check out simulation.py
            # Rollout MPC
            start_time = 0.0
            print("rolling out MPC")
            mpc_state, mpc_action, mpc_goal, _, mpc_base, _ = \
                        self.simulation.rollout_mpc(self.episode_length_eval, start_time, v_des, w_des, gait, nominal=True)
            
            
            # Rollout Policy
            print("Rolling out policy...")
            desired_goal = self._construct_desired_goal(v_des, w_des, gait)
            policy_state,policy_action,policy_goal,_,policy_base,_,_ = \
                self.simulation.rollout_policy(
                    self.episode_length_eval,start_time,v_des,w_des,gait,
                    self.vc_network,des_goal=desired_goal,
                    norm_policy_input=self.database.get_database_mean_std()
                )
            
            # Compute errors
            e_mpc = utils.compute_vc_mse(v_des, w_des, mpc_state[:, :2], mpc_state[:, 5])[0]
            e_policy = utils.compute_vc_mse(v_des, w_des, policy_state[:, :2], policy_state[:, 5])[0]
            print(f"e_MPC={e_mpc}, e_policy={e_policy}")
            
            # Update dataset
            if e_mpc < e_policy:
                self.database.append(mpc_state, mpc_action, vc_goals=mpc_goal)
                print("Added MPC samples to dataset.")
            else:
                self.database.append(policy_state, policy_action, vc_goals=policy_goal)
                print("Added policy samples to dataset.")
            
            # Train Policy
            print("Training policy...")
            self.database.set_goal_type('vc')
            self.vc_network = self.train_network(
                self.vc_network, batch_size=self.batch_size,
                learning_rate=self.learning_rate, n_epoch=self.n_epoch_data
            )
            
            # Save policy
            self.save_network(self.vc_network, name=f"policy_{i+1}")
        pass

@hydra.main(config_path='cfgs', config_name='safedagger_modified_config')
def main(cfg):
    icc = LocoSafeDagger(cfg)
    # icc.warmup()
    icc.database.load_saved_database(filename='/home/atari_ws/data/dagger_safedagger_warmup/dataset/database_112188.hdf5')
    icc.run() 

if __name__ == '__main__':
    main()   
    