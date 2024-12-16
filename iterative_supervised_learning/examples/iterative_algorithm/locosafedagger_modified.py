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
import json
import sys
import time
import wandb
from test_trained_policy import TestTrainedPolicy as TTP
from contact_planner import ContactPlanner
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

from networks import GoalConditionedPolicyNet
import torch.serialization
from networks import GoalConditionedPolicyNet

# Allowlist the custom class
torch.serialization.add_safe_globals([GoalConditionedPolicyNet])



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
        self.errors = []
        self.goals = []
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
        self.num_iterations_locosafedagger = cfg.num_iterations_locosafedagger
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
        self.vc_network = file['network'].to(self.device)
        self.vc_network.eval()
        self.policy_input_parameters = file['norm_policy_input']

        print("Policy Network loaded from: " + filename)
        if self.policy_input_parameters is None:
            print('Policy Input will NOT be normalized')
        else:
            print('Policy Input will be normalized')
    
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
        """Rollout MPC once and do behavior cloning on the dataset to train a working policy first
        """
        pass
    
    def compute_likelihood(self,vx_vals, vy_vals, w_vals, observed_goal, error, sigma=0.1):
        """
        Compute the likelihood P(e | vx, vy, w) for each grid point.

        Args:
            vx_vals (array): Discretized vx values.
            vy_vals (array): Discretized vy values.
            w_vals (array): Discretized w values.
            observed_goal (tuple): Observed goal (vx, vy, w).
            error (float): Observed error associated with the goal.
            sigma (float): Standard deviation for the Gaussian.

        Returns:
            ndarray: Likelihood values for the entire grid.
        """
        vx_obs, vy_obs, w_obs = observed_goal
        likelihood = np.zeros((len(vx_vals), len(vy_vals), len(w_vals)))

        for i, vx in enumerate(vx_vals):
            for j, vy in enumerate(vy_vals):
                for k, w in enumerate(w_vals):
                    # Gaussian likelihood centered at the observed goal
                    goal_diff = np.array([vx - vx_obs, vy - vy_obs, w - w_obs])
                    likelihood[i, j, k] = np.exp(-np.sum(goal_diff**2) / (2 * sigma**2))

        # Normalize likelihood (optional for stability)
        likelihood /= np.sum(likelihood)
        return likelihood
    
    def update_goal_distribution(self,P_vxvyw, likelihood):
        """
        Update the goal distribution using the likelihood.

        Args:
            P_vxvyw (ndarray): Current prior distribution P(vx, vy, w).
            likelihood (ndarray): Likelihood P(e | vx, vy, w).

        Returns:
            ndarray: Updated posterior distribution P(vx, vy, w | e).
        """
        # Compute the unnormalized posterior
        posterior = P_vxvyw * likelihood

        # Normalize the posterior to sum to 1
        posterior /= np.sum(posterior)
        return posterior
    
    def random_sample_from_distribution(self,P_vxvyw, vx_vals, vy_vals, w_vals):
        """
        Sample a goal (vx, vy, w) from the updated distribution.

        Args:
            P_vxvyw (ndarray): Updated posterior distribution.
            vx_vals (array): Discretized vx values.
            vy_vals (array): Discretized vy values.
            w_vals (array): Discretized w values.

        Returns:
            tuple: Sampled goal (vx, vy, w).
        """
        # Flatten the distribution and sample an index
        flat_distribution = P_vxvyw.flatten()
        sampled_index = np.random.choice(len(flat_distribution), p=flat_distribution)

        # Convert the index back to grid coordinates
        i, j, k = np.unravel_index(sampled_index, P_vxvyw.shape)
        return vx_vals[i], vy_vals[j], w_vals[k]

    def error_based_sample_from_distribution(self):
        pass

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

        plan = utils.get_plan(gait)
        cp = ContactPlanner(plan)
        contact_schedule, cnt_plan = cp.get_contact_schedule(pin_robot, urdf_path, q0, v0, v_des, w_des, self.episode_length_data, start_time)
        return contact_schedule, cnt_plan
    
    def run_unperturbed(self):
        # TODO:Run through a pipeline that: sample goals -> rollout MPC -> rollout Policy ->
        # collect the realized contact plan for MPC and policy -> compute error -> data aggregation
        # -> update policy -> update goal distribution depending on the error 
        """Run the modified locosafedagger algorithm
        """
        # Define the range and resolution of the goal space
        vx_min,vx_max,vx_bins = self.vx_des_min, self.vx_des_max, 100
        vy_min,vy_max,vy_bins = self.vy_des_min, self.vy_des_max, 100
        w_min,w_max,w_bins= self.w_des_min,self.w_des_max,100
        
        # Create a 3D grid for (vx, vy, w)
        vx_vals = np.linspace(vx_min, vx_max, vx_bins)
        vy_vals = np.linspace(vy_min, vy_max, vy_bins)
        w_vals = np.linspace(w_min, w_max, w_bins)
        
        # initialize the uniform distribution over the grid
        P_vxvyw = np.ones((vx_bins, vy_bins, w_bins)) / (vx_bins * vy_bins * w_bins)
            
        for i in range(self.num_iterations_locosafedagger):
            print(f"============ Iteration {i+1} ==============")
            
            ## Train policy
            wandb.init(project=project_name, config={'database_size':len(self.database), 'iteration':i+1}, job_type='training', name='training')
            print('=== Training VC Policy ===')
            self.database.set_goal_type('vc')
            if i == 0:  # warmup (different epoch!)
                self.vc_network = self.train_network(self.vc_network, batch_size=self.batch_size, learning_rate=self.learning_rate, n_epoch=self.n_epoch_warmup)
            else:  # normal training
                self.vc_network = self.train_network(self.vc_network, batch_size=self.batch_size, learning_rate=self.learning_rate, n_epoch=self.n_epoch_data)
                
            self.save_network(self.vc_network, name='policy_'+str(i+1))
            wandb.finish()
            print('Policy {} training complete',i)
            
            ## sample goals from the updated distribution
            new_goal = self.random_sample_from_distribution(P_vxvyw, vx_vals, vy_vals, w_vals)
            print(f"Sampled new goal: {new_goal}")
            
            gait = 'trot'
            # gait = random.choice(self.gaits)
            print("gait chose is ",gait)
            
            v_des = np.array([new_goal[0], new_goal[1], 0]) # [vx_des,vy_des,vz_des] with vz_des = 0 always
            w_des = np.array(new_goal[2])
            
            # v_des, w_des = utils.get_des_velocities(self.vx_des_max, self.vx_des_min, self.vy_des_max, self.vy_des_min, 
            #                                             self.w_des_max, self.w_des_min, gait, dist='uniform')
            
            ## Rollout MPC
            start_time = 0.0

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
                
            print("=== MPC Rollout ===")
            mpc_state, mpc_action, mpc_vc_goal, mpc_cc_goal, mpc_base, _ = \
                        self.simulation.rollout_mpc(self.episode_length_eval, start_time, v_des, w_des, gait, nominal=True)               
            # TODO: What's difference between vc_goal and cc_goal? 
            # TODO: find desired goal and realized goal
            
            # collect position and velocity of nominal trajectory
            nominal_pos, nominal_vel = self.simulation.q_nominal, self.simulation.v_nominal
            
            # get contact plan of benchmark mpc
            contact_plan = self.simulation.gg.cnt_plan

            #====================================================================================================================================================     
            ## Rollout Policy
            # why is wandb involved?
            # wandb.init(project=project_name, config={'database_size':len(self.database), 'iteration':i, 'gait':gait}, job_type='rollout_policy', 
            #                         name='rollout_policy_'+str(i)+'_'+gait)
            # wandb.log({'vx_des': v_des[0]}) 
            
            # VC desired goal
            start_i = int(start_time/self.sim_dt)
            desired_goal = np.zeros((self.episode_length_eval - start_i, 5))
            for t in range(start_i, self.episode_length_eval):
                desired_goal[t-start_i, 0] = utils.get_phase_percentage(t, self.sim_dt, gait)

            desired_goal[:, 1] = np.full(np.shape(desired_goal[:, 1]), v_des[0])
            desired_goal[:, 2] = np.full(np.shape(desired_goal[:, 2]), v_des[1])
            desired_goal[:, 3] = np.full(np.shape(desired_goal[:, 3]), w_des)
            desired_goal[:, 4] = np.full(np.shape(desired_goal[:, 4]), utils.get_vc_gait_value(gait))
           
            print("=== Policy Rollout ===")
            # for testing
            # model_path = "/home/atari_ws/iterative_supervised_learning/examples/iterative_algorithm/data/safedagger/trot/Dec_16_2024_14_12_55/network/policy_1.pth"
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.vc_network = self.initialize_network(input_size=self.vc_input_size, output_size=self.output_size, 
            #                                         num_hidden_layer=self.cfg.num_hidden_layer, hidden_dim=self.cfg.hidden_dim,
            #                                         batch_norm=True)
            # state_dict = torch.load(model_path, map_location=device,weights_only=True)
            # self.vc_network.load_state_dict(state_dict)
            # self.vc_network.eval()
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            self.database.set_goal_type('vc')
            policy_state, policy_action, policy_vc_goal, policy_cc_goal, policy_base, _, _, frames = \
            self.simulation.rollout_policy(self.episode_length_eval, start_time, v_des, w_des, gait, 
                                            self.vc_network, des_goal=desired_goal, q0=None, v0=None, 
                                            norm_policy_input=self.database.get_database_mean_std(), save_video=True)
            
            ## calculate goal-reaching error
            weights = {
                "vx": 0.4,  # Weight for policy_vx_error
                "vy": 0.3,  # Weight for policy_vy_error
                "w": 0.3    # Weight for policy_w_error
            }
            # policy error
            policy_vx_error,policy_vy_error,policy_w_error = utils.compute_vc_mse(v_des, w_des, policy_state[:, 0:2], policy_state[:, 5])
            e_policy = (
                weights["vx"] * policy_vx_error**2 +
                weights["vy"] * policy_vy_error**2 +
                weights["w"] * policy_w_error**2
            )
            
            # mpc error
            mpc_vx_error,mpc_vy_error,mpc_w_error = utils.compute_vc_mse(v_des,w_des,mpc_state[:,0:2],mpc_state[:,5]) 
            e_mpc = (
                weights["vx"] * mpc_vx_error**2 +
                weights["vy"] * mpc_vy_error**2 +
                weights["w"] * mpc_w_error**2
            )
            
            if e_mpc > e_policy:
                self.errors.append(e_policy)
            else:
                self.errors.append(e_mpc)
                        
            self.goals.append([v_des,w_des])
            
            
            ## TODO:Update dataset
            
            ## TODO:Update goal distribution with observed errors
            # Compute the likelihood for the current observation
            likelihood = self.compute_likelihood(vx_vals, vy_vals, w_vals, self.goals[-1], self.errors[-1], sigma=0.1)
            
            # Update the goal distribution
            P_vxvyw = self.update_goal_distribution(P_vxvyw, likelihood)


@hydra.main(config_path='cfgs', config_name='locosafedagger_modified_config')
def main(cfg):
    icc = LocoSafeDagger(cfg)
    # icc.warmup()
    # icc.database.load_saved_database(filename='/home/atari_ws/data/dagger_safedagger_warmup/dataset/database_112188.hdf5')
    icc.database.load_saved_database(filename='/home/atari_ws/iterative_supervised_learning/examples/iterative_algorithm/data/behavior_cloning/trot/Dec_04_2024_16_51_02/dataset/database_1047158.hdf5')
    # icc.run()
    icc.run_unperturbed()  

if __name__ == '__main__':
    main()   
    