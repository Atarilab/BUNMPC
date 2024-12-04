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


# set random seet for reproducability
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# login to wandb
wandb.login()
project_name = 'mod_safedagger_strict_1k_3'

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


class SafeDagger():  

    def __init__(self, cfg):        
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
        """warmup the database with expert data
        """        
        
        # NOTE: Warmup Rollout Loop
        for i_rollout in range(self.num_rollouts_warmup):
            
            print(f"============ Warmup Rollout {i_rollout+1}  ==============")
            
            # condition on which iterations to show GUI for Pybullet    
            display_simu = False
            
            # init env for if no pybullet server is active
            if self.simulation.currently_displaying_gui is None:
                self.simulation.init_pybullet_env(display_simu=display_simu)
            # change pybullet environment between with/without display, depending on condition
            elif display_simu != self.simulation.currently_displaying_gui:
                self.simulation.kill_pybullet_env()
                self.simulation.init_pybullet_env(display_simu=display_simu)
                
                
            ## Sampling of Velocity and Gait
            # randomly decide which gait to simulate
            gait = random.choice(self.gaits)

            # get desired velocities from probability distribution
            v_des, w_des = utils.get_des_velocities(self.vx_des_max, self.vx_des_min, self.vy_des_max, self.vy_des_min, 
                                                    self.w_des_max, self.w_des_min, gait, dist='uniform')
            
            # print selected gait and desired velocities
            print(f"-> gait: {gait} | v_des: {v_des} | w_des: {w_des}")
            
            
            # NOTE: Rollout benchmark MPC
            # set simulation start time to 0.0
            start_time = 0.0
            
            print("=== Benchmark MPC Rollout ===")
            
            # rollout mpc
            benchmark_state, benchmark_action, benchmark_vc_goal, benchmark_cc_goal, benchmark_base, _ = \
                self.simulation.rollout_mpc(self.episode_length_warmup, start_time, v_des, w_des, gait, nominal=True)
            
            # collect position and velocity of nominal trajectory
            nominal_pos, nominal_vel = self.simulation.q_nominal, self.simulation.v_nominal
            
            # get contact plan of benchmark mpc
            contact_plan = self.simulation.gg.cnt_plan
            
            # calculate number of replannings
            num_replanning = int(self.simulation.gait_params.gait_period/self.simulation.plan_freq)
            
            # Also record nominal mpc
            self.database.append(benchmark_state, benchmark_action, vc_goals=benchmark_vc_goal, cc_goals=benchmark_cc_goal)
            
            # NOTE: Rollout Loop with Pertubations
            for i_replan in range(num_replanning):
                
                ## Get Jacobian from nominal trajectory and contact plan for pertubation
                # get start time for simulation from state in nominal trajectory (within one full gait cycle)
                start_time = i_replan * self.simulation.plan_freq
                
                # get new q0 and v0 from the recorded nominal trajectory from benchmark
                new_q0 = nominal_pos[int(start_time/self.simulation.plan_freq)]
                new_v0 = nominal_vel[int(start_time/self.simulation.plan_freq)]
                
                # perform forward kinematic and jacobian calculation with new initial configuration
                self.simulation.pin_robot.computeJointJacobians(new_q0)
                self.simulation.pin_robot.framesForwardKinematics(new_q0)

                # find end-effectors in contact from contact plan
                ee_in_contact = []
                for ee in range(len(self.simulation.gg.eff_names)):
                    if contact_plan[i_replan][ee][0] == 1:
                        ee_in_contact.append(self.simulation.gg.eff_names[ee])
                        
                # initialize jacobian matrix
                cnt_jac = np.zeros((3*len(ee_in_contact), len(new_v0)))
                cnt_jac_dot = np.zeros((3*len(ee_in_contact), len(new_v0)))

                # compute Jacobian of end-effectors in contact and its derivative
                for ee_cnt in range(len(ee_in_contact)):
                    jac = pin.getFrameJacobian(self.simulation.pin_robot.model,\
                        self.simulation.pin_robot.data,\
                        self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]),\
                        pin.ReferenceFrame.LOCAL)
                    
                    cnt_jac[3*ee_cnt:3*(ee_cnt+1),] = utils.rotate_jacobian(self.simulation, jac,\
                        self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]))[0:3,]
                    
                    jac_dot = pin.getFrameJacobianTimeVariation(self.simulation.pin_robot.model,\
                        self.simulation.pin_robot.data,\
                        self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]),\
                        pin.ReferenceFrame.LOCAL)
                    
                    cnt_jac_dot[3*ee_cnt:3*(ee_cnt+1),] = utils.rotate_jacobian(self.simulation, jac_dot,\
                        self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]))[0:3,]
                
                
                # NOTE: Pertubation Loop
                for i_pertubation in range(self.num_pertubations_per_replanning_warmup):
                    
                    ## Calculate Contact Conditioned Pertubation
                    # perform pertubation until no foot is below the ground
                    min_ee_height = .0
                    while min_ee_height >= 0:
                        # sample pertubations (not yet contact conditioned)
                        perturbation_pos = np.concatenate((np.random.normal(self.mu_base_pos[gait], self.sigma_base_pos[gait], 3),\
                                                            np.random.normal(self.mu_base_ori[gait], self.sigma_base_ori[gait], 3), \
                                                            np.random.normal(self.mu_joint_pos[gait], self.sigma_joint_pos[gait], len(new_v0)-6)))
                        
                        perturbation_vel = np.random.normal(self.mu_vel[gait], self.sigma_vel[gait], len(new_v0))
                        
                        # Perform contact conditioned projection of pertubations
                        if ee_in_contact == []:
                            random_pos_vec = perturbation_pos
                            random_vel_vec = perturbation_vel
                        else:
                            random_pos_vec = (np.identity(len(new_v0)) - np.linalg.pinv(cnt_jac)@\
                                        cnt_jac) @ perturbation_pos
                            jac_vel = cnt_jac_dot * perturbation_pos + cnt_jac * perturbation_vel
                            random_vel_vec = (np.identity(len(new_v0)) - np.linalg.pinv(jac_vel)@\
                                        jac_vel) @ perturbation_pos

                        # add perturbation to nominal trajectory
                        new_v0 = nominal_vel[int(start_time/self.simulation.plan_freq)] + random_vel_vec
                        new_q0 = pin.integrate(self.simulation.pin_robot.model, \
                            nominal_pos[int(start_time/self.simulation.plan_freq)], random_pos_vec)

                        # check if the swing foot is below the ground
                        self.simulation.pin_robot.framesForwardKinematics(new_q0)
                        ee_below_ground = []
                        for e in range(len(self.simulation.gg.eff_names)):
                            frame_id = self.simulation.pin_robot.model.getFrameId(self.simulation.gg.eff_names[e])
                            if self.simulation.pin_robot.data.oMf[frame_id].translation[2] < 0.:
                                ee_below_ground.append(self.simulation.gg.eff_names[e])
                        if ee_below_ground == []:
                            min_ee_height = -1.
                    
                    # NOTE: MPC Rollout with Pertubation
                    ### perform rollout
                    print("=== MPC Rollout with pertubation - nom. traj. pos. ", str(i_replan+1), " pertubation ", str(i_pertubation + 1), " ===")
                    mpc_state, mpc_action, mpc_vc_goal, mpc_cc_goal, mpc_base, _ = self.simulation.rollout_mpc(self.episode_length_warmup, start_time, v_des, w_des, gait, 
                                                                                        nominal=True, q0=new_q0, v0=new_v0)
                    
                    print('MPC rollout completed. No. Datapoints: ' + str(len(mpc_cc_goal)))      
                    
                    if len(mpc_state) != 0:
                        self.database.append(mpc_state, mpc_action, vc_goals=mpc_vc_goal, cc_goals=mpc_cc_goal)
                        print("MPC data saved into database")
                        print('database size: ' + str(len(self.database)))    
                    else:
                        print('mpc rollout failed')
            
            # Save Database   
            self.save_dataset(iter=0)
        
        
    def run(self): 
        """Run SafeDAGGER 
        """        
        # filename = '/home/atari_ws/data/dagger/trot/May_26_2024_12_54_41/dataset/database_0.hdf5'
        # self.database.load_saved_database(filename=filename)
        
        # NOTE: Loop for policy data collection
        for i_main in range(self.num_iterations_safedagger):
            
            print(f"============ Iteration {i_main+1}  ==============")
            
            # NOTE: Train Policy
            wandb.init(project=project_name, config={'database_size':len(self.database), 'iteration':i_main+1}, job_type='training', name='training')
            
            print('=== Training VC Policy ===')
            self.database.set_goal_type('vc')
            if i_main == 0:
                self.vc_network = self.train_network(self.vc_network, batch_size=self.batch_size, learning_rate=self.learning_rate, n_epoch=self.n_epoch_warmup)
            else:
                self.vc_network = self.train_network(self.vc_network, batch_size=self.batch_size, learning_rate=self.learning_rate, n_epoch=self.n_epoch_data)
            self.save_network(self.vc_network, name='policy_'+str(i_main+1))
            wandb.finish()
            print('Policy training complete')
            
        
            # NOTE: Evaluation Loop
            # equal distance velocity sampling
            vx_des_list = np.linspace(self.vx_des_min, self.vx_des_max, num=self.num_rollouts_eval)
            
            for i_eval in range(self.num_rollouts_eval):
                
                for gait in self.gaits:
                
                    # condition on which iterations to show GUI for Pybullet    
                    display_simu = False
                    
                    # init env for if no pybullet server is active
                    if self.simulation.currently_displaying_gui is None:
                        self.simulation.init_pybullet_env(display_simu=display_simu)
                    # change pybullet environment between with/without display, depending on condition
                    elif display_simu != self.simulation.currently_displaying_gui:
                        self.simulation.kill_pybullet_env()
                        self.simulation.init_pybullet_env(display_simu=display_simu)
                        
                        
                    print(f"============ Evaluation Rollout {i_eval+1}  ==============")
                    
                    v_des = np.array([vx_des_list[i_eval], 0.0, 0.0])
                    w_des = 0.0
                    
                    # print selected gait and desired velocities
                    print(f"-> gait: {gait} | v_des: {v_des} | w_des: {w_des}")
                    
                    # set simulation start time to 0.0
                    start_time = 0.0
                    
                    print("=== Benchmark MPC Rollout ===")
                    
                    # rollout mpc
                    benchmark_state, benchmark_action, benchmark_vc_goal, benchmark_cc_goal, benchmark_base, _ = \
                        self.simulation.rollout_mpc(self.episode_length_eval, start_time, v_des, w_des, gait, nominal=True)
                    
                    # collect position and velocity of nominal trajectory
                    nominal_pos, nominal_vel = self.simulation.q_nominal, self.simulation.v_nominal
                    
                    # get contact plan of benchmark mpc
                    contact_plan = self.simulation.gg.cnt_plan
                    
                    # calculate number of replannings
                    num_replanning = int(self.simulation.gait_params.gait_period/self.simulation.plan_freq)
                    
                    
                    # NOTE: Rollout Loop with Pertubations
                    for i_replan in range(num_replanning):
                        
                        ## Get Jacobian from nominal trajectory and contact plan for pertubation
                        # get start time for simulation from state in nominal trajectory (within one full gait cycle)
                        start_time = i_replan * self.simulation.plan_freq
                        
                        # get new q0 and v0 from the recorded nominal trajectory from benchmark
                        new_q0 = nominal_pos[int(start_time/self.simulation.plan_freq)]
                        new_v0 = nominal_vel[int(start_time/self.simulation.plan_freq)]
                        
                        # perform forward kinematic and jacobian calculation with new initial configuration
                        self.simulation.pin_robot.computeJointJacobians(new_q0)
                        self.simulation.pin_robot.framesForwardKinematics(new_q0)

                        # find end-effectors in contact from contact plan
                        ee_in_contact = []
                        for ee in range(len(self.simulation.gg.eff_names)):
                            if contact_plan[i_replan][ee][0] == 1:
                                ee_in_contact.append(self.simulation.gg.eff_names[ee])
                                
                        # initialize jacobian matrix
                        cnt_jac = np.zeros((3*len(ee_in_contact), len(new_v0)))
                        cnt_jac_dot = np.zeros((3*len(ee_in_contact), len(new_v0)))

                        # compute Jacobian of end-effectors in contact and its derivative
                        for ee_cnt in range(len(ee_in_contact)):
                            jac = pin.getFrameJacobian(self.simulation.pin_robot.model,\
                                self.simulation.pin_robot.data,\
                                self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]),\
                                pin.ReferenceFrame.LOCAL)
                            
                            cnt_jac[3*ee_cnt:3*(ee_cnt+1),] = utils.rotate_jacobian(self.simulation, jac,\
                                self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]))[0:3,]
                            
                            jac_dot = pin.getFrameJacobianTimeVariation(self.simulation.pin_robot.model,\
                                self.simulation.pin_robot.data,\
                                self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]),\
                                pin.ReferenceFrame.LOCAL)
                            
                            cnt_jac_dot[3*ee_cnt:3*(ee_cnt+1),] = utils.rotate_jacobian(self.simulation, jac_dot,\
                                self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]))[0:3,]
                        
                        
                        # NOTE: Pertubation Loop
                        for i_pertubation in range(self.num_pertubations_per_replanning_eval):
                            
                            ## Calculate Contact Conditioned Pertubation
                            # perform pertubation until no foot is below the ground
                            min_ee_height = .0
                            while min_ee_height >= 0:
                                # sample pertubations (not yet contact conditioned)
                                perturbation_pos = np.concatenate((np.random.normal(self.mu_base_pos[gait], self.sigma_base_pos[gait], 3),\
                                                                    np.random.normal(self.mu_base_ori[gait], self.sigma_base_ori[gait], 3), \
                                                                    np.random.normal(self.mu_joint_pos[gait], self.sigma_joint_pos[gait], len(new_v0)-6)))
                                
                                perturbation_vel = np.random.normal(self.mu_vel[gait], self.sigma_vel[gait], len(new_v0))
                                
                                # Perform contact conditioned projection of pertubations
                                if ee_in_contact == []:
                                    random_pos_vec = perturbation_pos
                                    random_vel_vec = perturbation_vel
                                else:
                                    random_pos_vec = (np.identity(len(new_v0)) - np.linalg.pinv(cnt_jac)@\
                                                cnt_jac) @ perturbation_pos
                                    jac_vel = cnt_jac_dot * perturbation_pos + cnt_jac * perturbation_vel
                                    random_vel_vec = (np.identity(len(new_v0)) - np.linalg.pinv(jac_vel)@\
                                                jac_vel) @ perturbation_pos

                                # add perturbation to nominal trajectory
                                new_v0 = nominal_vel[int(start_time/self.simulation.plan_freq)] + random_vel_vec
                                new_q0 = pin.integrate(self.simulation.pin_robot.model, \
                                    nominal_pos[int(start_time/self.simulation.plan_freq)], random_pos_vec)

                                # check if the swing foot is below the ground
                                self.simulation.pin_robot.framesForwardKinematics(new_q0)
                                ee_below_ground = []
                                for e in range(len(self.simulation.gg.eff_names)):
                                    frame_id = self.simulation.pin_robot.model.getFrameId(self.simulation.gg.eff_names[e])
                                    if self.simulation.pin_robot.data.oMf[frame_id].translation[2] < 0.:
                                        ee_below_ground.append(self.simulation.gg.eff_names[e])
                                if ee_below_ground == []:
                                    min_ee_height = -1.
                                    
                            ## Rollout Policy for evaluation
                            wandb.init(project=project_name, config={'database_size':len(self.database), 'iteration':i_main+1, 'gait':gait}, job_type='evaluation', 
                                    name='eval_'+str(i_eval)+'_'+str(i_replan)+'_'+str(i_pertubation)+'_'+gait)
                            
                            wandb.log({'vx_des': v_des[0]})                
                            
                            # VC desired goal
                            start_i = int(start_time/self.sim_dt)
                            desired_goal = np.zeros((self.episode_length_eval - start_i, 5))
                            for t in range(start_i, self.episode_length_eval):
                                desired_goal[t-start_i, 0] = utils.get_phase_percentage(t, self.sim_dt, gait)
                            
                            desired_goal[:, 1] = np.full(np.shape(desired_goal[:, 1]), v_des[0])
                            desired_goal[:, 2] = np.full(np.shape(desired_goal[:, 2]), v_des[1])
                            desired_goal[:, 3] = np.full(np.shape(desired_goal[:, 3]), w_des)
                            desired_goal[:, 4] = np.full(np.shape(desired_goal[:, 4]), utils.get_vc_gait_value(gait))
                            
                            ### perform rollout
                            print("=== Policy Rollout with pertubation - ", str(i_main+1), '_', str(i_replan+1), '_', str(i_pertubation), " ===")
                            
                            self.database.set_goal_type('vc')
                            policy_state, policy_action, policy_vc_goal, policy_cc_goal, policy_base, _, _, frames = \
                            self.simulation.rollout_policy(self.episode_length_eval, start_time, v_des, w_des, gait, 
                                                            self.vc_network, des_goal=desired_goal, q0=new_q0, v0=new_v0, 
                                                            norm_policy_input=self.database.get_database_mean_std(), save_video=True)
                            
                            print('Policy rollout completed. No. Datapoints: ' + str(len(policy_cc_goal)))   
                            
                            # Transpose the array to (time, channel, height, width) then log video
                            if len(frames) != 0:
                                video_array_transposed = np.array(frames).transpose(0, 3, 1, 2)
                                wandb.log({'video': wandb.Video(video_array_transposed, fps=self.simulation.video_fr)})    
                            
                            if len(policy_cc_goal) < (self.episode_length_eval * 2 / 3):
                                print('Policy Datapoints too little! Policy will be considered as failed')
                                wandb.log({'sim_successful': 0})
                            else:                            
                                wandb.log({'sim_successful': 1})
                                
                                for a in range(len(policy_state)):
                                    wandb.log({'v_x': policy_state[a, 0]})
                                    wandb.log({'v_y': policy_state[a, 1]})
                                    wandb.log({'v_z': policy_state[a, 2]})
                                    wandb.log({'w': policy_state[a, 5]})
                                    
                                    for b in range(len(policy_base[0])):
                                        key = 'policy_base_' + str(b)
                                        wandb.log({key: policy_base[a, b]})
                                        
                                        
                                ## Compute Goal Reaching Error
                                print("=== Computing Goal Reaching Error ===")

                                vx_error, vy_error, w_error = utils.compute_vc_mse(v_des, w_des, policy_state[:, 0:2], policy_state[:, 5])
                                wandb.log({'vx_mse': vx_error, 'vy_mse': vy_error, 'w_mse': w_error})
                                
                            wandb.finish()
                        
            # NOTE: Data Collection Loop
            for i_rollout in range(self.num_rollouts_per_iteration_data):
                
                # record number of successful data collection rollouts (that does not fail)
                number_of_successful_rollouts = 0
                total_number_of_rollouts = 0
                
                # condition on which iterations to show GUI for Pybullet    
                display_simu = False
                
                # init env for if no pybullet server is active
                if self.simulation.currently_displaying_gui is None:
                    self.simulation.init_pybullet_env(display_simu=display_simu)
                # change pybullet environment between with/without display, depending on condition
                elif display_simu != self.simulation.currently_displaying_gui:
                    self.simulation.kill_pybullet_env()
                    self.simulation.init_pybullet_env(display_simu=display_simu)
                    
                    
                print(f"============ Data Collection Rollout {i_rollout+1}  ==============")
                    
                # randomly decide which gait to simulate
                gait = random.choice(self.gaits)

                # get desired velocities from probability distribution
                v_des, w_des = utils.get_des_velocities(self.vx_des_max, self.vx_des_min, self.vy_des_max, self.vy_des_min, 
                                                        self.w_des_max, self.w_des_min, gait, dist='uniform')
                
                # print selected gait and desired velocities
                print(f"-> gait: {gait} | v_des: {v_des} | w_des: {w_des}")
                
                
                # set simulation start time to 0.0
                start_time = 0.0
                
                print("=== Benchmark MPC Rollout ===")
                
                # rollout mpc
                benchmark_state, benchmark_action, benchmark_vc_goal, benchmark_cc_goal, benchmark_base, _ = \
                    self.simulation.rollout_mpc(self.episode_length_data, start_time, v_des, w_des, gait, nominal=True)
                
                # collect position and velocity of nominal trajectory
                nominal_pos, nominal_vel = self.simulation.q_nominal, self.simulation.v_nominal
                
                # get contact plan of benchmark mpc
                contact_plan = self.simulation.gg.cnt_plan
                
                # calculate number of replannings
                # num_replanning = int(self.simulation.gait_params.gait_period/self.simulation.plan_freq)
                
                # random sample starting point along nominal trajectory
                replanning_list = np.random.randint(0, int(self.simulation.gait_params.gait_period/self.simulation.plan_freq), (self.num_replannings_on_nom_traj_data,)).tolist()
                
                # NOTE: Rollout Loop with Pertubations
                for i_replan in replanning_list:
                    
                    ## Get Jacobian from nominal trajectory and contact plan for pertubation
                    # get start time for simulation from state in nominal trajectory (within one full gait cycle)
                    start_time = i_replan * self.simulation.plan_freq
                    
                    # get new q0 and v0 from the recorded nominal trajectory from benchmark
                    new_q0 = nominal_pos[int(start_time/self.simulation.plan_freq)]
                    new_v0 = nominal_vel[int(start_time/self.simulation.plan_freq)]
                    
                    # perform forward kinematic and jacobian calculation with new initial configuration
                    self.simulation.pin_robot.computeJointJacobians(new_q0)
                    self.simulation.pin_robot.framesForwardKinematics(new_q0)

                    # find end-effectors in contact from contact plan
                    ee_in_contact = []
                    for ee in range(len(self.simulation.gg.eff_names)):
                        if contact_plan[i_replan][ee][0] == 1:
                            ee_in_contact.append(self.simulation.gg.eff_names[ee])
                            
                    # initialize jacobian matrix
                    cnt_jac = np.zeros((3*len(ee_in_contact), len(new_v0)))
                    cnt_jac_dot = np.zeros((3*len(ee_in_contact), len(new_v0)))

                    # compute Jacobian of end-effectors in contact and its derivative
                    for ee_cnt in range(len(ee_in_contact)):
                        jac = pin.getFrameJacobian(self.simulation.pin_robot.model,\
                            self.simulation.pin_robot.data,\
                            self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]),\
                            pin.ReferenceFrame.LOCAL)
                        
                        cnt_jac[3*ee_cnt:3*(ee_cnt+1),] = utils.rotate_jacobian(self.simulation, jac,\
                            self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]))[0:3,]
                        
                        jac_dot = pin.getFrameJacobianTimeVariation(self.simulation.pin_robot.model,\
                            self.simulation.pin_robot.data,\
                            self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]),\
                            pin.ReferenceFrame.LOCAL)
                        
                        cnt_jac_dot[3*ee_cnt:3*(ee_cnt+1),] = utils.rotate_jacobian(self.simulation, jac_dot,\
                            self.simulation.pin_robot.model.getFrameId(ee_in_contact[ee_cnt]))[0:3,]
                    
                    
                    # NOTE: Pertubation Loop
                    for i_pertubation in range(self.num_pertubations_per_replanning_data):
                            
                        # NOTE: Calculate Contact Conditioned Pertubation
                        # perform pertubation until no foot is below the ground
                        min_ee_height = .0
                        while min_ee_height >= 0:
                            # sample pertubations (not yet contact conditioned)
                            perturbation_pos = np.concatenate((np.random.normal(self.mu_base_pos[gait], self.sigma_base_pos[gait], 3),\
                                                                np.random.normal(self.mu_base_ori[gait], self.sigma_base_ori[gait], 3), \
                                                                np.random.normal(self.mu_joint_pos[gait], self.sigma_joint_pos[gait], len(new_v0)-6)))
                            
                            perturbation_vel = np.random.normal(self.mu_vel[gait], self.sigma_vel[gait], len(new_v0))
                            
                            # Perform contact conditioned projection of pertubations
                            if ee_in_contact == []:
                                random_pos_vec = perturbation_pos
                                random_vel_vec = perturbation_vel
                            else:
                                random_pos_vec = (np.identity(len(new_v0)) - np.linalg.pinv(cnt_jac)@\
                                            cnt_jac) @ perturbation_pos
                                jac_vel = cnt_jac_dot * perturbation_pos + cnt_jac * perturbation_vel
                                random_vel_vec = (np.identity(len(new_v0)) - np.linalg.pinv(jac_vel)@\
                                            jac_vel) @ perturbation_pos

                            # add perturbation to nominal trajectory
                            new_v0 = nominal_vel[int(start_time/self.simulation.plan_freq)] + random_vel_vec
                            new_q0 = pin.integrate(self.simulation.pin_robot.model, \
                                nominal_pos[int(start_time/self.simulation.plan_freq)], random_pos_vec)

                            # check if the swing foot is below the ground
                            self.simulation.pin_robot.framesForwardKinematics(new_q0)
                            ee_below_ground = []
                            for e in range(len(self.simulation.gg.eff_names)):
                                frame_id = self.simulation.pin_robot.model.getFrameId(self.simulation.gg.eff_names[e])
                                if self.simulation.pin_robot.data.oMf[frame_id].translation[2] < 0.:
                                    ee_below_ground.append(self.simulation.gg.eff_names[e])
                            if ee_below_ground == []:
                                min_ee_height = -1.
                            
                                
                        # NOTE: Rollout Policy for Data Collection
                        # VC desired goal
                        start_i = int(start_time/self.sim_dt)
                        desired_goal = np.zeros((self.episode_length_data - start_i, 5))
                        for t in range(start_i, self.episode_length_data):
                            desired_goal[t-start_i, 0] = utils.get_phase_percentage(t, self.sim_dt, gait)
                        
                        desired_goal[:, 1] = np.full(np.shape(desired_goal[:, 1]), v_des[0])
                        desired_goal[:, 2] = np.full(np.shape(desired_goal[:, 2]), v_des[1])
                        desired_goal[:, 3] = np.full(np.shape(desired_goal[:, 3]), w_des)
                        desired_goal[:, 4] = np.full(np.shape(desired_goal[:, 4]), utils.get_vc_gait_value(gait))
                        
                        ### perform rollout
                        print("=== Policy Rollout with pertubation - ", str(i_main+1), '_', str(i_replan+1), " ===")
                        
                        self.database.set_goal_type('vc')
                        policy_state, policy_action, policy_vc_goal, policy_base, q_history, v_history, mpc_usage, frames = \
                        self.simulation.rollout_safedagger(self.episode_length_data, start_time, v_des, w_des, gait, 
                                                        self.vc_network, des_goal=desired_goal, q0=new_q0, v0=new_v0, 
                                                        norm_policy_input=self.database.get_database_mean_std(), 
                                                        num_steps_to_block_under_safety=self.num_steps_to_block_under_safety, save_video=True,
                                                        bounds_dict=safety_bounds_dict)
                        
                        wandb.init(project=project_name, config={'iteration':i_main+1}, job_type='data_collection', 
                                    name='data_'+str(i_main)+'_'+str(i_rollout)+'_'+str(i_replan)+'_'+str(i_pertubation))
                        
                        # Transpose the array to (time, channel, height, width) then log video
                        if len(frames) != 0:
                            video_array_transposed = np.array(frames).transpose(0, 3, 1, 2)
                            wandb.log({'video': wandb.Video(video_array_transposed, fps=self.simulation.video_fr)}) 
                        
                        # record if policy rollout was successful
                        if len(policy_state) != 0:
                            number_of_successful_rollouts += 1
                            wandb.log({'mpc_usage':np.mean(mpc_usage)})
                        
                        # record total number of policy rollouts
                        total_number_of_rollouts += 1
                        
                        wandb.log({'rollout_length':len(q_history)})
                        print('Policy rollout completed. No. of datapoints: ' + str(len(policy_state)))  
                        wandb.finish()
                         
                        # save mpc data to database if sim is successful
                        if len(policy_state) != 0:
                            print('No. of expert datapoints: ' + str(len(policy_state[mpc_usage==1])))  
                            self.database.append(policy_state[mpc_usage==1], policy_action[mpc_usage==1], vc_goals=policy_vc_goal[mpc_usage==1])
                            print("data saved into database")
                            print('database size: ' + str(len(self.database)))
                        else:
                            print('MPC rollout failed!')
                            
                        # NOTE: Rollout MPC to increase dataset
                        if len(q_history) > 0:
                            if self.ending_mpc_rollout_episode_length > 0:
                                # rollout mpc from the last state of policy rollout
                                q_mpc, v_mpc = q_history[-1], v_history[-1]
                                start_time_mpc = int(len(q_mpc)*self.sim_dt)
                                mpc_state, mpc_action, mpc_vc_goal, mpc_cc_goal, mpc_base, _ = self.simulation.rollout_mpc(self.ending_mpc_rollout_episode_length + start_time_mpc, start_time_mpc, 
                                                                                                                        v_des, w_des, gait, q0=q_mpc, v0=v_mpc)
                                
                                print('MPC rollout completed. No. Datapoints: ' + str(len(mpc_cc_goal)))      
                                        
                                # save mpc data to database if sim is successful
                                if len(mpc_state) != 0:
                                    self.database.append(mpc_state, mpc_action, vc_goals=mpc_vc_goal)
                                    print("MPC data saved into database")
                                    print('database size: ' + str(len(self.database)))
                                else:
                                    print('mpc rollout failed')
                        
                        # save database
                        self.save_dataset(iter=0)
                            
                
                # record percentage of successful policy rollouts         
                wandb.init(project=project_name, config={'iteration':i_main+1}, job_type='data_collection', 
                                        name='data_rollout_'+str(i_main)+'_'+str(i_rollout)+'_'+gait)
                wandb.log({'data_collection_rollout_success_percentage': number_of_successful_rollouts / total_number_of_rollouts,
                           'total_number_of_data_rollouts': total_number_of_rollouts,
                           'number_of_successful_data_rollouts': number_of_successful_rollouts})
                wandb.finish()
                        
                         

@hydra.main(config_path='cfgs', config_name='safedagger_modified_config')
def main(cfg):
    icc = SafeDagger(cfg)
    # icc.warmup()
    icc.database.load_saved_database(filename='/home/atari_ws/data/dagger_safedagger_warmup/dataset/database_112188.hdf5')
    icc.run() 

if __name__ == '__main__':
    main()
    
        


    


