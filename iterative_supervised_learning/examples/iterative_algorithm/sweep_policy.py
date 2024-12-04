import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from simulation import Simulation
from contact_planner import ContactPlanner
import utils
import pinocchio as pin
from database import Database

import numpy as np
import random
import hydra
import os
from tqdm import tqdm
from datetime import datetime
import h5py
import pickle
import sys
import time
import wandb

# set random seet for reproducability
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class SweepPolicy():  

    def __init__(self, cfg):        
        # configuration file (containing the hyper/parameters)
        self.cfg = cfg
            
        # Simulation rollout properties
        self.episode_length = cfg.episode_length
        self.sim_dt = cfg.sim_dt
        
        # rollout pertubations
        self.mu_base_pos, self.sigma_base_pos = cfg.mu_base_pos, cfg.sigma_base_pos # base position
        self.mu_joint_pos, self.sigma_joint_pos = cfg.mu_joint_pos, cfg.sigma_joint_pos # joint position
        self.mu_base_ori, self.sigma_base_ori = cfg.mu_base_ori, cfg.sigma_base_ori # base orientation
        self.mu_vel, self.sigma_vel = cfg.mu_vel, cfg.sigma_vel # joint velocity
        
        # Model Parameters
        self.action_type = cfg.action_type
        self.normalize_policy_input = cfg.normalize_policy_input
        
        # Data related parameters 
        self.n_state = cfg.n_state
        self.n_action = cfg.n_action
        self.goal_horizon = cfg.goal_horizon
        
        # Desired Motion Parameters
        self.gaits = cfg.gaits
        self.vx_des_min, self.vx_des_max = cfg.vx_des_min, cfg.vx_des_max
        self.vy_des_min, self.vy_des_max = cfg.vy_des_min, cfg.vy_des_max
        self.w_des_min, self.w_des_max = cfg.w_des_min, cfg.w_des_max
        
        self.criterion = nn.L1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Nvidia GPU availability is ' + str(torch.cuda.is_available()))
        
        # Training properties
        # self.n_epoch = cfg.n_epoch  # per iteration
        # self.batch_size = cfg.batch_size
        self.n_train_frac = cfg.n_train_frac
        # self.learning_rate = cfg.learning_rate
        
        # init simulation class
        self.simulation = Simulation(cfg=self.cfg)
    
    
    def initialize_network(self, input_size=0, output_size=0, num_hidden_layer=3, hidden_dim=512, batch_norm=True):
        """
        load policy network and determine input and output sizes
        """        
        from networks import GoalConditionedPolicyNet
        
        network = GoalConditionedPolicyNet(input_size, output_size, num_hidden_layer=num_hidden_layer, 
                                                hidden_dim=hidden_dim, batch_norm=batch_norm).to(self.device)
        print("Policy Network initialized")
        return network
        
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
        contact_schedule, cnt_plan = cp.get_contact_schedule(pin_robot, urdf_path, q0, v0, v_des, w_des, self.episode_length, start_time)
        return contact_schedule, cnt_plan
    
    def train_network(self, network, batch_size=256, learning_rate=0.002, n_epoch=150):
        """
        Train and validate the policy network with samples from the current dataset

        Args:
            dataset (Pytorch Dataset): the sampled and splitted dataset for training
            current_iter (int): current simulation step (Not Time!)
            plot_loss (bool, optional): (Non-Blocking) Plot the Training and validation 
            loss. Defaults to False.


        """        
        
        # get the training dataset size (use whole dataset)
        train_set_size = len(self.database)

        print("Dataset size: " + str(train_set_size))
        print(f'Batch size: {batch_size}')
        print(f'learning rate: {learning_rate}')
        print(f'num of epochs: {n_epoch}')

        # define training and test set size
        n_train = int(self.n_train_frac*train_set_size)
        n_test = train_set_size - n_train
        
        print(f'training data size: {n_train}')
        print(f'validation data size: {n_test}')
        
        n_batches_train = int(n_train/batch_size)
        n_batches_test = int(n_test/batch_size)
        
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
            training_running_loss = 0
            
            # train network
            for x, y in train_loader:
                self.optimizer.zero_grad()
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_pred = network(x)
                loss = self.criterion(y_pred, y)
                
                loss.backward()
                self.optimizer.step()
                training_running_loss += loss.item()
            

            # test network
            test_running_loss = 0
            network.eval()
            for z, w in test_loader:
                z, w = z.to(self.device).float(), w.to(self.device).float()
                w_pred = network(z)
                test_loss = self.criterion(w_pred, w)
                test_running_loss += test_loss.item()
                
            tepoch.set_postfix({'training loss': training_running_loss/n_batches_train,
                                'validation loss': test_running_loss/n_batches_test})
            
            # wandb log
            wandb.log({'Training Loss': training_running_loss/n_batches_train,
                       'Validation Loss': test_running_loss/n_batches_test})
            
        return network
        
    def run(self, sweep_cfg=None):   
        
        # NOTE: Initialize Network
        self.cc_input_size = self.n_state + (self.goal_horizon * 3 * 4)
        self.vc_input_size = self.n_state + 4  # phi, vx, vy, w
        
        self.output_size = self.n_action
        
        # Initialize policy network
        self.vc_network = self.initialize_network(input_size=self.vc_input_size, output_size=self.output_size, 
                                                    num_hidden_layer=sweep_cfg.num_hidden_layer, hidden_dim=sweep_cfg.hidden_dim,
                                                    batch_norm=True)
        
        self.cc_network = self.initialize_network(input_size=self.cc_input_size, output_size=self.output_size, 
                                                    num_hidden_layer=sweep_cfg.num_hidden_layer, hidden_dim=sweep_cfg.hidden_dim,
                                                    batch_norm=True)
        
        # NOTE: Load database
        self.database = Database(limit=self.cfg.database_size, norm_input=self.normalize_policy_input)
        filename = '/home/atari_ws/data/behavior_cloning/trot/parameter_sweep/dataset/database_0.hdf5'
        self.database.load_saved_database(filename=filename)
        
        # NOTE: Train Policy
        print('=== Training VC Policy ===')
        self.database.set_goal_type('vc')
        self.vc_network = self.train_network(self.vc_network, batch_size=sweep_cfg.batch_size, learning_rate=sweep_cfg.learning_rate, n_epoch=sweep_cfg.epochs)
        
        print('=== Training CC Policy ===')
        self.database.set_goal_type('cc')
        self.cc_network = self.train_network(self.cc_network, batch_size=sweep_cfg.batch_size, learning_rate=sweep_cfg.learning_rate, n_epoch=sweep_cfg.epochs)
        
        
        # NOTE: setup pybullet environment
        # condition on which iterations to show GUI for Pybullet    
        display_simu = False
        
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
        v_des, w_des = utils.get_des_velocities(self.vx_des_max, self.vx_des_min, self.vy_des_max, self.vy_des_min, 
                                        self.w_des_max, self.w_des_min, gait, dist='uniform')
        
        # print selected gait and desired velocities
        print(f"-> gait: {gait} | v_des: {v_des} | w_des: {w_des}")
        
        
        # NOTE: Rollout benchmark MPC
        # set simulation start time to 0.0
        start_time = 0.0
        
        print("=== Benchmark MPC Rollout ===")
        
        # rollout mpc
        benchmark_state, benchmark_action, benchmark_vc_goal, benchmark_cc_goal, benchmark_base = \
            self.simulation.rollout_mpc(self.episode_length, start_time, v_des, w_des, gait, nominal=True)
        
        # collect position and velocity of nominal trajectory
        nominal_pos, nominal_vel = self.simulation.q_nominal, self.simulation.v_nominal
        
        # get contact plan of benchmark mpc
        contact_plan = self.simulation.gg.cnt_plan
            
        # calculate number of replannings
        num_replanning = int(self.simulation.gait_params.gait_period/self.simulation.plan_freq)
        
        vc_num_successful_rollouts = 0
        cc_num_successful_rollouts = 0
        
        # NOTE: Rollout Loop with Pertubations
        for i_replan in range(num_replanning):
            
            # NOTE: Get Jacobian from nominal trajectory and contact plan for pertubation
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
            for i_pertubation in range(2):
                
                # NOTE: Calculate Contact Conditioned Pertubation
                # perform pertubation until no foot is below the ground
                min_ee_height = .0
                while min_ee_height >= 0:
                    # sample pertubations (not yet contact conditioned)
                    perturbation_pos = np.concatenate((np.random.normal(self.mu_base_pos, self.sigma_base_pos, 3),\
                                                        np.random.normal(self.mu_base_ori, self.sigma_base_ori, 3), \
                                                        np.random.normal(self.mu_joint_pos, self.sigma_joint_pos, len(new_v0)-6)))
                    
                    perturbation_vel = np.random.normal(self.mu_vel, self.sigma_vel, len(new_v0))
                    
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
                
                
                # NOTE: Rollout VC
                print('=== VC ===') 
                self.database.set_goal_type('vc')
                
                start_i = int(start_time/self.sim_dt)
                desired_goal = np.zeros((self.episode_length - start_i, 4))
                for t in range(start_i, self.episode_length):
                    desired_goal[t-start_i, 0] = utils.get_phase_percentage(t, self.sim_dt, gait)
                
                desired_goal[:, 1] = np.full(np.shape(desired_goal[:, 1]), v_des[0])
                desired_goal[:, 2] = np.full(np.shape(desired_goal[:, 2]), v_des[1])
                desired_goal[:, 3] = np.full(np.shape(desired_goal[:, 3]), w_des)
                
                policy_state, policy_action, policy_vc_goal, policy_cc_goal, policy_base, _, _ = \
                    self.simulation.rollout_policy(self.episode_length, start_time, v_des, w_des, gait, 
                                                    self.vc_network, des_goal=desired_goal, q0=new_q0, v0=new_v0, 
                                                    norm_policy_input=self.database.get_database_mean_std())
                    
                print('Policy rollout completed. No. Datapoints: ' + str(len(policy_cc_goal)))      
                
                if len(policy_cc_goal) < (self.episode_length * 2 / 3):
                    print('Policy Datapoints too little! Policy will be considered as failed')
                else:
                    vc_num_successful_rollouts += 1
                    
                    print("=== Computing Goal Reaching Error ===")
                    vx_error, vy_error, w_error = utils.compute_vc_goal_reaching_error(v_des, w_des, policy_state[:, 0:2], policy_state[:, 5])
                    wandb.log({'vc_vx_error': vx_error, 'vc_vy_error': vy_error, 'vc_w_error': w_error})


                # NOTE: Rollout CC
                print('=== CC ===') 
                self.database.set_goal_type('cc')
                
                pin_robot, urdf_path = self.simulation.pin_robot, self.simulation.urdf_path
                n_eef = len(self.simulation.f_arr)
                start_i = int(start_time/self.sim_dt)
                
                # Create desired contact schedule with chosen gait and desired velocity
                desired_contact_schedule, _ = self.create_desired_contact_schedule(pin_robot, urdf_path, new_q0, new_v0, v_des, w_des, gait, start_time)

                # Calculate estimated center of mass of robot given the desired velocity
                estimated_com = utils.get_estimated_com(pin_robot, new_q0, new_v0, v_des, self.episode_length + 1, self.sim_dt, utils.get_plan(gait))
                
                # Construct desired goal
                desired_goal = utils.construct_cc_goal(self.episode_length + 1, n_eef, desired_contact_schedule, estimated_com, 
                                                    goal_horizon=self.goal_horizon, sim_dt=self.sim_dt, start_step=start_i)
                
                policy_state, policy_action, policy_vc_goal, policy_cc_goal, policy_base, _, _ = \
                    self.simulation.rollout_policy(self.episode_length, start_time, v_des, w_des, gait, 
                                                    self.cc_network, des_goal=desired_goal, q0=new_q0, v0=new_v0, 
                                                    norm_policy_input=self.database.get_database_mean_std())
                    
                print('Policy rollout completed. No. Datapoints: ' + str(len(policy_cc_goal)))      
                
                if len(policy_cc_goal) < (self.episode_length * 2 / 3):
                    print('Policy Datapoints too little! Policy will be considered as failed')
                else:
                    cc_num_successful_rollouts += 1
                    
                    print("=== Computing Goal Reaching Error ===")
                    vx_error, vy_error, w_error = utils.compute_vc_goal_reaching_error(v_des, w_des, policy_state[:, 0:2], policy_state[:, 5])
                    wandb.log({'cc_vx_error': vx_error, 'cc_vy_error': vy_error, 'cc_w_error': w_error})
      
        wandb.log({'num_successful_rollouts': vc_num_successful_rollouts+cc_num_successful_rollouts,
                   'vc_num_successful_rollouts': vc_num_successful_rollouts,
                   'cc_num_successful_rollouts': cc_num_successful_rollouts})
        
@hydra.main(config_path='cfgs', config_name='parameter_sweep_config')
def main(cfg):
    wandb.init(project=project_name)
    icc = SweepPolicy(cfg)
    icc.run(wandb.config) 
    

project_name = 'behavior_cloning_sg_sweep'

if __name__ == '__main__':
    main()
    
        


    


