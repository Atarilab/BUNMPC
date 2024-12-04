import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from simulation import Simulation
from contact_planner import ContactPlanner
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
import wandb

# set random seet for reproducability
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# login to wandb
# wandb.login()


class BehavioralCloning():  

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
        
        # Policy Network Properties
        self.goal_type = cfg.goal_type
        if self.goal_type not in ['cc', 'vc']:
            raise ValueError('goal type should be cc or vc only!')
        
        if self.goal_type == 'cc':
            self.input_size = self.n_state + (self.goal_horizon * 3 * 4)
        elif self.goal_type == 'vc':
            self.input_size = self.n_state + 4  # phi, vx, vy, w
        
        self.output_size = self.n_action
        
        self.network = None
        self.criterion = nn.L1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Nvidia GPU availability is ' + str(torch.cuda.is_available()))
        
        # Training properties
        self.n_epoch = cfg.n_epoch  # per iteration
        self.batch_size = cfg.batch_size
        self.n_train_frac = cfg.n_train_frac
        self.learning_rate = cfg.learning_rate
        
        # init simulation class
        self.simulation = Simulation(cfg=self.cfg)
    
    
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
    
        
    def load_network(self, filename=None, goal_type=None):
        """
        load policy network and determine input and output sizes
        """    
        if len(filename) == 0:
            raise FileNotFoundError()
        file = torch.load(filename, map_location=self.device)
        
        if goal_type == 'cc':
            self.cc_network = file['network'].to(self.device)
            self.cc_network.eval()
            self.cc_policy_input_parameters = file['norm_policy_input']

            print("CC Policy Network loaded from: " + filename)
            if self.cc_policy_input_parameters is None:
                print('Policy Input will NOT be normalized')
            else:
                print('Policy Input will be normalized')
                
        elif goal_type == 'vc':
            self.vc_network = file['network'].to(self.device)
            self.vc_network.eval()
            self.vc_policy_input_parameters = file['norm_policy_input']

            print("VC Policy Network loaded from: " + filename)
            if self.vc_policy_input_parameters is None:
                print('Policy Input will NOT be normalized')
            else:
                print('Policy Input will be normalized')
        
         
    def run(self):   
        '''
        Run behavioral cloning
        '''
        
        self.load_network(filename='/home/atari_ws/data/behavior_cloning/trot/bc_single_gait_multi_goal/network/cc_policy.pth', goal_type='cc')
        self.load_network(filename='/home/atari_ws/data/behavior_cloning/trot/bc_single_gait_multi_goal/network/vc_policy.pth', goal_type='vc')
        
        
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
            
        for i_main in range(1):
            
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
                for i_pertubation in range(5):
                    
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
                            
                    
                    # loop between goal type: 
                    for gt in ['cc', 'vc']:
                        print('=== ' + gt + ' ===')                
                        
                        # NOTE: Make desired goal
                        # the goals are cade from start time 0.0, as the policy rollout will choose the correct start time.
                        if gt == 'vc':
                            start_i = int(start_time/self.sim_dt)
                            desired_goal = np.zeros((self.episode_length - start_i, 5))
                            for t in range(start_i, self.episode_length):
                                desired_goal[t-start_i, 0] = utils.get_phase_percentage(t, self.sim_dt, gait)
                            
                            desired_goal[:, 1] = np.full(np.shape(desired_goal[:, 1]), v_des[0])
                            desired_goal[:, 2] = np.full(np.shape(desired_goal[:, 2]), v_des[1])
                            desired_goal[:, 3] = np.full(np.shape(desired_goal[:, 3]), w_des)
                            desired_goal[:, 4] = np.full(np.shape(desired_goal[:, 4]), utils.get_vc_gait_value(gait))
                        
                        elif gt == 'cc':
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
                        
                        
                        
                        # NOTE: MPC Rollout with Pertubation
                        ### perform rollout
                        print("=== Policy Rollout with pertubation - ", str(i_main+1), '_', str(i_replan+1), '_', str(i_pertubation), " ===")

                        
                        if gt == 'vc':
                            network = self.vc_network
                            policy_input_parameters = self.vc_policy_input_parameters
                            
                            policy_state, policy_action, policy_vc_goal, policy_cc_goal, policy_base, _, _, frames = \
                            self.simulation.rollout_policy(self.episode_length, start_time, v_des, w_des, gait, 
                                                            network, des_goal=desired_goal, q0=new_q0, v0=new_v0, 
                                                            norm_policy_input=policy_input_parameters, save_video=True)
                            
                        elif gt == 'cc':
                            network = self.cc_network
                            policy_input_parameters = self.cc_policy_input_parameters
                            
                            policy_state, policy_action, policy_vc_goal, policy_cc_goal, policy_base, _, _, frames = \
                            self.simulation.rollout_policy_with_cc_replanning(self.episode_length, start_time, v_des, w_des, gait, 
                                                            network, q0=new_q0, v0=new_v0, 
                                                            norm_policy_input=policy_input_parameters, save_video=False)
                        
                        
                        
                        print('Policy rollout completed. No. Datapoints: ' + str(len(policy_cc_goal)))      
                        
                        if len(policy_cc_goal) < (self.episode_length * 2 / 3):
                            print('Policy Datapoints too little! Policy will be considered as failed')

                        else:                                    
                                    
                            # NOTE: Compute Goal Reaching Error
                            print("=== Computing Goal Reaching Error ===")
                            
                            vx_error, vy_error, w_error = utils.compute_vc_mse(v_des, w_des, policy_state[:, 0:2], policy_state[:, 5])
  
                
        
@hydra.main(config_path='cfgs', config_name='bc_config')
def main(cfg):
    icc = BehavioralCloning(cfg) 
    icc.run() 

if __name__ == '__main__':
    main()
    
        


    


