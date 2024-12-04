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
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# login to wandb
wandb.login()

project_name = 'bc_benchmark_3_diff'


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
        
        # Evaluation
        self.num_rollouts_eval = cfg.num_rollouts_eval
        self.num_pertubations_per_replanning_eval = cfg.num_pertubations_per_replanning_eval
        
        # set pytorch device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Nvidia GPU availability is ' + str(torch.cuda.is_available()))
        
        # init simulation class
        self.simulation = Simulation(cfg=self.cfg)
    
    def load_network(self, filename=None, goal_type='vc'):
        """load trained network

        Args:
            filename (_type_, optional): file path. Defaults to None.
            goal_type (_type_, optional): type of goal conditioning. can be velocity (vc) or contact(cc) conditioned. Defaults to None.

        Raises:
            FileNotFoundError: if filepath provided is invalid
        """        
         
        if len(filename) == 0:
            raise FileNotFoundError()
        file = torch.load(filename, map_location=self.device)
                
        if goal_type == 'vc':
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
        Run evaluation
        '''
        
        network_dir = '/home/atari_ws/data/behavior_cloning/trot/bc_benchmark_3/network'
        files = os.listdir(network_dir)
        
        # NOTE: Iterate over networks from each training iteration
        for file in files:
            
            database_size = int(file[10:-4])
            
            print('=== Evaluating Network from database size ' + str(database_size) + ' ===')
            
            # NOTE: load trained networks
            self.load_network(filename=os.path.join(network_dir, file), goal_type='vc')
            
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
            
            # NOTE: Main evaluation loop
            # equal sampling for desired velocity (dont do it randomly to be consistent in evaluation)
            vx_des_list = np.linspace(self.vx_des_min, self.vx_des_max, num=self.num_rollouts_eval)
            
            # for loop for number of evaluation rollouts
            for i_main in range(self.num_rollouts_eval):
                
                # for loop for the different gaits
                for gait in self.gaits:

                    # get desired velocities from probability distribution
                    # v_des, w_des = utils.get_des_velocities(self.vx_des_max, self.vx_des_min, self.vy_des_max, self.vy_des_min, 
                    #                                 self.w_des_max, self.w_des_min, gait, dist='uniform')
                    
                    v_des = np.array([vx_des_list[i_main], 0.0, 0.0])
                    w_des = 0.0
                    
                    # print selected gait and desired velocities
                    print(f"-> gait: {gait} | v_des: {v_des} | w_des: {w_des}")
                    
                    
                    # NOTE: Rollout benchmark MPC
                    # set simulation start time to 0.0
                    start_time = 0.0
                    
                    print("=== Benchmark MPC Rollout ===")
                    
                    # rollout mpc
                    benchmark_state, benchmark_action, benchmark_vc_goal, benchmark_cc_goal, benchmark_base, _ = \
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
                        for i_pertubation in range(self.num_pertubations_per_replanning_eval):
                            
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
                                    
                                    
                            wandb.init(project=project_name, config={'database_size': database_size, 'gait':gait, 'vx_des':v_des[0]}, name=str(i_main)+'_'+str(i_replan)+'_'+str(i_pertubation))                        
                            
                            # NOTE: Make desired goal
                            # the goals are cade from start time 0.0, as the policy rollout will choose the correct start time.

                            start_i = int(start_time/self.sim_dt)
                            desired_goal = np.zeros((self.episode_length - start_i, 5))
                            for t in range(start_i, self.episode_length):
                                desired_goal[t-start_i, 0] = utils.get_phase_percentage(t, self.sim_dt, gait)
                            
                            desired_goal[:, 1] = np.full(np.shape(desired_goal[:, 1]), v_des[0])
                            desired_goal[:, 2] = np.full(np.shape(desired_goal[:, 2]), v_des[1])
                            desired_goal[:, 3] = np.full(np.shape(desired_goal[:, 3]), w_des)
                            desired_goal[:, 4] = np.full(np.shape(desired_goal[:, 4]), utils.get_vc_gait_value(gait))
                                                
                            
                            # NOTE: MPC Rollout with Pertubation
                            ### perform rollout
                            print("=== Policy Rollout with pertubation - ", str(i_main+1), '_', str(i_replan+1), '_', str(i_pertubation), " ===")
                            
                            network = self.vc_network
                            policy_input_parameters = self.vc_policy_input_parameters
                            
                            policy_state, policy_action, policy_vc_goal, policy_cc_goal, policy_base, _, _, frames = \
                            self.simulation.rollout_policy(self.episode_length, start_time, v_des, w_des, gait, 
                                                            network, des_goal=desired_goal, q0=new_q0, v0=new_v0, 
                                                            norm_policy_input=policy_input_parameters, save_video=True)
                            
                            print('Policy rollout completed. No. Datapoints: ' + str(len(policy_cc_goal))) 
                            
                            
                            # Transpose the array to (time, channel, height, width) then log video
                            if len(frames) != 0:
                                video_array_transposed = np.array(frames).transpose(0, 3, 1, 2)
                                wandb.log({'video': wandb.Video(video_array_transposed, fps=self.simulation.video_fr)}) 
                            
                            if len(policy_cc_goal) < (self.episode_length * 2 / 3):
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
                                        
                                        
                                # NOTE: Compute Goal Reaching Error
                                print("=== Computing Velocity Reaching Error ===")
                                
                                vx_error, vy_error, w_error = utils.compute_vc_mse(v_des, w_des, policy_state[:, 0:2], policy_state[:, 5])
                                wandb.log({'vx_mse': vx_error, 'vy_mse': vy_error, 'w_mse': w_error})
                                
                                
                            wandb.finish()            
                
        
@hydra.main(config_path='cfgs', config_name='bc_config')
def main(cfg):
    icc = BehavioralCloning(cfg) 
    icc.run() 

if __name__ == '__main__':
    main()
    
        


    


