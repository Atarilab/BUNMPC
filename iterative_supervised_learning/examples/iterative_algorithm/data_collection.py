#########################################################################
# This script is used to collect data for training using behavior cloning
# The config file can be found in cfgs/data_collection_config.yaml
#########################################################################

import torch
from omegaconf import OmegaConf
from simulation import Simulation
import utils
import pinocchio as pin
from database import Database
import numpy as np
import random
import hydra
import os
from datetime import datetime
import h5py
import pickle
import sys
import time

## Comment out if using workstation. this will cause errors
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

## set random seet for reproducability
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)


class DataCollection():  

    def __init__(self, cfg):        
        # configuration file (containing the hyper/parameters)
        self.cfg = cfg
            
        # Simulation rollout properties
        self.episode_length = cfg.episode_length
        self.sim_dt = cfg.sim_dt
        
        # MPC rollout pertubations
        self.mu_base_pos, self.sigma_base_pos = cfg.mu_base_pos, cfg.sigma_base_pos # base position
        self.mu_joint_pos, self.sigma_joint_pos = cfg.mu_joint_pos, cfg.sigma_joint_pos # joint position
        self.mu_base_ori, self.sigma_base_ori = cfg.mu_base_ori, cfg.sigma_base_ori # base orientation
        self.mu_vel, self.sigma_vel = cfg.mu_vel, cfg.sigma_vel # joint velocity
        
        # Model Parameters
        self.action_type = cfg.action_type
        self.normalize_policy_input = cfg.normalize_policy_input
        
        # Iterations
        self.n_iteration = cfg.n_iteration
        self.num_pertubations_per_replanning = cfg.num_pertubations_per_replanning
        
        print('number of iterations: ' + str(self.n_iteration))
        print('number of pertubations per positon: ' + str(self.num_pertubations_per_replanning))
        max_dataset_size = self.n_iteration * 10 * self.num_pertubations_per_replanning * self.episode_length
        print('extimated dataset size: ' + str(max_dataset_size))
        
        # Desired Motion Parameters
        self.gaits = cfg.gaits
        self.vx_des_min, self.vx_des_max = cfg.vx_des_min, cfg.vx_des_max
        self.vy_des_min, self.vy_des_max = cfg.vy_des_min, cfg.vy_des_max
        self.w_des_min, self.w_des_max = cfg.w_des_min, cfg.w_des_max
        
        # init simulation class
        self.simulation = Simulation(cfg=self.cfg)
        
        # define log file name
        str_gaits = ''
        for gait in self.gaits:
            str_gaits = str_gaits + gait
        self.str_gaits = str_gaits
        
        current_date = datetime.today().strftime("%b_%d_%Y_")
        current_time = datetime.now().strftime("%H_%M_%S")

        save_path_base = "/behavior_cloning/" + str_gaits
        if cfg.suffix != '':
            save_path_base += "_"+cfg.suffix
        save_path_base += "/" +  current_date + current_time
        
        self.data_save_path = self.cfg.data_save_path + save_path_base
        self.dataset_savepath = self.data_save_path + '/dataset'
        
        # Declare Database
        self.database = Database(limit=cfg.database_size) # goal type and normalize input does not need to be set, as no training is done here
    
        
    def save_dataset(self, iter):
        """save the current database as h5py file and the config as pkl

        Args:
            iter (int): the current iteration
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
            hf.create_dataset('cc_goals', data=self.database.cc_goals[:data_len])
            hf.create_dataset('actions', data=self.database.actions[:data_len]) 
        
        # save config as pickle only once
        if os.path.exists(self.dataset_savepath + "/config.pkl") == False:
            # convert hydra cfg to config dict
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)
            
            f = open(self.dataset_savepath + "/config.pkl", "wb")
            pickle.dump(config_dict, f)
            f.close()
        
        print("saved dataset at iteration " + str(iter))
        
        
    def run(self):   
        """run the data collection process
        """            
        
        # NOTE: Main Iteration Loop
        for i_main in range(self.n_iteration):
            
            # condition on which iterations to show GUI for Pybullet    
            display_simu = False
            
            # init env for if no pybullet server is active
            if self.simulation.currently_displaying_gui is None:
                self.simulation.init_pybullet_env(display_simu=display_simu)
            # change pybullet environment between with/without display, depending on condition
            elif display_simu != self.simulation.currently_displaying_gui:
                self.simulation.kill_pybullet_env()
                self.simulation.init_pybullet_env(display_simu=display_simu)
                
            print(f"============ Iteration {i_main+1}  ==============")
                
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
            benchmark_state, benchmark_action, benchmark_vc_goal, benchmark_cc_goal, benchmark_base, _ = \
                self.simulation.rollout_mpc(self.episode_length, start_time, v_des, w_des, gait, nominal=True)
            
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
                for i_pertubation in range(self.num_pertubations_per_replanning):
                    
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
                    
                    # NOTE: MPC Rollout with Perturbation
                    ### perform rollout
                    print("=== MPC Rollout with perturbation - nom. traj. pos. ", str(i_replan+1), " perturbation ", str(i_pertubation + 1), " ===")
                    mpc_state, mpc_action, mpc_vc_goal, mpc_cc_goal, mpc_base, _ = self.simulation.rollout_mpc(self.episode_length, start_time, v_des, w_des, gait, 
                                                                                        nominal=True, q0=new_q0, v0=new_v0)
                    
                    print('MPC rollout completed. No. Datapoints: ' + str(len(mpc_cc_goal)))      
                    
                    if len(mpc_state) != 0:
                        self.database.append(mpc_state, mpc_action, vc_goals=mpc_vc_goal, cc_goals=mpc_cc_goal)
                        print("MPC data saved into database")
                        print('database size: ' + str(len(self.database)))    
                    else:
                        print('mpc rollout failed')
            
            # Save Database   
            self.save_dataset(iter=len(self.database))

@hydra.main(config_path='cfgs', config_name='data_collection_config')
def main(cfg):
    dc = DataCollection(cfg)
    dc.run() 

if __name__ == '__main__':
    main()
    
        


    


