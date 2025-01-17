import sys
sys.path.append('../')

import time
import numpy as np
import pandas as pd
import pinocchio as pin
import pybullet
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from mpc.abstract_cyclic_gen import SoloMpcGaitGen
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController
import utils
import torch
from contact_planner import ContactPlanner
import cv2
from datetime import datetime
import os
from PIL import Image, ImageFont, ImageDraw


class Simulation():
    def __init__(self, cfg=None):
        """Init simulation class

        Args:
            cfg (_type_, optional): hydra config file. Defaults to None.
        """        
        
        self.cfg = cfg
        
        # robot config and init
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.urdf_path = Solo12Config.urdf_path
        self.f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]

        ### specifications for data collection and learning
        self.kp = cfg.kp
        self.kd = cfg.kd
        self.sim_dt = cfg.sim_dt
        
        ### MPC specific parameters ###
        # MPC simulation timing
        self.plan_freq = 0.05 # sec
        self.update_time = 0 # sec (time of lag)
        self.lag = 0
        self.index = 0
        ### END ###

        # Video recording
        self.video_dir = "/home/atari_ws/video"
        self.video_fr, self.video_width, self.video_height = 30.0, 640, 480
        
        os.makedirs(self.video_dir, exist_ok=True)

        # Noise added to the measured base states
        self.dq_pos = np.random.normal(0., .01, 3) # noise to base height
        self.dq_ori = np.random.normal(0., .01, 4) # noise to base orientation
        self.dq_joint = np.random.normal(0., .01, 12) # noise to base orientation
        self.dv_pos = np.random.normal(0., 0.15, 6) # noise to base velocity
        self.dv_joint = np.random.normal(0., 0.15, 12) # noise to base velocity
        
        # external disturbance
        self.f_ext = [0., 0., 0.]
        self.m_ext = [0., 0., 0.]
        self.t_dist = [0., 1.]
        
        # initialize inverse dynamics controller
        self.robot_id_ctrl = InverseDynamicsController(self.pin_robot, self.f_arr)
        
        ### Spawn robot in pybullet environment ###
        # robot initial configurations and positions
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.q0[0:2] = 0.0
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        
        # self.init_pybullet_env(display_simu=False)
        self.currently_displaying_gui = None
        
    def init_pybullet_env(self, display_simu=False):
        """start pybullet environment

        Args:
            display_simu (bool, optional): boolean if to display GUI or use DIRECT (No GUI). Defaults to False.
        """    
        # set current flag to request    
        self.currently_displaying_gui = display_simu
        
        # set appropriate server
        if display_simu:
            server = pybullet.GUI
            print('Initializing Pybullet environment with GUI')
        else:
            server = pybullet.DIRECT
            print('Initializing Pybullet environment with DIRECT')
            
        # update pinocchio
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, self.q0, self.v0)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
        
        # initialize pybullet server
        self.pybullet_env = PyBulletEnv(Solo12Robot, self.q0, self.v0, server=server)
        
        # set pybullet view angle and interface style
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        pybullet.resetDebugVisualizerCamera(1, 75, -20, (0.5, .0, 0.)) 
         
    def kill_pybullet_env(self):
        """Kill current pybullet environment
        """        
        print('Killing pybullet environment')
        pybullet.disconnect(self.pybullet_env.env.physics_client)
        
    def create_mp4_video(self, video_path, frames):
        """combine captured frames into mp4 video

        Args:
            video_path (_type_): file path to save video
            frames (_type_): captured frames
        """        
        # write frames into temp mp4 video with cv2
        out = cv2.VideoWriter('_temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.video_fr, (self.video_width, self.video_height))
        for frame in frames:
            out.write(frame)
        out.release()
        
        # convert temp video to h264 encoding to view in web
        os.system(f"ffmpeg -i _temp.mp4 -vcodec libx264 -f mp4 {video_path}")
        
        # remove temp video
        os.remove('_temp.mp4')
        print('Rollout Video saved')
        
    def calc_best_cam_pos(self, q):
        """move camera position depending on robot base position

        Args:
            q (_type_): current robot configuration

        Returns:
            cam_dist, cam_yaw, cam_pitch, target
        """      
        # default values  
        cam_dist = 1.0
        cam_yaw = 75
        cam_pitch = -20
        
        # change camera aim target base on robot base positon
        target = [0.5 + q[0], q[1], 0.]
        
        # change camera position in pybullet debug if using GUI
        pybullet.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, target) 
        
        return cam_dist, cam_yaw, cam_pitch, target
    
    def base_wrt_foot(self, q):
        """Calculate relative x, y distance of robot base frame from end effector

        Args:
            q (_type_): current robot configuration

        Returns:
            out: [x, y] * number of end effectors
        """    
        # initilize output array    
        out = np.zeros(2*len(self.f_arr))
        
        # loop for each end effector
        for i in range(len(self.f_arr)):
            # get translation of end effector from origin frame
            foot = self.pin_robot.data.oMf[self.pin_robot.model.getFrameId(self.f_arr[i])].translation
            # get relative distance of robot base frame from end effector
            out[2*i:2*(i+1)] = q[0:2] - foot[0:2]
            
        return out

    def phase_percentage(self, t:int):
        """get current gait phase percentage based on gait period

        Args:
            t (int): current sim step (NOT sim time!)

        Returns:
            phi: current gait phase. between 0 - 1
        """        
        phi = ((t*self.sim_dt) % self.gait_params.gait_period)/self.gait_params.gait_period
        return phi
    
    def failed_states(self, q, gait, time_elapsed, fail_angle=30):
        """check if robot has entered a failed state

        Args:
            q (_type_): current robot configuration
            gait (str): desired gait
            time_elapsed (int): time into simulation
            fail_angle (int): minimum roll and pitch angle of robot base which to be considered failed. default to 30

        Returns:
            sim_failed: True if failed
        """        
        
        # make sure that the policy does not count as fail directly due to the initial condition!
        if time_elapsed > (self.gait_params.gait_period / self.sim_dt):
            # get robot euler orientation
            rpy = utils.quaternion_to_euler_angle(q[3], q[4], q[5], q[6])
            
            # be more laxist when the gait is jump or bound
            if gait=='jump' or gait == 'bound': 
                # constraint allowed robot base height and roll/pitch angle
                if q[2]<.05 or q[2]>2.0 or abs(rpy[0])>fail_angle or abs(rpy[1])>fail_angle:
                    return True
                else:
                    return False
            # constraint allowed robot base height and roll/pitch angle
            elif q[2]<.1 or q[2]>2.0 or abs(rpy[0])>fail_angle or abs(rpy[1])>fail_angle:
                return True
            else:
                return False
        else:
            return False
            
    def safedagger_state_is_dangerous(self, q, gait, bounds_dict=None):
        """check if robot has entered a dangerous state

        Args:
            q (_type_): current robot configuration
            gait (str): desired gait
            bounds_dict (_type_, optional): dictionary of safety bounds. Defaults to None.

        Returns:
            True if dangerous. False otherwise
        """        
        # Use default if no bounds are given:
        if bounds_dict is None:
            bounds_dict = {
                'z_height': [0.15, 1.0],
                'body_angle': 25,
                'HAA_L': [-0.8, 1.5],
                'HAA_R': [-1.5, 0.8],
                'HFE_F': [-2.0, 2.0],                
                'HFE_B': [-2.0, 2.0],
                'KFE_F': [-3.0, 3.0],
                'KFE_B': [-3.0, 3.0]
            }
        
        ### Check robot base position and orientation    
        # get robot euler orientation
        rpy = utils.quaternion_to_euler_angle(q[3], q[4], q[5], q[6])
        
        # be more laxist when the gait is jump or bound
        if gait=='jump' or gait == 'bound': 
            # constraint allowed robot base height and roll/pitch angle
            if q[2]<bounds_dict['z_height'][0] or q[2]>bounds_dict['z_height'][1] or \
                abs(rpy[0])>bounds_dict['body_angle'] or abs(rpy[1])>bounds_dict['body_angle']:
                return True
            
        # constraint allowed robot base height and roll/pitch angle
        else:
            if q[2]<bounds_dict['z_height'][0] or q[2]>bounds_dict['z_height'][1] or \
                abs(rpy[0])>bounds_dict['body_angle'] or abs(rpy[1])>bounds_dict['body_angle']:
                return True
        
        ### check joint limits
        ## FL
        if q[7] > bounds_dict['HAA_L'][1] or q[7] < bounds_dict['HAA_L'][0]:
            return True
        elif q[8] > bounds_dict['HFE_F'][1] or q[8] < bounds_dict['HFE_F'][0]:
            return True
        elif q[9] > bounds_dict['KFE_F'][1] or q[9] < bounds_dict['KFE_F'][0]:
            return True
        
        ## FR
        elif q[10] > bounds_dict['HAA_R'][1] or q[10] < bounds_dict['HAA_R'][0]:
            return True
        elif q[11] > bounds_dict['HFE_F'][1] or q[11] < bounds_dict['HFE_F'][0]:
            return True
        elif q[12] > bounds_dict['KFE_F'][1] or q[12] < bounds_dict['KFE_F'][0]:
            return True
        
        ## BL
        elif q[13] > bounds_dict['HAA_L'][1] or q[13] < bounds_dict['HAA_L'][0]:
            return True
        elif q[14] > bounds_dict['HFE_B'][1] or q[14] < bounds_dict['HFE_B'][0]:
            return True
        elif q[15] > bounds_dict['KFE_B'][1] or q[15] < bounds_dict['KFE_B'][0]:
            return True
        
        ## BR
        elif q[16] > bounds_dict['HAA_R'][1] or q[16] < bounds_dict['HAA_R'][0]:
            return True
        elif q[17] > bounds_dict['HFE_B'][1] or q[17] < bounds_dict['HFE_B'][0]:
            return True
        elif q[18] > bounds_dict['KFE_B'][1] or q[18] < bounds_dict['KFE_B'][0]:
            return True
        
        # return false if none is true
        return False

    def new_ee_contact(self, ee_cur, ee_pre):
        """check if the end effector contact has moved from previous contact

        Args:
            ee_cur (_type_): current ee contact
            ee_pre (_type_): previous ee contact

        Returns:
            new_contact: list of ee contacts
        """        
        new_contact = []
        threshold = 1e-12
        for i in range(len(self.f_arr)):
            if abs(ee_cur[i][0])>threshold and abs(ee_pre[i][0])<=threshold:
                new_contact = np.hstack((new_contact,i))
        return new_contact

    def create_desired_contact_schedule(self, pin_robot, urdf_path, q0, v0, v_des, w_des, gait, start_time, episode_length):
        """create contact schedule for a desired robot velocity and gait.

        Args:
            pin_robot (_type_): pinocchio robot model
            urdf_path (_type_): robot urdf path
            q0 (_type_): robot initial configuration
            v0 (_type_): robot initial velocity
            v_des (_type_): desired translational velocity of robot com
            w_des (_type_): desired yaw of robot CoM
            gait (_type_): desired gait
            start_time (_type_): desired simulation start time (in sim time NOT sim step!)
            episode_length (_type_): imulation episode length (in sim step NOT sim time!)

        Returns:
            contact_schedule (np.array): [n_eff x number of contact events x (time, x, y, z)]
            cnt_plan (np.array): [planning horizon x n_eff x (in contact?, x, y, z)]
        """          

        plan = utils.get_plan(gait)
        cp = ContactPlanner(plan)
        contact_schedule, cnt_plan = cp.get_contact_schedule(pin_robot, urdf_path, q0, v0, v_des, w_des, episode_length, start_time)
        return contact_schedule, cnt_plan

    def rollout_mpc(self, episode_length, start_time, v_des, w_des, gait, q0=None, v0=None,
                    nominal=True, uneven_terrain = False, save_video = False, add_noise = False):
        """Rollout robot with BiConMP

        Args:
            episode_length (int): simulation episode length (in sim step NOT sim time!)
            start_time (_type_): simulation start time (in sim time NOT sim step!)
            v_des (_type_): desired CoM translational velocity
            w_des (_type_): desired CoM yaw
            gait (_type_): desired gait
            q0 (_type_, optional): initial robot configuration. Defaults to None.
            v0 (_type_, optional): initial robot velocity. Defaults to None.
            nominal (bool, optional): set if nominal trajectory is to be saved and returned. Defaults to True.
            uneven_terrain (bool, optional): set if simulation environment should have uneven terrain. Defaults to False.
            save_video (bool, optional): set if a video of the simulation should be saved. Defaults to False.
            add_noise (bool, optional): set if noise should be added to robot state. Defaults to False.

        Returns:
            state_history: robot state history. [] if failed
            action_history: robot action history. [] if failed
            vc_goal_history: velocity conditioned goal history. [] if failed
            cc_goal_history: contact conditioned goal history. [] if failed
            base_history: robot base absolute coordinate history. [] if failed
            frames: captured frames
        """         
        
        # check if initial robot configuration is given
        if q0 is None:
            q0 = self.q0
        else:
            assert len(q0) == self.pin_robot.model.nq, 'size of q0 given not OK!'
        
        # check if initial robot velocity is given
        if v0 is None:
            v0 = self.v0
        else:
            assert len(v0) == self.pin_robot.model.nv, 'size of v0 given not OK!'
            
        x0 = np.concatenate([q0, pin.utils.zero(self.pin_robot.model.nv)])
        
        # set robot to initial position
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q0, v0)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)    
        
        # reset robot state
        self.pybullet_env.reset_robot_state(q0, v0) 
        
        # get gait files
        plan = utils.get_plan(gait)
        self.gait_params = plan
        
        # action type
        action_type = self.cfg.action_type
        
        # data related variables
        n_action = self.cfg.n_action
        n_state = self.cfg.n_state
        goal_horizon = self.cfg.goal_horizon
        
        # MPC: declare variables
        pln_ctr = 0
        index = 0
        sim_failed = False
        last_contact_switch = 0
        nominal_counter = 0
        sim_t = start_time

        # MPC: initialize MPC
        self.gg = SoloMpcGaitGen(self.pin_robot, self.urdf_path, x0, self.plan_freq, q0, None) 
        
        # MPC: set inverse kinematic gains
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)
        
        # MPC: update gait parameters for gait generation
        self.gg.update_gait_params(self.gait_params, sim_t)  # sim_t is actually not used in update_gait_params
        
        # generate uneven terrain if enabled
        if uneven_terrain:
            self.pybullet_env.generate_terrain()
        
        # if save video enabled
        frames = []
        if save_video:
            video_cnt = int(1.0 / (self.sim_dt * self.video_fr))
            current_date = datetime.today().strftime("%b_%d_%Y_")
            current_time = datetime.now().strftime("%H_%M_%S")
            
            
        # State variables
        state_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_state))
        base_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        com_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        
        # goal variables
        vc_goal_history = np.zeros((episode_length - int(start_time/self.sim_dt), 5))
        
        # variables for contacts
        ee_pos = np.zeros(len(self.f_arr)*3)
        pre_ee_pos = np.zeros((len(self.f_arr),3))
        new_contact_pos = []
        
        # Action variables
        if action_type == "structured":
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3*n_action))
        else:
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_action))
            
        # Nominal trajectory positions
        self.q_nominal = np.zeros((int(self.gait_params.gait_period / self.plan_freq),\
                        self.pin_robot.model.nq))
        self.v_nominal = np.zeros((int(self.gait_params.gait_period / self.plan_freq),\
                        self.pin_robot.model.nv)) 
        
                
        # NOTE: main simulation loop
        for o in range(int(start_time/self.sim_dt), episode_length):
            
            time_elapsed = o - int(start_time/self.sim_dt)
            
            # get current robot state
            q, v = self.pybullet_env.get_state()
            
            # capture frame
            if save_video:
                if o % video_cnt == 0:
                    cam_dist, cam_yaw, cam_pitch, target = self.calc_best_cam_pos(q)
                    image = self.pybullet_env.capture_image_frame(width=self.video_width, height=self.video_height, 
                                                                  cam_dist=cam_dist, cam_yaw=cam_yaw, cam_pitch=cam_pitch, target=target)
                    frames.append(image)
            
            # add sensor noise if enabled
            if add_noise:
                q[0:3] += self.dq_pos
                q[3:7] += self.dq_ori
                q[3:7] = q[3:7]/np.linalg.norm(q[3:7])
                q[7:] += self.dq_joint
                v[0:6] += self.dv_pos
                v[6:] += self.dv_joint
            
            # Perform forward kinematics and updates in pinocchio
            pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q, v)
            pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
            
            # collect state history
            base_history[time_elapsed, :] = q[0 : 3]
            com_history[time_elapsed, :] = pin.centerOfMass(self.pin_robot.model, self.pin_robot.data, q)
            
            state_history[time_elapsed, :self.pin_robot.model.nv] = v
            state_history[time_elapsed, self.pin_robot.model.nv:self.pin_robot.model.nv + 2*len(self.f_arr)] = self.base_wrt_foot(q)
            state_history[time_elapsed, self.pin_robot.model.nv + 2*len(self.f_arr):] = q[2:]
            
            # collect vc goal
            vc_goal_history[time_elapsed, 0] = self.phase_percentage(o)
            vc_goal_history[time_elapsed, 1:3] = v_des[0:2]
            vc_goal_history[time_elapsed, 3] = w_des
            vc_goal_history[time_elapsed, 4] = utils.get_vc_gait_value(gait)
                            
            # MPC: Planning Control. Replan motion
            if pln_ctr == 0:
                # run MPC
                xs_plan, us_plan, f_plan = self.gg.optimize(q, v, np.round(sim_t, 3), v_des, w_des)
                index = 0
                
                # collect nominal trajectory
                if nominal and (nominal_counter < int(self.gait_params.gait_period / self.plan_freq)):
                    self.q_nominal[nominal_counter] = q
                    self.v_nominal[nominal_counter] = v
                    nominal_counter += 1

            # evaluate MPC plan
            xs = xs_plan
            us = us_plan
            f = f_plan
            if(pd.isna(f).any()):
                sim_failed = True
                print('MPC diverged. MPC rollout failed')
                break

            q_des = xs[index][:self.pin_robot.model.nq].copy()
            dq_des = xs[index][self.pin_robot.model.nq:].copy()
            tau_ff, tau_fb = self.robot_id_ctrl.id_joint_torques(q, v, q_des, dq_des,\
            us[index], f[index])  # tau_fb is the kp kd gains of PD Policy

            # collect action history
            tau = tau_ff + tau_fb
            if action_type == "torque":
                action_history[time_elapsed,:] = tau
            elif action_type == "pd_target":
                action_history[time_elapsed,:] = (tau + self.kd * v[6:])/self.kp + q[7:]
            elif action_type == "structured":
                x_des = np.hstack((q_des[7:], dq_des[6:]))
                action_history[time_elapsed,:] = np.hstack((tau_ff, x_des))
                    
            # send joint commands to robot
            self.pybullet_env.send_joint_command(tau)

            # collect goal-related quantities          
            ee_pos, ee_force = self.pybullet_env.get_contact_positions_and_forces()
            
            # see if there are new end effector contacts with the ground
            new_ee = self.new_ee_contact(ee_pos, pre_ee_pos)
            pre_ee_pos[:] = ee_pos[:]
            
            # record new eef contacts as [id, time, x, y, z pos]
            if not o == 0: # we do not need the switch at initialization
                for ee in new_ee:
                    new_ee_entry = np.hstack((np.hstack((ee, o)), ee_pos[int(ee)]))
                    if len(new_contact_pos) == 0:
                        new_contact_pos = new_ee_entry
                    else:
                        new_contact_pos = np.vstack((new_contact_pos, new_ee_entry))
                        
                    pybullet.addUserDebugPoints([ee_pos[int(ee)]], [[1, 0, 0]], pointSize=5, lifeTime=3.0)

            # exert disturbance
            if o>self.t_dist[0] and o<self.t_dist[1]:
                self.pybullet_env.apply_external_force(self.f_ext, self.m_ext)

            # MPC: step in MPC time
            sim_t += self.sim_dt
            pln_ctr = int((pln_ctr + 1)%(self.plan_freq/self.sim_dt))
            index += 1
        
        # collect measured goal history if sim is successful
        if sim_failed is False:
            n_eff = len(self.f_arr)
            self.contact_schedule = utils.construct_contact_schedule(new_contact_pos, n_eff)
            cc_goal_history = utils.construct_cc_goal(episode_length, n_eff, self.contact_schedule, com_history, 
                                        goal_horizon=goal_horizon, sim_dt=self.sim_dt, start_step=int(start_time/self.sim_dt))

        # save simulation video
        # if save_video:
        #     video_path = self.video_dir + '/' + current_date + current_time + '.mp4'
            # self.create_mp4_video(video_path, frames)

        # return histories if the simulation does not diverge
        if not sim_failed:
            end_time = len(cc_goal_history)
            return state_history[0:end_time,:], action_history[0:end_time,:], vc_goal_history[0:end_time, :], cc_goal_history, base_history[0:end_time, :], frames
        else:
            return [], [], [], [], [], frames
      
    def rollout_policy(self, episode_length, start_time, v_des, w_des, gait, policy_network, des_goal,
                       q0=None, v0=None, norm_policy_input:list=None,
                       uneven_terrain = False, save_video = False, add_noise = False, return_robot_state_if_fail=False,
                       push_f=[0, 0, 0], push_t=2.0, push_dt=0.001, fail_angle=30): 
        """Rollout robot with policy network

        Args:
            episode_length (int): simulation episode length (in sim step NOT sim time!)
            start_time (_type_): simulation start time (in sim time NOT sim step!)
            v_des (_type_): desired CoM translational velocity
            w_des (_type_): desired CoM yaw
            gait (_type_): desired gait
            policy_network (_type_): Policy network used to control robot
            des_goal (_type_): desired goal input to policy network
            q0 (_type_, optional): initial robot configuration. Defaults to None.
            v0 (_type_, optional): initial robot velocity. Defaults to None.
            norm_policy_input (list, optional): input normalization parameters. Defaults to None.
            uneven_terrain (bool, optional): set if simulation environment should have uneven terrain. Defaults to False.
            save_video (bool, optional): set if a video of the simulation should be saved. Defaults to False.
            add_noise (bool, optional): set if noise should be added to robot state. Defaults to False.
            return_robot_state_if_fail (bool, optional): set if robot q and v should be returned even if robot sim fails. Defaults to False.
            push_f (list, optional): external push force. Defaults to [0, 0, 0].
            push_t (float, optional): external push time (like at 2s in a 5s simulation). Defaults to 2.0.
            push_dt (float, optional): external push dt (to create an impulse). Defaults to 0.001.
            fail_angle (int, optional): angle to consider robot as failed. Defaults to 30.

        Returns:
            state_history: robot state history. [] if failed
            action_history: robot action history. [] if failed
            vc_goal_history: velocity conditioned goal history. [] if failed
            cc_goal_history: contact conditioned goal history. [] if failed
            base_history: robot base absolute coordinate history. [] if failed
            q_history: robot q history. (only if return_robot_state_if_fail is True)
            v_history: robot v history. (only if return_robot_state_if_fail is True)
            frames: captured frames
        """        
          
        # check if initial robot configuration is given
        if q0 is None:
            q0 = self.q0
        else:
            assert len(q0) == self.pin_robot.model.nq, 'size of q0 given not OK!'
        
        # check if initial robot velocity is given
        if v0 is None:
            v0 = self.v0
        else:
            assert len(v0) == self.pin_robot.model.nv, 'size of v0 given not OK!'
        
        # set robot to initial position
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q0, v0)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)    
        
        # reset robot state
        self.pybullet_env.reset_robot_state(q0, v0) 
        
        # get gait files
        plan = utils.get_plan(gait)
        self.gait_params = plan
        
        # action type
        action_type = self.cfg.action_type
        
        # data related variables
        n_action = self.cfg.n_action
        n_state = self.cfg.n_state
        goal_horizon = self.cfg.goal_horizon
        
        # check if input normalization parameters are ok if given
        if norm_policy_input is not None:
            assert len(norm_policy_input) == 4, 'norm_policy_input should have the terms [state_mean, state_std, goal_mean, goal_std]' 
        
        # WATCHOUT: torch device double reclared!
        # Policy: Initialize Torch device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Policy: set network in evaluation mode
        policy_network.eval()
        
        # generate uneven terrain if enabled
        if uneven_terrain:
            self.pybullet_env.generate_terrain()
        
        # if save video enabled
        frames = []
        if save_video:
            video_cnt = int(1.0 / (self.sim_dt * self.video_fr))
            current_date = datetime.today().strftime("%b_%d_%Y_")
            current_time = datetime.now().strftime("%H_%M_%S")
            
        
        # State variables
        state_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_state))
        base_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        com_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        q_history, v_history = [], []
        
        # goal variables
        vc_goal_history = np.zeros((episode_length - int(start_time/self.sim_dt), 5))
        
        # variables for contacts
        ee_pos = np.zeros(len(self.f_arr)*3)
        pre_ee_pos = np.zeros((len(self.f_arr),3))
        new_contact_pos = []
        
        # Action variables
        if action_type == "structured":
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3*n_action))
        else:
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_action))
            
        # NOTE: main simulation loop
        for o in range(int(start_time/self.sim_dt), episode_length):
            
            time_elapsed = o - int(start_time/self.sim_dt)
            
            # get current robot state
            q, v = self.pybullet_env.get_state()
            
            # capture frames
            if save_video:
                if o % video_cnt == 0:
                    cam_dist, cam_yaw, cam_pitch, target = self.calc_best_cam_pos(q)
                    image = self.pybullet_env.capture_image_frame(width=self.video_width, height=self.video_height, 
                                                                  cam_dist=cam_dist, cam_yaw=cam_yaw, cam_pitch=cam_pitch, target=target)
                    frames.append(image)
            
            # record q and v history
            q_history.append(q)
            v_history.append(v)
            
            # add sensor noise if enabled
            if add_noise:
                q[0:3] += self.dq_pos
                q[3:7] += self.dq_ori
                q[3:7] = q[3:7]/np.linalg.norm(q[3:7])
                q[7:] += self.dq_joint
                v[0:6] += self.dv_pos
                v[6:] += self.dv_joint
            
            # Perform forward kinematics and updates in pinocchio
            pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q, v)
            pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

            # collect state history = current robot state
            base_history[time_elapsed, :] = q[0 : 3]
            com_history[time_elapsed, :] = pin.centerOfMass(self.pin_robot.model, self.pin_robot.data, q)
            
            state_history[time_elapsed, :self.pin_robot.model.nv] = v
            state_history[time_elapsed, self.pin_robot.model.nv:self.pin_robot.model.nv + 2*len(self.f_arr)] = self.base_wrt_foot(q)
            state_history[time_elapsed, self.pin_robot.model.nv + 2*len(self.f_arr):] = q[2:]

            # collect vc goal
            vc_goal_history[time_elapsed, 0] = self.phase_percentage(o)
            vc_goal_history[time_elapsed, 1:3] = v_des[0:2]
            vc_goal_history[time_elapsed, 3] = w_des
            vc_goal_history[time_elapsed, 4] = utils.get_vc_gait_value(gait)
                    
            # get state from state history
            state = state_history[time_elapsed:time_elapsed+1]
            
            # desired goal also considers start time!
            goal = des_goal[time_elapsed:time_elapsed+1, :]
            
            # normalize policy input if required
            if norm_policy_input is not None:
                # norm_policy_input => [state_mean, state_std, goal_mean, goal_std]
                state = (state - norm_policy_input[0]) / norm_policy_input[1]
                goal = (goal - norm_policy_input[2]) / norm_policy_input[3]
            
            # construct input
            input = np.hstack((state, goal))
            
            # forward pass on policy network
            action = policy_network(torch.from_numpy(input).to(device).float())
            action = action.detach().cpu().numpy().reshape(-1)
                
            # compute action using given action type
            if action_type =="torque":
                tau = action
                action_history[time_elapsed, :] = tau
                
            elif action_type == "pd_target":
                q_des = action
                tau = self.kp * (q_des - q[7:]) - self.kd * v[6:]
                # WATCHOUT: action history for pd target is q_des and not tau!!!
                action_history[time_elapsed, :] = q_des
                
            elif action_type == "structured":
                tau_ff = action[:self.pin_robot.model.nv - 6]
                q_des = action[self.pin_robot.model.nv - 6:2*(self.pin_robot.model.nv - 6)]
                dq_des = action[2*(self.pin_robot.model.nv - 6):3*(self.pin_robot.model.nv - 6)]
                
                tau = tau_ff + self.kp * (q_des - q[7:]) + self.kd * (dq_des- v[6:])
                x_des = np.hstack((q_des[7:], dq_des[6:]))
                action_history[time_elapsed,:] = np.hstack((tau_ff, x_des))
            
            # check if robot state failed
            sim_failed = self.failed_states(q, gait, time_elapsed, fail_angle=fail_angle)
            if sim_failed:
                print('Policy Rollout Failed at sim step ' + str(o))
                break

            # send joint commands to robot
            self.pybullet_env.send_joint_command(tau)

            # collect goal-related quantities          
            ee_pos, ee_force = self.pybullet_env.get_contact_positions_and_forces()
            
            # see if there are new end effector contacts with the ground
            new_ee = self.new_ee_contact(ee_pos, pre_ee_pos)
            pre_ee_pos[:] = ee_pos[:]
            
            # record new eef contacts as [id, time, x, y, z pos]
            if not o == 0: # we do not need the switch at initialization
                for ee in new_ee:
                    new_ee_entry = np.hstack((np.hstack((ee, o)), ee_pos[int(ee)]))
                    if len(new_contact_pos) == 0:
                        new_contact_pos = new_ee_entry
                    else:
                        new_contact_pos = np.vstack((new_contact_pos, new_ee_entry))
                        
                    pybullet.addUserDebugPoints([ee_pos[int(ee)]], [[1, 0, 0]], pointSize=5, lifeTime=3.0)

            # exert disturbance
            if o>=(push_t/self.sim_dt) and o<((push_t+push_dt)/self.sim_dt):
                self.pybullet_env.apply_external_force(push_f, [0, 0, 0])
            
        
        # collect measured goal history if sim is successful
        if sim_failed is False:
            n_eff = len(self.f_arr)
            self.contact_schedule = utils.construct_contact_schedule(new_contact_pos, n_eff)
            cc_goal_history = utils.construct_cc_goal(episode_length, n_eff, self.contact_schedule, com_history, 
                                        goal_horizon=goal_horizon, sim_dt=self.sim_dt, start_step=int(start_time/self.sim_dt))

        # save simulation video
        # if save_video:
        #     video_path = self.video_dir + '/' + current_date + current_time + '.mp4'
            # self.create_mp4_video(video_path, frames)

        # return histories if the simulation does not diverge
        if not sim_failed:
            end_time = len(cc_goal_history)
            return state_history[0:end_time,:], action_history[0:end_time,:], vc_goal_history[0:end_time, :], cc_goal_history, base_history[0:end_time, :], np.array(q_history), np.array(v_history), frames
        
        elif sim_failed and return_robot_state_if_fail:
            return [], [], [], [], [], np.array(q_history), np.array(v_history), frames
        
        else:
            return [], [], [], [], [], [], [], frames
        
    def rollout_policy_with_cc_replanning(self, episode_length, start_time, v_des, w_des, gait, policy_network,
                       q0=None, v0=None, norm_policy_input:list=None,
                       uneven_terrain = False, save_video = False, add_noise = False):   
        """(DOES NOT WORK FOR VELOCITY GOAL CONDITIONING!) Rollout robot with policy network with replanning of contact conditioning goal for stability

        Args:
            episode_length (_type_): simulation episode length (in sim step NOT sim time!)
            start_time (_type_): simulation start time (in sim time NOT sim step!)
            v_des (_type_): desired CoM translational velocity
            w_des (_type_): desired CoM yaw
            gait (_type_): desired gait
            policy_network (_type_): policy network
            q0 (_type_, optional): initial robot configuration. Defaults to None.
            v0 (_type_, optional): initial robot velocity. Defaults to None.
            norm_policy_input (list, optional): set if policy inputs should be normalized. Defaults to None.
            uneven_terrain (bool, optional): set if simulation environment should have uneven terrain. Defaults to False.
            save_video (bool, optional): set if a video of the simulation should be saved. Defaults to False.
            add_noise (bool, optional): set if noise should be added to robot state. Defaults to False.

        Returns:
            state_history: robot state history. [] if failed
            action_history: robot action history. [] if failed
            vc_goal_history: velocity conditioned goal history. [] if failed
            cc_goal_history: contact conditioned goal history. [] if failed
            base_history: robot base absolute coordinate history. [] if failed
            q_history: robot q history
            v_history: robot v history
            frames: captured frames
        """   
            
        # check if initial robot configuration is given
        if q0 is None:
            q0 = self.q0
        else:
            assert len(q0) == self.pin_robot.model.nq, 'size of q0 given not OK!'
        
        # check if initial robot velocity is given
        if v0 is None:
            v0 = self.v0
        else:
            assert len(v0) == self.pin_robot.model.nv, 'size of v0 given not OK!'
        
        # set robot to initial position
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q0, v0)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)    
        
        # reset robot state
        self.pybullet_env.reset_robot_state(q0, v0) 
        
        # get gait files
        plan = utils.get_plan(gait)
        self.gait_params = plan
        
        # action type
        action_type = self.cfg.action_type
        
        # data related variables
        n_action = self.cfg.n_action
        n_state = self.cfg.n_state
        goal_horizon = self.cfg.goal_horizon
        
        # check if input normalization parameters are ok if given
        if norm_policy_input is not None:
            assert len(norm_policy_input) == 4, 'norm_policy_input should have the terms [state_mean, state_std, goal_mean, goal_std]' 
        
        # Policy: Initialize Torch device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Policy: set network in evaluation mode
        policy_network.eval()
        
        # generate uneven terrain if enabled
        if uneven_terrain:
            self.pybullet_env.generate_terrain()
        
        # if save video enabled
        frames = []
        if save_video:
            video_cnt = int(1.0 / (self.sim_dt * self.video_fr))
            current_date = datetime.today().strftime("%b_%d_%Y_")
            current_time = datetime.now().strftime("%H_%M_%S")
            
        # State variables
        state_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_state))
        base_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        com_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        q_history, v_history = [], []
        
        # goal variables
        vc_goal_history = np.zeros((episode_length - int(start_time/self.sim_dt), 5))
        
        # variables for contacts
        ee_pos = np.zeros(len(self.f_arr)*3)
        pre_ee_pos = np.zeros((len(self.f_arr),3))
        new_contact_pos = []
        
        # Action variables
        if action_type == "structured":
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3*n_action))
        else:
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_action))
        
        # replanning variables
        pln_ctr = 0
        index = 0
        sim_t = start_time
            
        # NOTE: main simulation loop
        for o in range(int(start_time/self.sim_dt), episode_length):
            
            time_elapsed = o - int(start_time/self.sim_dt)
            
            # get current robot state
            q, v = self.pybullet_env.get_state()
            
            # capture frame
            if save_video:
                if o % video_cnt == 0:
                    cam_dist, cam_yaw, cam_pitch, target = self.calc_best_cam_pos(q)
                    image = self.pybullet_env.capture_image_frame(width=self.video_width, height=self.video_height, 
                                                                  cam_dist=cam_dist, cam_yaw=cam_yaw, cam_pitch=cam_pitch, target=target)
                    frames.append(image)
            
            # record q and v history       
            q_history.append(q)
            v_history.append(v)
            
            # add sensor noise if enabled
            if add_noise:
                q[0:3] += self.dq_pos
                q[3:7] += self.dq_ori
                q[3:7] = q[3:7]/np.linalg.norm(q[3:7])
                q[7:] += self.dq_joint
                v[0:6] += self.dv_pos
                v[6:] += self.dv_joint
            
            # Perform forward kinematics and updates in pinocchio
            pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q, v)
            pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

            # collect state history = current robot state            
            base_history[time_elapsed, :] = q[0 : 3]
            com_history[time_elapsed, :] = pin.centerOfMass(self.pin_robot.model, self.pin_robot.data, q)
            
            state_history[time_elapsed, :self.pin_robot.model.nv] = v
            state_history[time_elapsed, self.pin_robot.model.nv:self.pin_robot.model.nv + 2*len(self.f_arr)] = self.base_wrt_foot(q)
            state_history[time_elapsed, self.pin_robot.model.nv + 2*len(self.f_arr):] = q[2:]

            # collect vc goal
            vc_goal_history[time_elapsed, 0] = self.phase_percentage(o)
            vc_goal_history[time_elapsed, 1:3] = v_des[0:2]
            vc_goal_history[time_elapsed, 3] = w_des
            vc_goal_history[time_elapsed, 4] = utils.get_vc_gait_value(gait)
                    
            # get state from state history
            state = state_history[time_elapsed:time_elapsed+1]
            
            # NOTE: desired goal replanning
            if pln_ctr == 0:
                index = 0
                pin_robot, urdf_path = self.pin_robot, self.urdf_path
                n_eef = len(self.f_arr)
                start_i = int(sim_t/self.sim_dt)
                
                # Create desired contact schedule with chosen gait and desired velocity
                desired_contact_schedule, cnt_plan = self.create_desired_contact_schedule(pin_robot, urdf_path, q, v, v_des, w_des, gait, sim_t, episode_length)

                # Calculate estimated center of mass of robot given the desired velocity
                estimated_com = utils.get_estimated_com(pin_robot, q, v, v_des, episode_length + 1, self.sim_dt, utils.get_plan(gait))
                
                # Construct desired goal
                desired_goal = utils.construct_cc_goal(episode_length + 1, n_eef, desired_contact_schedule, estimated_com, 
                                                    goal_horizon=goal_horizon, sim_dt=self.sim_dt, start_step=start_i)
                
            # desired goal also considers start time!
            goal = desired_goal[index:index+1, :]
            
            # normalize policy input if required
            if norm_policy_input is not None:
                # norm_policy_input => [state_mean, state_std, goal_mean, goal_std]
                state = (state - norm_policy_input[0]) / norm_policy_input[1]
                goal = (goal - norm_policy_input[2]) / norm_policy_input[3]
                
            input = np.hstack((state, goal))
            
            # forward pass on policy network
            action = policy_network(torch.from_numpy(input).to(device).float())
            action = action.detach().cpu().numpy().reshape(-1)
                
            # compute action using given action type
            if action_type =="torque":
                tau = action
                action_history[time_elapsed, :] = tau
                
            elif action_type == "pd_target":
                q_des = action
                tau = self.kp * (q_des - q[7:]) - self.kd * v[6:]
                action_history[time_elapsed, :] = q_des
                
            elif action_type == "structured":
                tau_ff = action[:self.pin_robot.model.nv - 6]
                q_des = action[self.pin_robot.model.nv - 6:2*(self.pin_robot.model.nv - 6)]
                dq_des = action[2*(self.pin_robot.model.nv - 6):3*(self.pin_robot.model.nv - 6)]
                
                tau = tau_ff + self.kp * (q_des - q[7:]) + self.kd * (dq_des- v[6:])
                x_des = np.hstack((q_des[7:], dq_des[6:]))
                action_history[time_elapsed,:] = np.hstack((tau_ff, x_des))
            
            # check if robot state failed
            sim_failed = self.failed_states(q, gait, time_elapsed)
            if sim_failed:
                print('Policy Rollout Failed at sim step ' + str(o))
                break

            # send joint commands to robot
            self.pybullet_env.send_joint_command(tau)

            # collect goal-related quantities          
            ee_pos, ee_force = self.pybullet_env.get_contact_positions_and_forces()
            
            # see if there are new end effector contacts with the ground
            new_ee = self.new_ee_contact(ee_pos, pre_ee_pos)
            pre_ee_pos[:] = ee_pos[:]
            
            # record new eef contacts as [id, time, x, y, z pos]
            if not o == 0: # we do not need the switch at initialization
                for ee in new_ee:
                    new_ee_entry = np.hstack((np.hstack((ee, o)), ee_pos[int(ee)]))
                    if len(new_contact_pos) == 0:
                        new_contact_pos = new_ee_entry
                    else:
                        new_contact_pos = np.vstack((new_contact_pos, new_ee_entry))
                        
                    pybullet.addUserDebugPoints([ee_pos[int(ee)]], [[1, 0, 0]], pointSize=5, lifeTime=3.0)

            # exert disturbance
            if o>self.t_dist[0] and o<self.t_dist[1]:
                self.pybullet_env.apply_external_force(self.f_ext, self.m_ext)
                
            # update goal replanning variables
            pln_ctr = int((pln_ctr + 1)%(self.plan_freq/self.sim_dt))
            index += 1
            sim_t += self.sim_dt
        
        # collect measured goal history if sim is successful
        if sim_failed is False:
            n_eff = len(self.f_arr)
            self.contact_schedule = utils.construct_contact_schedule(new_contact_pos, n_eff)
            cc_goal_history = utils.construct_cc_goal(episode_length, n_eff, self.contact_schedule, com_history, 
                                        goal_horizon=goal_horizon, sim_dt=self.sim_dt, start_step=int(start_time/self.sim_dt))

        # save simulation video
        # if save_video:
        #     video_path = self.video_dir + '/' + current_date + current_time + '.mp4'
            # self.create_mp4_video(video_path, frames)

        # return histories if the simulation does not diverge
        if not sim_failed:
            end_time = len(cc_goal_history)
            return state_history[0:end_time,:], action_history[0:end_time,:], vc_goal_history[0:end_time, :], cc_goal_history, base_history[0:end_time, :], np.array(q_history), np.array(v_history), frames
        else:
            return [], [], [], [], [], [], [], frames
        
    def rollout_safedagger(self, episode_length, start_time, v_des, w_des, gait, policy_network, des_goal,
                           q0=None, v0=None, norm_policy_input:list=None, num_steps_to_block_under_safety=500,
                           uneven_terrain = False, save_video = False, add_noise = False, bounds_dict=None): 
        """Rollout robot in SafeDAGGER style

        Args:
            episode_length (int): simulation episode length (in sim step NOT sim time!)
            start_time (_type_): simulation start time (in sim time NOT sim step!)
            v_des (_type_): desired CoM translational velocity
            w_des (_type_): desired CoM yaw
            gait (_type_): desired gait
            policy_network (_type_): Policy network used to control robot
            des_goal (_type_): desired goal input to policy network
            q0 (_type_, optional): initial robot configuration. Defaults to None.
            v0 (_type_, optional): initial robot velocity. Defaults to None.
            norm_policy_input (list, optional): input normalization parameters. Defaults to None.
            num_steps_to_block_under_safety (int, optional): If robot is unsafe, minimum time to MPC should takeover. Defaults to 500.
            uneven_terrain (bool, optional): set if simulation environment should have uneven terrain. Defaults to False.
            save_video (bool, optional): set if a video of the simulation should be saved. Defaults to False.
            add_noise (bool, optional): set if noise should be added to robot state. Defaults to False.
            bounds_dict (_type_, optional): bounds where the robot configurations are consider safe. Defaults to None.

        Returns:
            state_history: robot state history. [] if failed
            action_history: robot action history. [] if failed
            vc_goal_history: velocity conditioned goal history. [] if failed
            base_history: robot base absolute coordinate history. [] if failed
            q_history: robot q history
            v_history: robot v history
            mpc_usage: record at each time step if the policy or the mpc is in control
            frames: captured frames   
        """             
          
        # check if initial robot configuration is given
        if q0 is None:
            q0 = self.q0
        else:
            assert len(q0) == self.pin_robot.model.nq, 'size of q0 given not OK!'
        
        # check if initial robot velocity is given
        if v0 is None:
            v0 = self.v0
        else:
            assert len(v0) == self.pin_robot.model.nv, 'size of v0 given not OK!'
        
        # set robot to initial position
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q0, v0)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)    
        
        # reset robot state
        self.pybullet_env.reset_robot_state(q0, v0) 
        
        # get gait files
        plan = utils.get_plan(gait)
        self.gait_params = plan
        
        # action type
        action_type = self.cfg.action_type
        
        # data related variables
        n_action = self.cfg.n_action
        n_state = self.cfg.n_state
        goal_horizon = self.cfg.goal_horizon
        
        # MPC Variables
        x0 = np.concatenate([q0, pin.utils.zero(self.pin_robot.model.nv)])
        mpc_pln_ctr = 0
        mpc_index = 0
        mpc_failed = False
        sim_t = start_time
        
        # MPC: initialize MPC
        self.gg = SoloMpcGaitGen(self.pin_robot, self.urdf_path, x0, self.plan_freq, q0, None) 
        
        # MPC: set inverse kinematic gains
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)
        
        # MPC: update gait parameters for gait generation
        self.gg.update_gait_params(self.gait_params, sim_t)  # sim_t is actually not used in update_gait_params
        
        
        # Policy Variables
        # check if input normalization parameters are ok if given
        if norm_policy_input is not None:
            assert len(norm_policy_input) == 4, 'norm_policy_input should have the terms [state_mean, state_std, goal_mean, goal_std]' 
        
        # Policy: Initialize Torch device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Policy: set network in evaluation mode
        policy_network.eval()
        
        # generate uneven terrain if enabled
        if uneven_terrain:
            self.pybullet_env.generate_terrain()
        
        # if save video enabled
        frames = []
        if save_video:
            video_cnt = int(1.0 / (self.sim_dt * self.video_fr))
            current_date = datetime.today().strftime("%b_%d_%Y_")
            current_time = datetime.now().strftime("%H_%M_%S")
            
        # State variables
        state_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_state))
        base_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        com_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        q_history, v_history = [], []
        
        # goal variables
        vc_goal_history = np.zeros((episode_length - int(start_time/self.sim_dt), 5))
        
        # variables for contacts
        ee_pos = np.zeros(len(self.f_arr)*3)
        pre_ee_pos = np.zeros((len(self.f_arr),3))
        new_contact_pos = []
        
        # Action variables
        if action_type == "structured":
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3*n_action))
        else:
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_action))
            
        # SafeDagger specific variables
        use_mpc = False
        mpc_usage = np.zeros((episode_length - int(start_time/self.sim_dt),))
        steps_blocked = 0            
            
        # NOTE: main simulation loop
        for o in range(int(start_time/self.sim_dt), episode_length):
            
            time_elapsed = o - int(start_time/self.sim_dt)
            
            # get current robot state
            q, v = self.pybullet_env.get_state()
            
            # capture frame
            if save_video:
                if o % video_cnt == 0:
                    cam_dist, cam_yaw, cam_pitch, target = self.calc_best_cam_pos(q)
                    image = self.pybullet_env.capture_image_frame(width=self.video_width, height=self.video_height, 
                                                                  cam_dist=cam_dist, cam_yaw=cam_yaw, cam_pitch=cam_pitch, target=target)
                    
                    # mark frames with text to know if mpc or policy is in control
                    if use_mpc is True:
                        text = "State is dangerous. Using MPC"
                        color = (255, 0, 0)  # (RGB)
                    else:
                        text = "State is safe. Using Policy"
                        color = (0, 255, 0)  # (RGB)
                    
                    # Add text to show if MPC or Policy is in use
                    pil_image = Image.fromarray(image)
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((0, 0), text, fill=color)
                        
                    frames.append(np.array(pil_image))

            # record q and v history
            q_history.append(q)
            v_history.append(v)
            
            # add sensor noise if enabled
            if add_noise:
                q[0:3] += self.dq_pos
                q[3:7] += self.dq_ori
                q[3:7] = q[3:7]/np.linalg.norm(q[3:7])
                q[7:] += self.dq_joint
                v[0:6] += self.dv_pos
                v[6:] += self.dv_joint
            
            # Perform forward kinematics and updates in pinocchio
            pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q, v)
            pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

            # collect state history = current robot state
            base_history[time_elapsed, :] = q[0 : 3]
            com_history[time_elapsed, :] = pin.centerOfMass(self.pin_robot.model, self.pin_robot.data, q)
            
            state_history[time_elapsed, :self.pin_robot.model.nv] = v
            state_history[time_elapsed, self.pin_robot.model.nv:self.pin_robot.model.nv + 2*len(self.f_arr)] = self.base_wrt_foot(q)
            state_history[time_elapsed, self.pin_robot.model.nv + 2*len(self.f_arr):] = q[2:]

            # collect vc goal
            vc_goal_history[time_elapsed, 0] = self.phase_percentage(o)
            vc_goal_history[time_elapsed, 1:3] = v_des[0:2]
            vc_goal_history[time_elapsed, 3] = w_des
            vc_goal_history[time_elapsed, 4] = utils.get_vc_gait_value(gait)
                    
            # get state from state history
            state = state_history[time_elapsed:time_elapsed+1]
            
            # NOTE: Check safety criteria
            if self.safedagger_state_is_dangerous(q, gait, bounds_dict=bounds_dict):  # state is dangerous. Use MPC
                # if policy was in use and robot is in danger state
                if use_mpc is False:
                    print('Policy Rollout Failed at sim step ' + str(o) + ' switch to MPC')
                    pybullet.addUserDebugText("Dangerous State. use MPC", [.0, .0, 1.0], [1, 0, 0], textSize=5.0, lifeTime=2.0)
                    use_mpc = True
                    steps_blocked = 0  # reset blocker
                    mpc_pln_ctr = 0
                    
                # mpc is already in use
                else:
                    use_mpc = True
                    steps_blocked += 1
                    
            else:  # state is safe
                # if mpc is in use and blocking is finish
                if use_mpc is True and (steps_blocked >= num_steps_to_block_under_safety):  
                    print('state safe at sim step ' + str(o) + '. switch back to policy')
                    use_mpc = False
                    steps_blocked = 0
                    
                # if mpc is in use but blocking is not yet finish
                elif use_mpc is True and (steps_blocked < num_steps_to_block_under_safety):
                    use_mpc = True
                    steps_blocked += 1
                
                # if mpc not in use
                else:
                    use_mpc = False
                    
            # NOTE: Rollouts (MPC or Policy)
            if use_mpc:  # Use MPC
                # record usage
                mpc_usage[time_elapsed] = 1
                
                # replanner
                if mpc_pln_ctr == 0:
                    # run MPC
                    xs_plan, us_plan, f_plan = self.gg.optimize(q, v, np.round(sim_t, 3), v_des, w_des)
                    # reset prediction index
                    mpc_index = 0
                    
                # evaluate MPC plan
                xs = xs_plan
                us = us_plan
                f = f_plan
                
                # handle divergence
                if(pd.isna(f).any()):
                    mpc_failed = True
                    print('MPC diverged. MPC rollout failed at step ' + str(o))
                    break
                
                # calculate torque
                q_des = xs[mpc_index][:self.pin_robot.model.nq].copy()
                dq_des = xs[mpc_index][self.pin_robot.model.nq:].copy()
                tau_ff, tau_fb = self.robot_id_ctrl.id_joint_torques(q, v, q_des, dq_des,\
                us[mpc_index], f[mpc_index])  # tau_fb is the kp kd gains of PD Policy

                # collect action history
                tau = tau_ff + tau_fb
                if action_type == "torque":
                    action_history[time_elapsed,:] = tau
                elif action_type == "pd_target":
                    action_history[time_elapsed,:] = (tau + self.kd * v[6:])/self.kp + q[7:]
                elif action_type == "structured":
                    x_des = np.hstack((q_des[7:], dq_des[6:]))
                    action_history[time_elapsed,:] = np.hstack((tau_ff, x_des))
                
                # update incremental mpc variables
                mpc_pln_ctr = int((mpc_pln_ctr + 1)%(self.plan_freq/self.sim_dt))
                mpc_index += 1
            
            
            else:  # Use Policy
                # get desired goal for this sim step. desired goal also considers start time!
                goal = des_goal[time_elapsed:time_elapsed+1, :]
                
                # normalize policy input if required
                if norm_policy_input is not None:
                    # norm_policy_input => [state_mean, state_std, goal_mean, goal_std]
                    state = (state - norm_policy_input[0]) / norm_policy_input[1]
                    goal = (goal - norm_policy_input[2]) / norm_policy_input[3]
                    
                input = np.hstack((state, goal))
                
                # forward pass on policy network
                action = policy_network(torch.from_numpy(input).to(device).float())
                action = action.detach().cpu().numpy().reshape(-1)
                    
                # compute action using given action type
                if action_type =="torque":
                    tau = action
                    action_history[time_elapsed, :] = tau
                    
                elif action_type == "pd_target":
                    q_des = action
                    tau = self.kp * (q_des - q[7:]) - self.kd * v[6:]
                    action_history[time_elapsed, :] = q_des
                    
                elif action_type == "structured":
                    tau_ff = action[:self.pin_robot.model.nv - 6]
                    q_des = action[self.pin_robot.model.nv - 6:2*(self.pin_robot.model.nv - 6)]
                    dq_des = action[2*(self.pin_robot.model.nv - 6):3*(self.pin_robot.model.nv - 6)]
                    
                    tau = tau_ff + self.kp * (q_des - q[7:]) + self.kd * (dq_des- v[6:])
                    x_des = np.hstack((q_des[7:], dq_des[6:]))
                    action_history[time_elapsed,:] = np.hstack((tau_ff, x_des))

            # send joint commands to robot
            self.pybullet_env.send_joint_command(tau)

            # collect goal-related quantities          
            ee_pos, ee_force = self.pybullet_env.get_contact_positions_and_forces()
            
            # see if there are new end effector contacts with the ground
            new_ee = self.new_ee_contact(ee_pos, pre_ee_pos)
            pre_ee_pos[:] = ee_pos[:]
            
            # step sim time for MPC
            sim_t += self.sim_dt
            
            # record new eef contacts as [id, time, x, y, z pos]
            if not o == 0: # we do not need the switch at initialization
                for ee in new_ee:
                    new_ee_entry = np.hstack((np.hstack((ee, o)), ee_pos[int(ee)]))
                    if len(new_contact_pos) == 0:
                        new_contact_pos = new_ee_entry
                    else:
                        new_contact_pos = np.vstack((new_contact_pos, new_ee_entry))
                        
                    pybullet.addUserDebugPoints([ee_pos[int(ee)]], [[1, 0, 0]], pointSize=5, lifeTime=3.0)

            # exert disturbance
            if o>self.t_dist[0] and o<self.t_dist[1]:
                self.pybullet_env.apply_external_force(self.f_ext, self.m_ext)
            
        
        # collect measured cc goal history if sim is successful
        # if mpc_failed is False:
        #     n_eff = len(self.f_arr)
        #     self.contact_schedule = utils.construct_contact_schedule(new_contact_pos, n_eff)
        #     cc_goal_history = utils.construct_cc_goal(episode_length, n_eff, self.contact_schedule, com_history, 
        #                                 goal_horizon=goal_horizon, sim_dt=self.sim_dt, start_step=int(start_time/self.sim_dt))

        # save simulation video
        # if save_video:
        #     video_path = self.video_dir + '/' + current_date + current_time + '.mp4'
            # self.create_mp4_video(video_path, frames)

        ### return histories if the simulation does not diverge
        if not mpc_failed:
            # end_time = len(cc_goal_history)
            # return state_history[0:end_time,:], action_history[0:end_time,:], vc_goal_history[0:end_time, :], cc_goal_history, base_history[0:end_time, :], \
            #         np.array(q_history), np.array(v_history), mpc_usage
            return state_history, action_history, vc_goal_history, base_history, np.array(q_history), np.array(v_history), mpc_usage, frames
        
        else:
            return [], [], [], [], np.array(q_history), np.array(v_history), mpc_usage, frames
        
    def rollout_dagger(self, episode_length, start_time, v_des, w_des, gait, policy_network, des_goal,
                           q0=None, v0=None, norm_policy_input:list=None, mpc_usage_percentage=0.9,
                           uneven_terrain = False, save_video = False, add_noise = False): 
        """Rollout robot in DAGGER style

        Args:
            episode_length (int): simulation episode length (in sim step NOT sim time!)
            start_time (_type_): simulation start time (in sim time NOT sim step!)
            v_des (_type_): desired CoM translational velocity
            w_des (_type_): desired CoM yaw
            gait (_type_): desired gait
            policy_network (_type_): Policy network used to control robot
            des_goal (_type_): desired goal input to policy network
            q0 (_type_, optional): initial robot configuration. Defaults to None.
            v0 (_type_, optional): initial robot velocity. Defaults to None.
            norm_policy_input (list, optional): input normalization parameters. Defaults to None.
            mpc_usage_percentage (float, optional): expert influence percentage. Defaults to 0.9.
            uneven_terrain (bool, optional): set if simulation environment should have uneven terrain. Defaults to False.
            save_video (bool, optional): set if a video of the simulation should be saved. Defaults to False.
            add_noise (bool, optional): set if noise should be added to robot state. Defaults to False.

        Returns:
            state_history: robot state history. [] if failed
            action_history: robot action history. [] if failed
            vc_goal_history: velocity conditioned goal history. [] if failed
            base_history: robot base absolute coordinate history. [] if failed
            q_history: robot q history
            v_history: robot v history
            mpc_usage: record at each time step if the policy or the mpc is in control
            frames: captured frames 
        """         
        
        # check mpc usage percentage
        assert (mpc_usage_percentage >= 0 and mpc_usage_percentage <= 1), 'mpc_usage_percentage should be between 0 and 1'
          
        # check if initial robot configuration is given
        if q0 is None:
            q0 = self.q0
        else:
            assert len(q0) == self.pin_robot.model.nq, 'size of q0 given not OK!'
        
        # check if initial robot velocity is given
        if v0 is None:
            v0 = self.v0
        else:
            assert len(v0) == self.pin_robot.model.nv, 'size of v0 given not OK!'
        
        # set robot to initial position
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q0, v0)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)    
        
        # reset robot state
        self.pybullet_env.reset_robot_state(q0, v0) 
        
        # get gait files
        plan = utils.get_plan(gait)
        self.gait_params = plan
        
        # action type
        action_type = self.cfg.action_type
        
        # data related variables
        n_action = self.cfg.n_action
        n_state = self.cfg.n_state
        goal_horizon = self.cfg.goal_horizon
        
        # MPC Variables
        x0 = np.concatenate([q0, pin.utils.zero(self.pin_robot.model.nv)])
        mpc_pln_ctr = 0
        mpc_index = 0
        mpc_failed = False
        sim_t = start_time
        
        # MPC: initialize MPC
        self.gg = SoloMpcGaitGen(self.pin_robot, self.urdf_path, x0, self.plan_freq, q0, None) 
        
         # MPC: set inverse kinematic gains
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)
        
        # MPC: update gait parameters for gait generation
        self.gg.update_gait_params(self.gait_params, sim_t)  # sim_t is actually not used in update_gait_params
        
        # Policy Variables
        # check if input normalization parameters are ok if given
        if norm_policy_input is not None:
            assert len(norm_policy_input) == 4, 'norm_policy_input should have the terms [state_mean, state_std, goal_mean, goal_std]' 

        # Policy: Initialize Torch device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Policy: set network in evaluation mode
        policy_network.eval()
        
        # generate uneven terrain if enabled
        if uneven_terrain:
            self.pybullet_env.generate_terrain()
        
        # if save video enabled
        frames = []
        if save_video:
            video_cnt = int(1.0 / (self.sim_dt * self.video_fr))
            current_date = datetime.today().strftime("%b_%d_%Y_")
            current_time = datetime.now().strftime("%H_%M_%S")
            
        
        # State variables
        state_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_state))
        base_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        com_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        q_history, v_history = [], []
        
        # goal variables
        vc_goal_history = np.zeros((episode_length - int(start_time/self.sim_dt), 5))
        
        # variables for contacts
        ee_pos = np.zeros(len(self.f_arr)*3)
        pre_ee_pos = np.zeros((len(self.f_arr),3))
        new_contact_pos = []
        
        # Action variables
        if action_type == "structured":
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3*n_action))
        else:
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_action))
            
        # SafeDagger specific variables
        use_mpc = False
        mpc_usage = np.zeros((episode_length - int(start_time/self.sim_dt),))
        steps_blocked = 0
        sim_failed = False
            
        # NOTE: main simulation loop
        for o in range(int(start_time/self.sim_dt), episode_length):
            
            # NOTE: Decide if to use MPC or Policy
            # only repeat in the planning frequency of the mpc
            if o % int(self.plan_freq / self.sim_dt) == 0:
                if np.random.randint(0, 100) < (mpc_usage_percentage*100):
                    use_mpc = True
                else:
                    use_mpc = False
            
            time_elapsed = o - int(start_time/self.sim_dt)
            
            # get current robot state
            q, v = self.pybullet_env.get_state()
            
            # capture frame
            if save_video:
                if o % video_cnt == 0:
                    cam_dist, cam_yaw, cam_pitch, target = self.calc_best_cam_pos(q)
                    image = self.pybullet_env.capture_image_frame(width=self.video_width, height=self.video_height, 
                                                                  cam_dist=cam_dist, cam_yaw=cam_yaw, cam_pitch=cam_pitch, target=target)
                    
                    # mark frame to let user know if mpc or policy is in control
                    if use_mpc is True:
                        text = "Using MPC"
                        color = (255, 0, 0)  # (RGB)
                    else:
                        text = "Using Policy"
                        color = (0, 255, 0)  # (RGB)
                    
                    # Add text to show if MPC or Policy is in use
                    pil_image = Image.fromarray(image)
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((0, 0), text, fill=color)
                        
                    frames.append(np.array(pil_image))

            # record q and v history
            q_history.append(q)
            v_history.append(v)
            
            # add sensor noise if enabled
            if add_noise:
                q[0:3] += self.dq_pos
                q[3:7] += self.dq_ori
                q[3:7] = q[3:7]/np.linalg.norm(q[3:7])
                q[7:] += self.dq_joint
                v[0:6] += self.dv_pos
                v[6:] += self.dv_joint
            
            # Perform forward kinematics and updates in pinocchio
            pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q, v)
            pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

            # collect state history = current robot state
            base_history[time_elapsed, :] = q[0 : 3]
            com_history[time_elapsed, :] = pin.centerOfMass(self.pin_robot.model, self.pin_robot.data, q)
            
            state_history[time_elapsed, :self.pin_robot.model.nv] = v
            state_history[time_elapsed, self.pin_robot.model.nv:self.pin_robot.model.nv + 2*len(self.f_arr)] = self.base_wrt_foot(q)
            state_history[time_elapsed, self.pin_robot.model.nv + 2*len(self.f_arr):] = q[2:]

            # collect vc goal
            vc_goal_history[time_elapsed, 0] = self.phase_percentage(o)
            vc_goal_history[time_elapsed, 1:3] = v_des[0:2]
            vc_goal_history[time_elapsed, 3] = w_des
            vc_goal_history[time_elapsed, 4] = utils.get_vc_gait_value(gait)
                    
            # get state from state history
            state = state_history[time_elapsed:time_elapsed+1]
                    
            # NOTE: Rollouts (MPC or Policy)
            if use_mpc:  # Use MPC
                # record usage
                mpc_usage[time_elapsed] = 1
                
                # replanner
                if mpc_pln_ctr == 0:
                    # run MPC
                    xs_plan, us_plan, f_plan = self.gg.optimize(q, v, np.round(sim_t, 3), v_des, w_des)
                    # reset prediction index
                    mpc_index = 0
                    
                # evaluate MPC plan
                xs = xs_plan
                us = us_plan
                f = f_plan
                
                # handle divergence
                if(pd.isna(f).any()):
                    sim_failed = True
                    print('MPC diverged. MPC rollout failed at step ' + str(o))
                    break
                
                # calculate torque
                q_des = xs[mpc_index][:self.pin_robot.model.nq].copy()
                dq_des = xs[mpc_index][self.pin_robot.model.nq:].copy()
                tau_ff, tau_fb = self.robot_id_ctrl.id_joint_torques(q, v, q_des, dq_des,\
                us[mpc_index], f[mpc_index])  # tau_fb is the kp kd gains of PD Policy

                # collect action history
                tau = tau_ff + tau_fb
                if action_type == "torque":
                    action_history[time_elapsed,:] = tau
                elif action_type == "pd_target":
                    action_history[time_elapsed,:] = (tau + self.kd * v[6:])/self.kp + q[7:]
                elif action_type == "structured":
                    x_des = np.hstack((q_des[7:], dq_des[6:]))
                    action_history[time_elapsed,:] = np.hstack((tau_ff, x_des))
                
                # update incremental mpc variables
                mpc_pln_ctr = int((mpc_pln_ctr + 1)%(self.plan_freq/self.sim_dt))
                mpc_index += 1
            
            
            else:  # Use Policy
                # get desired goal for this sim step. desired goal also considers start time!
                goal = des_goal[time_elapsed:time_elapsed+1, :]
                
                # normalize policy input if required
                if norm_policy_input is not None:
                    # norm_policy_input => [state_mean, state_std, goal_mean, goal_std]
                    state = (state - norm_policy_input[0]) / norm_policy_input[1]
                    goal = (goal - norm_policy_input[2]) / norm_policy_input[3]
                    
                input = np.hstack((state, goal))
                
                # forward pass on policy network
                action = policy_network(torch.from_numpy(input).to(device).float())
                action = action.detach().cpu().numpy().reshape(-1)
                    
                # compute action using given action type
                if action_type =="torque":
                    tau = action
                    action_history[time_elapsed, :] = tau
                    
                elif action_type == "pd_target":
                    q_des = action
                    tau = self.kp * (q_des - q[7:]) - self.kd * v[6:]
                    # WATCHOUT: action history for pd target is q_des and not tau!!!
                    action_history[time_elapsed, :] = q_des
                    
                elif action_type == "structured":
                    tau_ff = action[:self.pin_robot.model.nv - 6]
                    q_des = action[self.pin_robot.model.nv - 6:2*(self.pin_robot.model.nv - 6)]
                    dq_des = action[2*(self.pin_robot.model.nv - 6):3*(self.pin_robot.model.nv - 6)]
                    
                    tau = tau_ff + self.kp * (q_des - q[7:]) + self.kd * (dq_des- v[6:])
                    x_des = np.hstack((q_des[7:], dq_des[6:]))
                    action_history[time_elapsed,:] = np.hstack((tau_ff, x_des))
                
                # check if robot failed
                sim_failed = self.failed_states(q, gait, time_elapsed)
                if sim_failed:
                    print('Policy Rollout Failed at sim step ' + str(o))
                    break

            # send joint commands to robot
            self.pybullet_env.send_joint_command(tau)

            # collect goal-related quantities          
            ee_pos, ee_force = self.pybullet_env.get_contact_positions_and_forces()
            
            # see if there are new end effector contacts with the ground
            new_ee = self.new_ee_contact(ee_pos, pre_ee_pos)
            pre_ee_pos[:] = ee_pos[:]
            
            # step sim time for MPC
            sim_t += self.sim_dt
            
            # record new eef contacts as [id, time, x, y, z pos]
            if not o == 0: # we do not need the switch at initialization
                for ee in new_ee:
                    new_ee_entry = np.hstack((np.hstack((ee, o)), ee_pos[int(ee)]))
                    if len(new_contact_pos) == 0:
                        new_contact_pos = new_ee_entry
                    else:
                        new_contact_pos = np.vstack((new_contact_pos, new_ee_entry))
                        
                    pybullet.addUserDebugPoints([ee_pos[int(ee)]], [[1, 0, 0]], pointSize=5, lifeTime=3.0)

            # exert disturbance
            if o>self.t_dist[0] and o<self.t_dist[1]:
                self.pybullet_env.apply_external_force(self.f_ext, self.m_ext)

        # save simulation video
        # if save_video:
        #     video_path = self.video_dir + '/' + current_date + current_time + '.mp4'
            # self.create_mp4_video(video_path, frames)

        # return histories if the simulation does not diverge or fail
        if not sim_failed:
            return state_history, action_history, vc_goal_history, base_history, np.array(q_history), np.array(v_history), mpc_usage, frames
        else:
            return [], [], [], [], np.array(q_history), np.array(v_history), mpc_usage, frames
    
    def rollout_policy_return_com_state(self, episode_length, start_time, v_des, w_des, gait, policy_network, des_goal,
                       q0=None, v0=None, norm_policy_input:list=None,
                       uneven_terrain = False, save_video = False, add_noise = False, return_robot_state_if_fail=False,
                       push_f=[0, 0, 0], push_t=2.0, push_dt=0.001, fail_angle=30): 
        """Rollout robot with policy network

        Args:
            episode_length (int): simulation episode length (in sim step NOT sim time!)
            start_time (_type_): simulation start time (in sim time NOT sim step!)
            v_des (_type_): desired CoM translational velocity
            w_des (_type_): desired CoM yaw
            gait (_type_): desired gait
            policy_network (_type_): Policy network used to control robot
            des_goal (_type_): desired goal input to policy network
            q0 (_type_, optional): initial robot configuration. Defaults to None.
            v0 (_type_, optional): initial robot velocity. Defaults to None.
            norm_policy_input (list, optional): input normalization parameters. Defaults to None.
            uneven_terrain (bool, optional): set if simulation environment should have uneven terrain. Defaults to False.
            save_video (bool, optional): set if a video of the simulation should be saved. Defaults to False.
            add_noise (bool, optional): set if noise should be added to robot state. Defaults to False.
            return_robot_state_if_fail (bool, optional): set if robot q and v should be returned even if robot sim fails. Defaults to False.
            push_f (list, optional): external push force. Defaults to [0, 0, 0].
            push_t (float, optional): external push time (like at 2s in a 5s simulation). Defaults to 2.0.
            push_dt (float, optional): external push dt (to create an impulse). Defaults to 0.001.
            fail_angle (int, optional): angle to consider robot as failed. Defaults to 30.

        Returns:
            state_history: robot state history. [] if failed
            action_history: robot action history. [] if failed
            vc_goal_history: velocity conditioned goal history. [] if failed
            cc_goal_history: contact conditioned goal history. [] if failed
            base_history: robot base absolute coordinate history. [] if failed
            q_history: robot q history. (only if return_robot_state_if_fail is True)
            v_history: robot v history. (only if return_robot_state_if_fail is True)
            frames: captured frames
        """        
          
        # check if initial robot configuration is given
        if q0 is None:
            q0 = self.q0
        else:
            assert len(q0) == self.pin_robot.model.nq, 'size of q0 given not OK!'
        
        # check if initial robot velocity is given
        if v0 is None:
            v0 = self.v0
        else:
            assert len(v0) == self.pin_robot.model.nv, 'size of v0 given not OK!'
        
        # set robot to initial position
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q0, v0)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)    
        
        # reset robot state
        self.pybullet_env.reset_robot_state(q0, v0) 
        
        # get gait files
        plan = utils.get_plan(gait)
        self.gait_params = plan
        
        # action type
        action_type = self.cfg.action_type
        
        # data related variables
        n_action = self.cfg.n_action
        n_state = self.cfg.n_state
        goal_horizon = self.cfg.goal_horizon
        
        # check if input normalization parameters are ok if given
        if norm_policy_input is not None:
            assert len(norm_policy_input) == 4, 'norm_policy_input should have the terms [state_mean, state_std, goal_mean, goal_std]' 
        
        # WATCHOUT: torch device double reclared!
        # Policy: Initialize Torch device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Policy: set network in evaluation mode
        policy_network.eval()
        
        # generate uneven terrain if enabled
        if uneven_terrain:
            self.pybullet_env.generate_terrain()
        
        # if save video enabled
        frames = []
        if save_video:
            video_cnt = int(1.0 / (self.sim_dt * self.video_fr))
            current_date = datetime.today().strftime("%b_%d_%Y_")
            current_time = datetime.now().strftime("%H_%M_%S")
            
        
        # State variables
        state_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_state))
        base_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        com_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        com_vel_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        q_history, v_history = [], []
        
        # goal variables
        vc_goal_history = np.zeros((episode_length - int(start_time/self.sim_dt), 5))
        
        # variables for contacts
        ee_pos = np.zeros(len(self.f_arr)*3)
        pre_ee_pos = np.zeros((len(self.f_arr),3))
        new_contact_pos = []
        
        # Action variables
        if action_type == "structured":
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3*n_action))
        else:
            action_history = np.zeros((episode_length - int(start_time/self.sim_dt), n_action))
        
        # initialize variables for finite difference
        previous_com_pos = None
        previous_yaw = None
            
        # NOTE: main simulation loop
        for o in range(int(start_time/self.sim_dt), episode_length):
            
            time_elapsed = o - int(start_time/self.sim_dt)
            
            # get current robot state
            q, v = self.pybullet_env.get_state()
            
            # capture frames
            if save_video:
                if o % video_cnt == 0:
                    cam_dist, cam_yaw, cam_pitch, target = self.calc_best_cam_pos(q)
                    image = self.pybullet_env.capture_image_frame(width=self.video_width, height=self.video_height, 
                                                                  cam_dist=cam_dist, cam_yaw=cam_yaw, cam_pitch=cam_pitch, target=target)
                    frames.append(image)
            
            # record q and v history
            q_history.append(q)
            v_history.append(v)
            
            # add sensor noise if enabled
            if add_noise:
                q[0:3] += self.dq_pos
                q[3:7] += self.dq_ori
                q[3:7] = q[3:7]/np.linalg.norm(q[3:7])
                q[7:] += self.dq_joint
                v[0:6] += self.dv_pos
                v[6:] += self.dv_joint
            
            # Perform forward kinematics and updates in pinocchio
            pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q, v)
            pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

            # collect state history = current robot state
            base_history[time_elapsed, :] = q[0 : 3]
            
            com_pos = pin.centerOfMass(self.pin_robot.model, self.pin_robot.data, q)          
            
            if previous_com_pos is not None:
                # Compute CoM velocities (vx, vy) using finite differences
                delta_pos = com_pos - previous_com_pos
                vx = delta_pos[0] / self.sim_dt
                vy = delta_pos[1] / self.sim_dt
              
                # Compute yaw (psi) and yaw rate (w)
                current_yaw = np.arctan2(com_pos[1], com_pos[0])
                if previous_yaw is not None:
                    delta_yaw = current_yaw - previous_yaw

                    # Handle angle wrapping
                    if delta_yaw > np.pi:
                        delta_yaw -= 2 * np.pi
                    elif delta_yaw < -np.pi:
                        delta_yaw += 2 * np.pi

                    w = delta_yaw / self.sim_dt
                else:
                    w = 0.0  # Assume no initial yaw rate
            else:
                vx, vy, w = 0.0, 0.0, 0.0  # Initialize at the first step 
            
            # Update previous state
            previous_com_pos = com_pos
            previous_yaw = np.arctan2(com_pos[1], com_pos[0])

            # Store results in com_history
            com_history[time_elapsed, :] = com_pos  # CoM position
            com_vel_history[time_elapsed, 0:2] = [vx, vy]  # CoM translational velocities
            com_vel_history[time_elapsed, 2] = w  # Yaw rate 
            
            state_history[time_elapsed, :self.pin_robot.model.nv] = v
            state_history[time_elapsed, self.pin_robot.model.nv:self.pin_robot.model.nv + 2*len(self.f_arr)] = self.base_wrt_foot(q)
            state_history[time_elapsed, self.pin_robot.model.nv + 2*len(self.f_arr):] = q[2:]

            # collect vc goal
            vc_goal_history[time_elapsed, 0] = self.phase_percentage(o)
            vc_goal_history[time_elapsed, 1:3] = v_des[0:2]
            vc_goal_history[time_elapsed, 3] = w_des
            vc_goal_history[time_elapsed, 4] = utils.get_vc_gait_value(gait)
                    
            # get state from state history
            state = state_history[time_elapsed:time_elapsed+1]
            
            # desired goal also considers start time!
            goal = des_goal[time_elapsed:time_elapsed+1, :]
            
            # normalize policy input if required
            if norm_policy_input is not None:
                # norm_policy_input => [state_mean, state_std, goal_mean, goal_std]
                state = (state - norm_policy_input[0]) / norm_policy_input[1]
                goal = (goal - norm_policy_input[2]) / norm_policy_input[3]
            
            # construct input
            input = np.hstack((state, goal))
            
            # forward pass on policy network
            action = policy_network(torch.from_numpy(input).to(device).float())
            action = action.detach().cpu().numpy().reshape(-1)
                
            # compute action using given action type
            if action_type =="torque":
                tau = action
                action_history[time_elapsed, :] = tau
                
            elif action_type == "pd_target":
                q_des = action
                tau = self.kp * (q_des - q[7:]) - self.kd * v[6:]
                # WATCHOUT: action history for pd target is q_des and not tau!!!
                action_history[time_elapsed, :] = q_des
                
            elif action_type == "structured":
                tau_ff = action[:self.pin_robot.model.nv - 6]
                q_des = action[self.pin_robot.model.nv - 6:2*(self.pin_robot.model.nv - 6)]
                dq_des = action[2*(self.pin_robot.model.nv - 6):3*(self.pin_robot.model.nv - 6)]
                
                tau = tau_ff + self.kp * (q_des - q[7:]) + self.kd * (dq_des- v[6:])
                x_des = np.hstack((q_des[7:], dq_des[6:]))
                action_history[time_elapsed,:] = np.hstack((tau_ff, x_des))
            
            # check if robot state failed
            sim_failed = self.failed_states(q, gait, time_elapsed, fail_angle=fail_angle)
            if sim_failed:
                print('Policy Rollout Failed at sim step ' + str(o))
                break

            # send joint commands to robot
            self.pybullet_env.send_joint_command(tau)

            # collect goal-related quantities          
            ee_pos, ee_force = self.pybullet_env.get_contact_positions_and_forces()
            
            # see if there are new end effector contacts with the ground
            new_ee = self.new_ee_contact(ee_pos, pre_ee_pos)
            pre_ee_pos[:] = ee_pos[:]
            
            # record new eef contacts as [id, time, x, y, z pos]
            if not o == 0: # we do not need the switch at initialization
                for ee in new_ee:
                    new_ee_entry = np.hstack((np.hstack((ee, o)), ee_pos[int(ee)]))
                    if len(new_contact_pos) == 0:
                        new_contact_pos = new_ee_entry
                    else:
                        new_contact_pos = np.vstack((new_contact_pos, new_ee_entry))
                        
                    pybullet.addUserDebugPoints([ee_pos[int(ee)]], [[1, 0, 0]], pointSize=5, lifeTime=3.0)

            # exert disturbance
            if o>=(push_t/self.sim_dt) and o<((push_t+push_dt)/self.sim_dt):
                self.pybullet_env.apply_external_force(push_f, [0, 0, 0])
            
        
        # collect measured goal history if sim is successful
        if sim_failed is False:
            n_eff = len(self.f_arr)
            self.contact_schedule = utils.construct_contact_schedule(new_contact_pos, n_eff)
            cc_goal_history = utils.construct_cc_goal(episode_length, n_eff, self.contact_schedule, com_history, 
                                        goal_horizon=goal_horizon, sim_dt=self.sim_dt, start_step=int(start_time/self.sim_dt))

        # save simulation video
        # if save_video:
        #     video_path = self.video_dir + '/' + current_date + current_time + '.mp4'
            # self.create_mp4_video(video_path, frames)

        # return histories if the simulation does not diverge
        # Return histories if the simulation does not diverge
        if not sim_failed:
            end_time = len(cc_goal_history)
            return (state_history[0:end_time, :], 
                    action_history[0:end_time, :], 
                    vc_goal_history[0:end_time, :], 
                    cc_goal_history, 
                    base_history[0:end_time, :], 
                    com_history[0:end_time, :],  # Include com_history here
                    com_vel_history[0:end_time, :],
                    np.array(q_history), 
                    np.array(v_history), 
                    frames)

        elif sim_failed and return_robot_state_if_fail:
            return ([], 
                    [], 
                    [], 
                    [], 
                    [], 
                    com_history[:],  # Return the entire com_history up to the failure
                    com_vel_history[:],
                    np.array(q_history), 
                    np.array(v_history), 
                    frames)

        else:
            return ([], 
                    [], 
                    [], 
                    [], 
                    [], 
                    [],  # Return an empty list for com_history
                    [],
                    [], 
                    [], 
                    frames)
