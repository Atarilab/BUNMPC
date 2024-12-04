## This is a demo for bound motion in mpc
## Author : Majid Khadiv
## Date : 01/09/2021
import sys
sys.path.append('../')

import time
import numpy as np
import pandas as pd
import pinocchio as pin
import pybullet
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from mpc.goal_conditioned_mpc import SoloMpcGC
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController
from utils import construct_contact_schedule, construct_goals, get_plan
from plotting import plot_goal


class SimulationGC():
    def __init__(self, gait, des_cnt_plan, des_swing_time, cfg=None):
        
        # get gait files
        self.gait = gait
        self.gait_params = get_plan(gait)
        self.des_cnt_plan = des_cnt_plan
        self.des_swing_time = des_swing_time

        ### robot config and init
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.urdf_path = Solo12Config.urdf_path

        self.q0 = np.array(Solo12Config.initial_configuration)
        self.q0[0:2] = 0.0

        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        self.x0 = np.concatenate([self.q0, pin.utils.zero(self.pin_robot.model.nv)])
        self.f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]

        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, self.q0, self.v0)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

        self.plan_freq = 0.05 # sec
        self.update_time = 0 # sec (time of lag)
        self.lag = 0
        self.index = 0

        ## Motion
        self.mpc = SoloMpcGC(self.pin_robot, self.urdf_path, self.x0,\
                  self.plan_freq, self.q0, None)

        # self.q_nominal = np.zeros((int(self.gait_params.gait_period / self.plan_freq),\
        #                 self.pin_robot.model.nq))
        # self.v_nominal = np.zeros((int(self.gait_params.gait_period / self.plan_freq),\
        #                 self.pin_robot.model.nv))

        ### external disturbance
        self.f_ext = [0., 0., 0.]
        self.m_ext = [0., 0., 0.]
        self.t_dist = [0., 1.]

        ### specifications for data collection and learning
        self.meas_vec_len = (self.pin_robot.model.nq - 7 + self.pin_robot.model.nv)
        self.kp = 2.
        self.kd = .1
        self.n_meas_history = 10
        self.dt_meas_history = 1  #ms
        self.sim_dt = .001

        ### paths to different repos
        # self.video_path = "/home/atari_ws/goal_conditioned_behavioral_cloning/video/"
        # self.estimator_path = "/home/atari_ws/goal_conditioned_behavioral_cloning/models/n_random_15_wo_test/"

        ### noise added to the measured base states
        self.dz = np.random.normal(0., .0, 1) # noise to base orientation
        self.dq = np.random.normal(0., .0, 4) # noise to base orientation
        self.dv = np.random.normal(0., 0.15, 6) # noise to base velocity

    def base_wrt_foot(self, q):
        out = np.zeros(2*len(self.f_arr))
        for i in range(len(self.f_arr)):
            foot = self.pin_robot.data.oMf[self.pin_robot.model.getFrameId(self.f_arr[i])].translation
            out[2*i:2*(i+1)] = q[0:2] - foot[0:2]
        return out

    def phase_percentage(self, t):
        phi = ((t*self.sim_dt) % self.gait_params.gait_period)/self.gait_params.gait_period
        return phi

    def estimate_state(self, measurement, n_hist, n_state):
        import torch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        from networks import EstimatorNet
        estimator_network = EstimatorNet(self.meas_vec_len*(n_hist+1)+4 ,n_state-4)
        estimator_network.eval()
        estimator_network = torch.load(self.estimator_path+"estimator_"+str(n_hist)+\
         "_pd_target_"+self.gait+"_498.pth")
        measurement
        state_est = estimator_network(torch.from_numpy(measurement).to(device).float())
        state_est = state_est.detach().cpu().numpy().reshape(-1)
        v = state_est[0:self.pin_robot.model.nv]
        q = state_est[self.pin_robot.model.nv + 2*len(self.f_arr):]
        q = np.hstack((np.zeros(2), q))

        return q, v

    def new_ee_contact(self, ee_cur, ee_pre):
        new_contact = []
        threshold = 1e-12
        for i in range(len(self.f_arr)):
            if abs(ee_cur[i][0])>threshold and abs(ee_pre[i][0])<=threshold:
                new_contact = np.hstack((new_contact,i))
        return new_contact


    def compute_rl_reward(self, actual_contact, desired_contact, n_eff, last_contact_switch):
        if len(actual_contact[0][0:, 0]) < len(actual_contact[0][0:, 0]):
            return 0., last_contact_switch
        
        switch_time_error = 0.
        pos_error = 0.
        for i in range(last_contact_switch, len(actual_contact[0][0:, 0])):
            for idx_ee in range(n_eff): # loop over the end effectors
                if actual_contact[idx_ee][i, 0] != 0: # => there is a contact switch for end effector idx_ee
                    #timestep = int(actual_contact[idx_ee, i, 0])-1 # timestep t in the RL simulation when the contact switch occurs
                    #print('\n timestep', timestep)
                    switch_time_error += desired_contact[idx_ee, i, 0]-actual_contact[idx_ee, i, 0]
                    pos_error += np.linalg.norm(desired_contact[idx_ee, i, 1:4]-\
                                            actual_contact[idx_ee, i, 1:4], 2)

                    last_idx_contact_switch = i

        rewards = -0.7 * (switch_time_error/5000 + pos_error)

        return rewards, last_idx_contact_switch
    
    def rollout(self, episode_length, start_time, v_des, w_des, action_types,\
                nominal = True, uneven_terrain = False, save_video = False,\
                use_estimator = False, add_noise = False, goal_horizon = 1, include_reward=False, desired_contact=None,\
                visualizer=False, server=pybullet.GUI):
        # declare variables
        pln_ctr = 0
        index = 0
        counter = 0
        simulation_diverges = False
        last_contact_switch = 0
        sim_t = start_time
        
        # update gait parameters for gait generation
        self.mpc.update_gait_params(self.gait_params, sim_t, self.des_cnt_plan, self.des_swing_time)
        
        # Spawn robot in pybullet
        self.robot = PyBulletEnv(Solo12Robot, self.q0, self.v0, server=server)
        
        # generate uneven terrain if enabled
        # if uneven_terrain:
        #     self.robot.generate_terrain()

        # set pybullet view angle and interface style
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        pybullet.resetDebugVisualizerCamera(1, 75, -20, (0.5, .0, 0.))
        
        # CHANGES: plot cnt_plan
        # for i in range(len(desired_contact[0])):
        #     max_ind = np.argmax()
        # for i in range(int(episode_length*self.gait_params.gait_dt)):
        #     for j in range(len(desired_contact[0])):
        #         if desired_contact[i, j, 0] == 1:
        #             pybullet.addUserDebugPoints([desired_contact[i, j, 1:]], [[1, 0, 0]], pointSize=8.0)
        
        # if save video enabled
        # if save_video:
        #     self.robot.start_video_recording(self.video_path+"_mpc.mp4")
            
        # initialize inverse dynamics controller
        self.robot_id_ctrl = InverseDynamicsController(self.pin_robot, self.f_arr)
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)
        
        # declare variables for all histories
        state_history = np.zeros((episode_length - int(start_time/self.sim_dt),\
                                self.pin_robot.model.nq + 3 + len(self.f_arr) * 2 + self.pin_robot.model.nv))
        measurement_history = np.zeros((episode_length - int(start_time/self.sim_dt),\
                                        (self.n_meas_history + 1) * self.meas_vec_len + 4))
        base_history = np.zeros((episode_length - int(start_time/self.sim_dt), 3))
        action_history = {}
        
        if include_reward:
            reward_history = np.zeros((episode_length - int(start_time/self.sim_dt), 1))
        
        # variables for contacts
        ee_pos = np.zeros(len(self.f_arr)*3)
        pre_ee_pos = np.zeros((len(self.f_arr),3))
        new_contact_pos = []
        
        
        # action history declaration for different action types
        for action_type in action_types:
            if action_type == "structured":
                action_history[action_type] = np.zeros((episode_length - int(start_time/self.sim_dt),\
                 3*(self.pin_robot.model.nv - 6)))
            else:
                action_history[action_type] = np.zeros((episode_length - int(start_time/self.sim_dt),\
                 self.pin_robot.model.nv - 6))
                
        # NOTE: main simulation loopü
        for o in range(int(start_time/self.sim_dt), episode_length):

            # get current robot state
            q, v = self.robot.get_state()
            
            # add sensor noise if enabled
            if add_noise:
                q[2] += self.dz
                q[3:7] += self.dq
                q[3:7] = q[3:7]/np.linalg.norm(q[3:7])
                v[0:6] += self.dv
            
            # get current IMU measurements
            lin_acc, ang_vel = self.robot.get_imu_data()
            
            # Perform forward kinematics and updates in pinocchio
            pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q, v)
            pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

            # collect state history
            time_elapsed = o-int(start_time/self.sim_dt)
            base_history[time_elapsed, :] = q[0 : 3]
            state_history[time_elapsed, 0] = self.phase_percentage(o)
            state_history[time_elapsed, 1:3] = v_des[0:2]
            state_history[time_elapsed, 3] = w_des
            state_history[time_elapsed, 4:self.pin_robot.model.nv + 4] = v
            state_history[time_elapsed, self.pin_robot.model.nv + 4:\
                    self.pin_robot.model.nv + 4 + 2*len(self.f_arr)] = self.base_wrt_foot(q)
            state_history[time_elapsed, self.pin_robot.model.nv + 4 +\
                    2*len(self.f_arr):-1] = q[2:]
            
            if self.gait=="trot":
                state_history[time_elapsed, -1] = 1.
            elif self.gait=="jump":
                state_history[time_elapsed, -1] = 2.
            elif self.gait=="bound":
                state_history[time_elapsed, -1] = 3.

            # collect measurement history
            measurement_history[time_elapsed, 0] = self.phase_percentage(o)
            measurement_history[time_elapsed, 1:3] = v_des[0:2]
            measurement_history[time_elapsed, 3] = w_des
            measurement_history[time_elapsed, 4:7] = lin_acc
            measurement_history[time_elapsed, 7:10] = ang_vel
            measurement_history[time_elapsed, 10:self.pin_robot.model.nv + 4] = v[6:]
            measurement_history[time_elapsed, self.pin_robot.model.nv + 4:self.meas_vec_len + 4] = q[7:]

            # add history of measurements
            for i in range(self.n_meas_history):
                if o >= (i+1) * self.dt_meas_history:
                    measurement_history[time_elapsed,\
                        (i+1) * self.meas_vec_len + 4 : (i+2) * self.meas_vec_len + 4] =\
                            measurement_history[time_elapsed - self.dt_meas_history,\
                            i * self.meas_vec_len + 4 : (i+1) * self.meas_vec_len + 4]
                else:
                     measurement_history[time_elapsed,\
                        (i+1) * self.meas_vec_len + 4 : (i+2) * self.meas_vec_len + 4] =\
                            measurement_history[0, i * self.meas_vec_len + 4 : (i+1) * self.meas_vec_len + 4]

            # use estimator to estimate current robot state from current measurements if enabled
            # if use_estimator:
            #     n_hist = 0
            #     n_state = len(state_history[0,:])
            #     q_est, v_est = self.estimate_state(measurement_history[time_elapsed:time_elapsed+1,\
            #      0:(n_hist+1)*self.meas_vec_len + 4], n_hist, n_state)
            #     q[0:7] = q_est[0:7]
            #     v[0:6] = v_est[0:6]

            # NOTE: Planning Control. Replan motion
            if pln_ctr == 0:
                
                # QUESTION: what is this nominal thing?
                # if (nominal and counter < int(self.gait_params.gait_period / self.plan_freq)):
                #     self.q_nominal[counter] = q
                #     self.v_nominal[counter] = v
                #     counter += 1


                contact_configuration = self.robot.get_current_contacts()
                
                xs_plan, us_plan, f_plan = self.mpc.optimize(q, v, np.round(sim_t,3), v_des, w_des, o)
                index = 0
                
                dot_positions = [(0, 0, 1.0), (1, 1, 0.1), (-1, -1, 0.1)]

            xs = xs_plan
            us = us_plan
            f = f_plan
            if(pd.isna(f).any()):
                simulation_diverges = True
                break

            q_des = xs[index][:self.pin_robot.model.nq].copy()
            dq_des = xs[index][self.pin_robot.model.nq:].copy()
            tau_ff, tau_fb = self.robot_id_ctrl.id_joint_torques(q, v, q_des, dq_des,\
             us[index], f[index], contact_configuration)  # tau_fb is the kp kd gains of PD Policy

            ### collect action history
            tau = tau_ff + tau_fb
            for action_type in action_types:
                if action_type == "torque":
                    action_history["torque"][time_elapsed,:] = tau
                elif action_type == "pd_target":
                    action_history["pd_target"][time_elapsed,:] = \
                                  (tau + self.kd * v[6:])/self.kp + q[7:]
                elif action_type == "structured":
                    x_des = np.hstack((q_des[7:], dq_des[6:]))
                    action_history["structured"][time_elapsed,:] = \
                    np.hstack((tau_ff, x_des))


            self.robot.send_joint_command(tau)

            
            # CHANGES: Record planned goals (contacts)
            # planned_gait_index = int(index * self.sim_dt / self.gait_params.gait_dt)  # for the case that gait_dt is very small!
            
            # for i in range(len(self.f_arr)):
            #     if (prev_planned_contact_pos[i] is False) and (self.mpc.cnt_plan[planned_gait_index, i, 0] == 1):
            #         prev_planned_contact_pos[i] = True
            #         contact_event = np.hstack((np.array([i, o]), self.mpc.cnt_plan[planned_gait_index, i, 1:]))
                    
            #         # stack recorded values    
            #         if len(planned_contact_pos) == 0:
            #             planned_contact_pos = contact_event
            #         else:
            #             planned_contact_pos = np.vstack((planned_contact_pos, contact_event))    
            #     elif (prev_planned_contact_pos[i] is True) and (self.mpc.cnt_plan[planned_gait_index, i, 0] == 0):
            #         prev_planned_contact_pos[i] = False
            #     else:
            #         pass
            # END CHANGES

            ### collect goal-related quantities
            ee_pos, ee_force = self.robot.get_contact_positions_and_forces()
            
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
            
            if include_reward:
                if len(new_contact_pos)==0 or len(new_contact_pos.shape)<2:
                    reward=0.
                else:
                    
                    actual_contact = construct_contact_schedule(new_contact_pos, len(self.f_arr))
                    reward, last_contact_switch = self.compute_rl_reward(actual_contact, desired_contact,
                                                            len(self.f_arr),
                                                            last_contact_switch)
                reward_history[time_elapsed,:] = reward

            ### exert disturbance
            if o>self.t_dist[0] and o<self.t_dist[1]:
                self.robot.apply_external_force(self.f_ext, self.m_ext)

            sim_t += self.sim_dt
            pln_ctr = int((pln_ctr + 1)%(self.plan_freq/self.sim_dt))
            index += 1

        # CHANGES: collect planned and measured goal history
        # for ind in range(len(planned_contact_pos)):
        #     t = int(planned_contact_pos[ind, 1])
        #     planned_contact_pos[ind, 2:-1] = planned_contact_pos[ind, 2:-1] + base_history[t, :2]
        
        ### collect measured goal history
        n_eff = len(self.f_arr)
        
        self.contact_schedule = construct_contact_schedule(new_contact_pos, n_eff)
        #select the minimum of the goal_horizon_th largest switch time
        end_time_meas = episode_length
        for i in range(len(self.f_arr)):
            max_time = np.sort(self.contact_schedule[i, :, 0])[-goal_horizon] 
            end_time_meas = int(min(end_time_meas, max_time))
            
        # self.planned_contact_schedule = construct_contact_schedule(planned_contact_pos, n_eff)
        # #select the minimum of the goal_horizon_th largest switch time
        # end_time_planned = episode_length
        # for i in range(len(self.f_arr)):
        #     max_time = np.sort(self.planned_contact_schedule[i, :, 0])[-goal_horizon] 
        #     end_time_planned = int(min(end_time_planned, max_time))
            
        #end_time = min(end_time_meas, end_time_planned)
        end_time = end_time_meas

        # np.save("contact_schedule.npy", contact_schedule)
        goal_history = np.zeros((end_time, 4*goal_horizon*n_eff))
        for i in range(end_time):
            goal_history[i, :] = construct_goals(i, base_history[i, :], self.contact_schedule, n_eff, goal_horizon)

        # np.save("contact_schedule.npy", contact_schedule)
        # planned_goal_history = np.zeros((end_time, 4*goal_horizon*n_eff))
        # for i in range(end_time):
        #     planned_goal_history[i, :] = construct_goals(i, base_history[i, :], self.planned_contact_schedule, n_eff, goal_horizon)
        
        # END TODO
        
        # plot_goal(goal_history, planned_goal_history)

        ### save simulation video
        if save_video:
            self.robot.stop_video_recording()
        
        
        
        ### stop pybullet
        pybullet.disconnect(self.robot.env.physics_client)

        ### return histories if the simulation does not diverge
        if not simulation_diverges:
            states = state_history[0:end_time,:]
            measurements = measurement_history[0:end_time,:]
            for action_type in action_types:
                action_history[action_type] = action_history[action_type][0:end_time,:]
            if include_reward:
                return states, measurements, action_history, goal_history, reward_history
            return states, measurements, action_history, goal_history
        else:
            if include_reward:
                return [], [], [], [], []
            return [], [], [], []
    