import numpy as np
import sys
sys.path.append('../')
from gait_planner_cpp import GaitPlanner
import pinocchio as pin
from mpc.abstract_cyclic_gen import SoloMpcGaitGen
from utils import construct_contact_schedule

class ContactPlanner():
    def __init__(self, plan):
        self.plan = plan  # from files like solo12_trot.py
        self.eff_names = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.hip_names = ["FL_HFE", "FR_HFE", "HL_HFE", "HR_HFE"]
        self.sim_dt = .001
        self.gait_planner = GaitPlanner(self.plan.gait_period, np.array(self.plan.stance_percent), \
                                        np.array(self.plan.phase_offset), self.plan.step_ht)
        self.gravity = 9.81
        self.foot_size = 0.018

    def get_end_effector_contact(self, cnt_index):
        """get end effectors that are in contact with ground

        Args:
            cnt_index (_type_): contact of all 4 legs at a certain time

        Returns:
            output: list of contact which are in contact
        """        
        out = []        
        for j in range(len(cnt_index)):
            if cnt_index[j] == 1.:
                out = np.hstack((out, j))
        return out

    def get_switches(self, cnt_plan, start_time=0):
        """get if end effector switched contact position

        Args:
            cnt_plan (_type_): contact plan
            start_time (int, optional): simulation start time. Defaults to 0.

        Returns:
            out: output contacts that switched position
        """        
        out = []
        pre_contact = self.get_end_effector_contact(cnt_plan[0, :, 0])
        for i in range(1, len(cnt_plan[:, 0, 0])):
            contact = self.get_end_effector_contact(cnt_plan[i, :, 0])
            for ee in contact:
                if not ee in pre_contact:
                    cnt = np.hstack((ee, start_time + i*self.plan.gait_dt/self.sim_dt))
                    cnt = np.hstack((cnt, cnt_plan[i, int(ee), 1:4]))
                    cnt[-1] = 1e-3 #TODO: for uneven terrain this should not be hard-coded
                    if len(out) == 0:
                        out = cnt
                    else:
                        out = np.vstack((out, cnt))
            pre_contact = contact
        return out
    
    def get_raibert_contact_plan(self, pin_robot, urdf_path, q0, v0, v_des, w_des, episode_length, start_time):
        """create a raibert contact plan for the robot eef. Used equations from https://arxiv.org/pdf/1909.06586.pdf 

        Args:
            pin_robot (_type_): pinocchio robot model
            urdf_path (_type_): robot urdf
            q0 (_type_): initial configuration
            v0 (_type_): initial velocity
            v_des (_type_): desired robot base translational velocity
            w_des (_type_): desired robot base rotational velocity
            episode_length (_type_): simulation length
            start_time (_type_): simulation start time

        Returns:
            cnt_plan: [horizon, eff index, (in contact? + x,y,z)]
            swing_time: swing time
        """        
        
        # update robot forward kinamatics
        pin.forwardKinematics(pin_robot.model, pin_robot.data, q0, v0)
        pin.updateFramePlacements(pin_robot.model, pin_robot.data)
        
        # get robot initial CoM
        com_init = pin.centerOfMass(pin_robot.model, pin_robot.data, q0, v0)
        com = np.round(com_init[0:2], 3)
        z_height = pin.centerOfMass(pin_robot.model, pin_robot.data, q0, v0)[2]
        
        # robot com velocity
        vcom = np.round(v0[0:3], 3)

        self.ee_frame_id = []
        
        # set offsets of endeffector contact location. offset of 0.0 is directly below hip
        self.offsets = np.zeros((len(self.eff_names), 3))
        for i in range(len(self.eff_names)):
            self.ee_frame_id.append(pin_robot.model.getFrameId(self.eff_names[i]))
            self.offsets[i] = pin_robot.data.oMf[pin_robot.model.getFrameId(self.hip_names[i])].translation - com_init.copy()
            self.offsets[i] = np.round(self.offsets[i], 3)
        # Contact-planning offsets
        self.offsets[0][0] -= 0.00 #Front Left_X
        self.offsets[0][1] += 0.04 #Front Left_Y

        self.offsets[1][0] -= 0.00 #Front Right_X
        self.offsets[1][1] -= 0.04 #Front Right_Y

        self.offsets[2][0] += 0.00 #Hind Left_X
        self.offsets[2][1] += 0.04 #Hind Left_Y

        self.offsets[3][0] += 0.00 #Hind Right X
        self.offsets[3][1] -= 0.04 #Hind Right Y
        self.apply_offset = True

        # Get current yaw (pitch and roll set to 0.0)
        R = pin.Quaternion(np.array(q0[3:7])).toRotationMatrix()
        
        #Rotate offsets to local frame
        for i in range(len(self.eff_names)):
            #Rotate offsets to local frame
            self.offsets[i] = np.matmul(R.T, self.offsets[i])
        
        rpy_vector = pin.rpy.matrixToRpy(R)
        rpy_vector[0] = 0.0
        rpy_vector[1] = 0.0
        R = pin.rpy.rpyToMatrix(rpy_vector)

        vtrack = v_des[0:2] # this effects the step location (if set to vcom it becomes raibert)
        
        # WATCHOUT:_20 is set as a buffer
        #horizon = int(20.*episode_length*self.sim_dt/self.plan.gait_period)
        horizon = int(20.0 * episode_length * self.sim_dt * self.plan.gait_horizon * self.plan.gait_period / self.plan.gait_dt)
        
        # initialize cnt plan array
        cnt_plan = np.zeros((horizon, len(self.eff_names), 4))  # 4 -> in contact? + x + y + z

        # This array determines when the swing foot cost should be enforced in the ik
        self.swing_time = np.zeros((horizon, len(self.eff_names)))
        self.prev_cnt = np.zeros((len(self.eff_names), 3))
        self.curr_cnt = np.zeros(len(self.eff_names))

        # Planning loop
        for i in range(horizon):
            
            # Loop over each eef
            for j in range(len(self.eff_names)):
                
                # if first timestep...
                if i == 0:
                    # if eef is at start of gait...
                    if self.gait_planner.get_phase(start_time, j) == 1:
                        # set in contact to true
                        cnt_plan[i][j][0] = 1
                        
                        # set position as current eef position
                        cnt_plan[i][j][1:4] = np.round(pin_robot.data.oMf[self.ee_frame_id[j]].translation, 3)
                        
                        # save prev cnt pos
                        self.prev_cnt[j] = cnt_plan[i][j][1:4]
                        
                    
                    else:
                        # set in contact to false
                        cnt_plan[i][j][0] = 0
                        
                        # set position as current eef position
                        cnt_plan[i][j][1:4] = np.round(pin_robot.data.oMf[self.ee_frame_id[j]].translation, 3)
                        
                        # TODO: Implement offset?
                        #     cnt_plan[i][j][1:3] += self.offsets[j]

                # All other time steps
                else:
                    # get current planning timestep
                    ft = np.round(start_time + i*self.plan.gait_dt,3)

                    # if current timestep is at the start of a gait cycle...
                    if self.gait_planner.get_phase(ft, j) == 1:
                        
                        # set in contact to true
                        cnt_plan[i][j][0] = 1

                        # if eef is already in contact in prev timestep...
                        if cnt_plan[i-1][j][0] == 1:
                            # contact position should not change from prev timestep
                            cnt_plan[i][j][1:4] = cnt_plan[i-1][j][1:4]
                            
                        else:
                            # get projected hip/shoulder position if current velocity continues
                            hip_loc = com + np.matmul(R, self.offsets[j])[0:2] + i*self.plan.gait_dt*vtrack
                            
                            # get Raibert heuristic -> legs landing and leaving angle should be identical if robot is travelling with v_des
                            raibert_step = 0.5*vtrack*self.plan.gait_period*self.plan.stance_percent[j] - 0.05*(vtrack - v_des[0:2])
                            
                            # get centrifugal term
                            ang_step = 0.5*np.sqrt(z_height/self.gravity)*vtrack
                            ang_step = np.cross(ang_step, [0.0, 0.0, w_des])

                            # contact location are all these terms combined
                            cnt_plan[i][j][1:3] = raibert_step[0:2] + hip_loc + ang_step[0:2]

                            # set contact location z
                            cnt_plan[i][j][3] = self.foot_size

                        self.prev_cnt[j] = cnt_plan[i][j][1:4]

                    # if current time step is not at start of gait cycle...
                    else:
                        # foot should not be in contact
                        cnt_plan[i][j][0] = 0
                        
                        # get phase percentage for current timestep
                        per_ph = np.round(self.gait_planner.get_percent_in_phase(ft, j), 3)
                        
                        # get projected hip/shoulder position if current velocity continues
                        hip_loc = com + np.matmul(R,self.offsets[j])[0:2] + i*self.plan.gait_dt*vtrack
                        
                        # get centrifugal term
                        ang_step = 0.5*np.sqrt(z_height/self.gravity)*vtrack
                        ang_step = np.cross(ang_step, [0.0, 0.0, w_des])

                        # if phase is still less than 50%
                        if per_ph < 0.5:
                            cnt_plan[i][j][1:3] = hip_loc + ang_step[0:2]
                            
                        else:
                            # CHANGES: added raibert step into cnt plan for phases > 0.5
                            raibert_step = 0.5*vtrack*self.plan.gait_period*self.plan.stance_percent[j] - 0.05*(vtrack - v_des[0:2])
                            cnt_plan[i][j][1:3] = hip_loc + ang_step[0:2] + raibert_step[0:2]

                        if per_ph - 0.5 < 0.02:
                            self.swing_time[i][j] = 1

                        cnt_plan[i][j][3] = self.foot_size

        return cnt_plan, self.swing_time

    def get_contact_schedule(self, pin_robot, urdf_path, q0, v0, v_des, w_des, episode_length, start_time):
        """create a raibert contact schedule 

        Args:
            pin_robot (_type_): pinocchio robot model
            urdf_path (_type_): robot urdf path
            q0 (_type_): robot initial configuration
            v0 (_type_): robot initial velocity
            v_des (_type_): desired robot base translational velocity
            w_des (_type_): desired robot base rotational velocity
            episode_length (_type_): simulation length
            start_time (_type_): simulation start time

        Returns:
            cnt_schedule: [n_eff, number of contact events, (time + x,y,z)]
            cnt_plan: [horizon, eff index, (in contact? + x,y,z)]
        """        
        cnt_plan, swing_time = self.get_raibert_contact_plan(pin_robot, urdf_path, q0, v0, v_des, w_des, episode_length, start_time)
        new_contact_pos = self.get_switches(cnt_plan, start_time/self.sim_dt)
        cnt_schedule = construct_contact_schedule(new_contact_pos, len(self.eff_names))
        return cnt_schedule, cnt_plan
    
