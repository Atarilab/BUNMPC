## Contains solo 12 gait params
## Author : Avadesh Meduri
## Date : 7/7/21

import numpy as np
from motions.weight_abstract import BiconvexMotionParams
from robot_properties_solo.config import Solo12Config

pin_robot = Solo12Config.buildRobotWrapper()
urdf_path = Solo12Config.urdf_path

#### Bound #######################################
plan = BiconvexMotionParams("solo12", "Bound")
#
# Cnt
plan.gait_period = 0.3
plan.stance_percent = [0.5, 0.5, 0.5, 0.5]
plan.gait_dt = 0.05
plan.phase_offset = [0.0, 0.0, 0.5, 0.5]

# IK
plan.state_wt = np.array([0., 0, 1e3] + [10, 10, 10] + [50.0] * (pin_robot.model.nv - 6) \
                        + [0.00] * 3 + [100, 10, 100] + [0.5] *(pin_robot.model.nv - 6))
#
plan.ctrl_wt = [0.5, 0.5, 0.5] + [1, 1, 1] + [0.5] *(pin_robot.model.nv - 6)
#
plan.swing_wt = [1e4, 1e4]
#plan.cent_wt = [5e+1, 5e+2] #
plan.cent_wt = [5e+1, 5e+2] # modified
plan.step_ht = 0.07
plan.reg_wt = [7e-3, 7e-5]
#
# Dyn
plan.W_X =        np.array([1e-5, 1e-5, 5e+4, 1e1, 1e1, 1e+3, 5e+3, 1e+4, 5e+3])
plan.W_X_ter = 10*np.array([1e-5, 1e-5, 5e+4, 1e1, 1e1, 1e+3, 1e+4, 1e+4, 1e+4])
plan.W_F = np.array(4*[1e1, 1e+1, 1.5e+1])
plan.nom_ht = 0.25
plan.rho = 5e+4
plan.ori_correction = [0.2, 0.8, 0.8]
plan.gait_horizon = 4.
#
# Gains
plan.kp = 3.0
plan.kd = 0.05

bound = plan 

#### Bound with Turning #######################################
bound_turn = BiconvexMotionParams("solo12", "bound_turn")
#
# Cnt
bound_turn.gait_period = 0.3
bound_turn.stance_percent = [0.5, 0.5, 0.5, 0.5]
bound_turn.gait_dt = 0.05
bound_turn.phase_offset = [0.0, 0.0, 0.5, 0.5]
#
# IK
bound_turn.state_wt = np.array([0., 0, 1e3] + [10, 10, 10] + [50.0] * (pin_robot.model.nv - 6) \
                        + [0.00] * 3 + [100, 10, 10] + [0.5] *(pin_robot.model.nv - 6))
#
bound_turn.ctrl_wt = [0.5, 0.5, 0.5] + [1, 1, 1] + [0.5] *(pin_robot.model.nv - 6)
#
bound_turn.swing_wt = [1e4, 1e4]
bound_turn.cent_wt = [5e+1, 5e+2]
bound_turn.step_ht = 0.07
bound_turn.reg_wt = [7e-3, 7e-5]
#
# Dyn
bound_turn.W_X =        np.array([1e-5, 1e-5, 5e+4, 1e1, 1e1, 1e+3, 5e+3, 1e+4, 5e+3])
bound_turn.W_X_ter = 10*np.array([1e-5, 1e-5, 5e+4, 1e1, 1e1, 1e+3, 1e+4, 1e+4, 1e+4])
bound_turn.W_F = np.array(4*[1e+1, 1e+1, 1.5e+1])
bound_turn.nom_ht = 0.25
bound_turn.rho = 5e+4
bound_turn.ori_correction = [0.2, 0.8, 0.8]
bound_turn.gait_horizon = 1.0
#
# Gains
bound_turn.kp = 3.0
bound_turn.kd = 0.05

############ Air_bound #################################################################
air_bound = BiconvexMotionParams("solo12", "air_bound")

# Cnt
air_bound.gait_period = 0.3
air_bound.stance_percent = [0.4, 0.4, 0.4, 0.4]
air_bound.gait_dt = 0.05
air_bound.phase_offset = [0.0, 0.0, 0.5, 0.5]

# IK
air_bound.state_wt = np.array([0., 0, 1e3] + [10, 10, 10] + [50.0] * (pin_robot.model.nv - 6) \
                        + [0.00] * 3 + [100, 10, 100] + [0.5] *(pin_robot.model.nv - 6))

air_bound.ctrl_wt = [0.5, 0.5, 0.5] + [1, 1, 1] + [0.5] *(pin_robot.model.nv - 6)

air_bound.swing_wt = [1e4, 1e4]
air_bound.cent_wt = [3*[5e+1], 6*[5e+2]] # modified from  [5e+1, 5e+2]
air_bound.step_ht = 0.07
air_bound.reg_wt = [7e-3, 7e-5]

# Dyn
air_bound.W_X =        np.array([1e-5, 1e-5, 5e+4, 1e1, 1e1, 1e+3, 5e+3, 1e+4, 5e+3])
air_bound.W_X_ter = 10*np.array([1e-5, 1e-5, 5e+4, 1e1, 1e1, 1e+3, 1e+4, 1e+4, 1e+4])
air_bound.W_F = np.array(4*[1e+1, 1e+1, 3e+1])
air_bound.nom_ht = 0.25
air_bound.rho = 5e+4
air_bound.ori_correction = [0.2, 0.8, 0.8]
air_bound.gait_horizon = 2.0

# Gains
air_bound.kp = 3.0
air_bound.kd = 0.05
