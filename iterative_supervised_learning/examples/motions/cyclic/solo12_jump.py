## Contains solo 12 gait params
## Author : Avadesh Meduri
## Date : 7/7/21

import numpy as np
from motions.weight_abstract import BiconvexMotionParams
from robot_properties_solo.config import Solo12Config

pin_robot = Solo12Config.buildRobotWrapper()
urdf_path = Solo12Config.urdf_path

#### jump #########################################
plan = BiconvexMotionParams("solo12", "Jump")


# Cnt
plan.gait_period = 0.5
plan.stance_percent = [0.3, 0.3, 0.3, 0.3]
plan.gait_dt = 0.05
plan.phase_offset = [0.7, 0.7, 0.7, 0.7]

# IK
plan.state_wt = np.array([0., 0, 10] + [1000] * 3 + [1.0] * (pin_robot.model.nv - 6) \
                        + [0.00] * 3 + [100] * 3 + [0.5] *(pin_robot.model.nv - 6))

plan.ctrl_wt = [0, 0, 1000] + [5e2, 5e2, 5e2] + [1.0] *(pin_robot.model.nv - 6)

plan.swing_wt = [1e4, 1e4]
plan.cent_wt = [0*5e+1, 5e+2]
plan.step_ht = 0.05
plan.nom_ht = 0.25
plan.reg_wt = [5e-2, 1e-5]

# Dyn
plan.W_X = np.array([1e-5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+4, 1e+4, 1e4])
plan.W_X_ter = 10*np.array([1e+5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+5, 1e+5, 1e+5])
plan.W_F = np.array(4*[1e+1, 1e+1, 1.5e+1])
plan.rho = 5e+4
plan.ori_correction = [0.2, 0.5, 0.4]
plan.gait_horizon = 3.0

# Gains
plan.kp = 2.5
plan.kd = 0.08

# Newly added
jump = plan
