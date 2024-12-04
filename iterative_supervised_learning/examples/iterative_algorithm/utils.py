import numpy as np
import math
import sys
import pinocchio as pin


def quaternion_to_euler_angle(x, y, z, w):
    """convert quaternion orientation to euler orientation

    Args:
        x (_type_): x
        y (_type_): y
        z (_type_): z
        w (_type_): w

    Returns:
        X, Y, Z: Roll, Pitch, Yaw
    """    
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def construct_cc_goal(episode_length, n_eff, contact_schedule, com, goal_horizon=1, sim_dt=0.001, start_step=0):
    """Construct contact conditioned goal

    Args:
        episode_length (_type_): simulation length (In sim step NOT sim time!)
        n_eff (_type_): number of end effectors
        contact_schedule (_type_): desired contact schedule from contact planner
        com (_type_): current robot center of mass
        goal_horizon (int, optional): goal horizon. Defaults to 1.
        sim_dt (float, optional): simulation dt. Defaults to 0.001.
        start_step (int, optional): starting sim step (NOT sim time!). Defaults to 0.

    Returns:
        desired_goal: constructed cc goal [time to contact, x, y] * n_eff
    """    
    # get maximum allowed episode length (normally less than episode length)
    end_time = episode_length
    for ee in range(n_eff):
        max_time = np.max(contact_schedule[ee, :, 0])
        end_time = int(min(end_time, max_time))
        
    desired_goal = np.zeros((end_time - start_step, 3*n_eff*goal_horizon))

    for t in range(start_step, end_time):
        for gh in range(goal_horizon):
            for ee in range(n_eff):
                # get current ee in contact
                ee_index = ee_contact_index(t, contact_schedule[ee, :, 0])
                ee_index += gh  # add goal horizon
                # construct goal
                desired_goal[t - start_step, 3*n_eff*gh + 3*ee : 3*n_eff*gh + 3*(ee+1)] = base_wrt_goal(t, com[t-start_step, :], contact_schedule[ee, ee_index, :], sim_dt=sim_dt)
    
    return desired_goal

def base_wrt_goal(time, base_pos, goal_contact, sim_dt=0.001):
    """calculate current com position wrt next contact location

    Args:
        time (_type_): current sim step (NOT sim time)
        base_pos (_type_): current CoM position
        goal_contact (_type_): next contact location
        sim_dt (float, optional): simulation dt. Defaults to 0.001.

    Returns:
        out: return [time to contact, x, y]
    """    
    sim_dt = 1.0
    out = np.hstack((sim_dt*(goal_contact[0] - time), base_pos[:-1] - goal_contact[1:-1]))
    
    return out

def ee_contact_index(time, ee_contact_schedule):
    """returns index of the next contact switch in the contact schedule

    Args:
        time (_type_): current sim step (NOT sim time)
        ee_contact_schedule (_type_): contact schedule

    Returns:
        out: index of next contact
    """    
    out = 0
    for sw in range(len(ee_contact_schedule)-1):
        if time>=ee_contact_schedule[sw] and time<ee_contact_schedule[sw+1]:
            out = sw+1
            break
    return out

def construct_contact_schedule(new_contact_pos, n_eff):
    """construct contact schedule from contact positions

    Args:
        new_contact_pos (_type_): new contact position
        n_eff (_type_): number of end effector

    Returns:
        out: contact schedule
    """    
    out = np.zeros((n_eff, len(new_contact_pos[:, 0]), 4))
    ee_index = np.zeros(n_eff, int)
    for i in range(len(new_contact_pos[:, 0])):
        cur_ee = int(new_contact_pos[i, 0])  # index of the current ee (ee: end effector)
        out[cur_ee, ee_index[cur_ee]:ee_index[cur_ee]+1, :] = new_contact_pos[i, 1:]
        ee_index[cur_ee] += 1
    return out

def get_plan(gait):
    """get gait plan

    Args:
        gait (str): desired gait

    Returns:
        plan: gait plan
    """    
    if gait == "bound":
        from motions.cyclic.solo12_bound import plan
    elif gait == "trot":
        from motions.cyclic.solo12_trot import plan
    elif gait == "jump":
        from motions.cyclic.solo12_jump import plan
    else:
        sys.exit("gait does not exist, make sure to choose a proper gait")
    return plan

def get_des_velocities(vx_des_max, vx_des_min, vy_des_max, vy_des_min, w_des_max, w_des_min, gait, dist='uniform'):
    """sample desired command velocity
    desired z velocity is always 0.0

    Args:
        vx_des_max (_type_): vx max
        vx_des_min (_type_): vx min
        vy_des_max (_type_): vy max
        vy_des_min (_type_): vy min
        w_des_max (_type_): yaw max
        w_des_min (_type_): yaw min
        gait (_type_): gait
        dist (str, optional): sampling distribution (normal or uniform). Defaults to 'uniform'.

    Returns:
        v_des, w_des: desired v and w
    """    
    
    if dist == 'uniform':
        v_des = np.array([np.random.uniform(vx_des_min, vx_des_max), np.random.uniform(vy_des_min, vy_des_max), 0.0]) # m/s
        w_des = np.random.uniform(w_des_min, w_des_max) #rad/s
        
    elif dist == 'normal':
        v_des = np.array([np.random.normal(loc=vx_des_max, scale = vx_des_max/4), np.random.normal(loc=0, scale = vy_des_max), 0.0]) # m/s            
        w_des = np.random.uniform(w_des_min, w_des_max) #rad/s
    else:
        v_des = None
        w_des = None
    
    # CHANGES: commented out random direction change!
    # randomly change direction of v_des and w_des
    # if v_des is not None:
    #     if np.random.uniform(0, 1) < 0.5:
    #         v_des[0] = -v_des[0]
    #     if np.random.uniform(0, 1) < 0.5:
    #         v_des[1] = -v_des[1]
    
    if w_des is not None:
        if np.random.uniform(0, 1) < 0.5:
            w_des = -w_des
    
    # if gait == 'bound':
    #     v_des[1] = 0.0
    
    return v_des, w_des

def get_estimated_com(pin_robot, q0, v0, v_des, end_time, sim_dt, plan):
    """estimate CoM trajectory

    Args:
        pin_robot (_type_): robot description
        q0 (_type_): initial robot configuration
        v0 (_type_): initial robot velocity
        v_des (_type_): desired velocity
        end_time (_type_): end time for trajectory
        sim_dt (_type_): sim dt
        plan (_type_): gait plan

    Returns:
        estimated_com: estimated com trajectory
    """    
    
    # perform pinocchio forward comput
    pin.forwardKinematics(pin_robot.model, pin_robot.data, q0, v0)
    pin.updateFramePlacements(pin_robot.model, pin_robot.data)
    
    # get robot com
    com = pin.centerOfMass(pin_robot.model, pin_robot.data, q0, v0)
    com = np.round(com, 3)
    
    # set com velocity
    vcom = v_des[0:2]
    estimated_com = np.zeros((end_time, 3))
    
    for i in range(0, end_time):
        # move com with contact velocity and get estimated position. com height is set to 0.0
        estimated_com[i] = np.hstack((com[:2] + i*sim_dt*vcom, np.array(0.0)))
        
    return estimated_com

def compute_vc_mse(des_v, des_w, actual_v, actual_w):
    """calculate velocity tracking mean squared error

    Args:
        des_v (_type_): desired velocity
        des_w (_type_): desired yaw
        actual_v (_type_): actual velocity
        actual_w (_type_): actual yaw

    Returns:
        vx_error, vy_error, w_error: mse
    """    
    vx_error = np.mean(np.square(actual_v[:, 0] - des_v[0]))
    vy_error = np.mean(np.square(actual_v[:, 1] - des_v[1]))
    w_error = np.mean(np.square(actual_w - des_w))
    
    return vx_error, vy_error, w_error

def rotate_jacobian(sim, jac, index):
    """change jacobian frame

    Args:
        sim (_type_): simulation object
        jac (_type_): jacobian
        index (_type_): ee index

    Returns:
        jac: rotated jacobian
    """    
    world_R_joint = pin.SE3(sim.pin_robot.data.oMf[index].rotation, pin.utils.zero(3))
    return world_R_joint.action @ jac

def get_phase_percentage(sim_step, sim_dt, gait):
    """get current gait phase percentage

    Args:
        sim_step (_type_): current sim step (not sim time!)
        sim_dt (_type_): sim dt
        gait (_type_): desired gait

    Returns:
        phi: gait phase. between 0-1
    """    
    plan = get_plan(gait)
    phi = ((sim_step*sim_dt) % plan.gait_period)/plan.gait_period
    return phi

def get_vc_gait_value(gait):
    """get gait type value for velocity goal conditioning
    
    trot: 1
    jump: 2
    bound: 3

    Args:
        gait (_type_): desired gait

    Returns:
        gait_value: integer value representing the gait type
    """    
    gait_value = 0.0
    if gait == "trot":
        gait_value = 1.
    elif gait == "jump":
        gait_value = 2.
    elif gait == "bound":
        gait_value = 3.
        
    return gait_value
        