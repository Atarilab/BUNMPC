## this file contains a simulation env with pybullet
## Author : Avadesh Meduri
## Date : 7/05/2021

from bullet_utils.env import BulletEnvWithGround
import pybullet
import subprocess#
import numpy as np

class PyBulletEnv:

    def __init__(self, robot, q0, v0, server=pybullet.DIRECT, z_height_init=0):
        """
        server: PyBullet server mode. pybullet.GUI creates a graphical frontend
          using OpenGL while pybullet.DIRECT does not. Defaults to pybullet.GUI."""
        print("loading bullet")
        #self.env = BulletEnvWithGround(server, z_height_init=z_height_init) 
        self.env = BulletEnvWithGround(server)
        self.robot = self.env.add_robot(robot())
        self.robot.reset_state(q0, v0)
        
    def reset_robot_state(self, q0, v0):
        self.robot.reset_state(q0, v0)

    def get_state(self):
        """
        returns the current state of the robot
        """
        q, v = self.robot.get_state()
        return q, v

    def get_imu_data(self):
        lin_acc = self.robot.get_base_imu_linacc()
        ang_vel = self.robot.get_base_imu_angvel()
        return lin_acc, ang_vel

    def send_joint_command(self, tau):
        """
        computes the torques using the ID controller and plugs the torques
        Input:
            tau : input torque
        """
        self.robot.send_joint_command(tau)
        self.env.step() # You can sleep here if you want to slow down the replay

    def get_current_contacts(self):
        """
        :return: an array of boolean 1/0 of end-effector current status of contact (0 = no contact, 1 = contact)
        """
        contact_configuration = self.robot.get_force()[0]
        return contact_configuration

    def get_ground_reaction_forces(self):
        """
        returns ground reaction forces from the simulator
        """
        forces = self.robot.get_contact_forces()
        return forces

    def get_contact_positions_and_forces(self):
        import numpy as np
        foot_link_ids = tuple(self.robot.bullet_endeff_ids)
        contact_forces = [np.zeros(3) for _ in range(len(foot_link_ids))]
        contact_positions = [np.zeros(3) for _ in range(len(foot_link_ids))]
        all_contacts = pybullet.getContactPoints(bodyA=self.robot.robot_id)
        for contact in all_contacts:
            (unused_flag, body_a_id, body_b_id, link_a_id, unused_link_b_id,
            unused_pos_on_a, unused_pos_on_b, contact_normal_b_to_a, unused_distance,
            normal_force, friction_1, friction_direction_1, friction_2,
            friction_direction_2) = contact
            if body_b_id == body_a_id:
                continue
            if link_a_id+1 in foot_link_ids:
                link_a_id += 1
                normal_force = np.array(contact_normal_b_to_a) * normal_force
                friction_force = np.array(friction_direction_1) * friction_1 + np.array(
                    friction_direction_2) * friction_2
                force = normal_force + friction_force
                force_norm = np.linalg.norm(force)
                toe_link_order = foot_link_ids.index(link_a_id)
                if force_norm >= 0.5:
                    contact_forces[toe_link_order] += force
                    # CHANGES: this causes jumps in position when one ee has 2 contact returns
                    # contact_positions[toe_link_order] += unused_pos_on_a
                    contact_positions[toe_link_order] = unused_pos_on_a
            else:
                continue
        return contact_positions, contact_forces

    def start_video_recording(self, file_name):
        """Starts video recording and save as a mp4 file.

        Args:
            file_name (str): The absolute path of the file to be saved.
        """
        self.file_name = file_name
        self.logging_id = pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name
        )

    def stop_video_recording(self):
        """Stops video recording if any.
        """
        if hasattr(self, "file_name") and hasattr(self, "logging_id"):
            pybullet.stopStateLogging(
                loggingId=self.logging_id
            )
            
    def capture_image_frame(self, width=640, height=480, fov=60, near=0.1, far=10.0, cam_dist=1, cam_yaw=75, cam_pitch=-20, target=[0.5, 0.0, 0.0]):
        """capture current pybullet env (works in both GUI and server mode)

        Args:
            width (int, optional): frame width. Defaults to 640.
            height (int, optional): frame height. Defaults to 480.
            fov (int, optional): camera view angle. Defaults to 60.
            near (float, optional): nearest object to capture. Defaults to 0.1.
            far (float, optional): furthest object to capture. Defaults to 10.0.
            cam_dist (int, optional): distance of camera to focal point. Defaults to 1.
            cam_yaw (int, optional): camera yaw position. Defaults to 75.
            cam_pitch (int, optional): camera pitch position. Defaults to -20.
            target (list, optional): camera aim position. Defaults to [0.5, 0.0, 0.0].

        Returns:
            numpy_array [height, width, 3]: rgb image 
        """        
        # Set the camera parameters
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)
        projection_matrix = pybullet.computeProjectionMatrixFOV(fov, float(width)/height, near, far)
        
        # Capture the image
        img_arr = pybullet.getCameraImage(width, height, view_matrix, projection_matrix, shadow=True,
                                            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    
        # Extract the RGB data
        rgb = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
        return rgb

    def apply_external_force(self, F, M):
        """Apply external disturbance to the base link.

        Args:
            F (np.array(3)): applied force.
            M (np.array(3)): applied moment.
        """
        pybullet.applyExternalForce(self.robot.robot_id, -1,
                  F, M, pybullet.LINK_FRAME)

    def generate_terrain(self, feasible_footholes):
        # TODO: Last testing failed. Problem not yet identified
        import numpy as np
        #from perlin_noise import PerlinNoise
        height_field_terrain_shape = None
        # remove the default plane
        pybullet.removeBody(0)

        # terrain patch properties
        length_per_index = 0.05 # fixed as only then the contact forces are simulated properly
        patch_length_x = 10 # subject to the trajectory length
        patch_length_y = 10 # subject to the trajectory length
        numHeightfieldRows = int(patch_length_x / length_per_index)
        numHeightfieldColumns = int(patch_length_y / length_per_index)
        terrainMap = np.zeros((numHeightfieldRows,numHeightfieldColumns))

        # hilly terrain generated through perlin noise
        if True:
            #noise = PerlinNoise(octaves=.0001)
            for i in range(numHeightfieldRows):
                for j in range(numHeightfieldColumns):
                    h = 0.1 * noise([i/numHeightfieldRows, j/numHeightfieldColumns])
                    terrainMap[i][j] = h


        heightfieldData = terrainMap.T.flatten()
        if height_field_terrain_shape == None:
            # first time, define the height field
            height_field_terrain_shape = pybullet.createCollisionShape(shapeType = pybullet.GEOM_HEIGHTFIELD,
                                                                            meshScale=[ length_per_index , length_per_index ,1],
                                                                            heightfieldTextureScaling=(numHeightfieldRows-1)/2,
                                                                            heightfieldData=heightfieldData,
                                                                            numHeightfieldRows=numHeightfieldRows,
                                                                            numHeightfieldColumns=numHeightfieldColumns)
        else:
            # from second time, simply update the height field
            # to prevent memory leak
            height_field_terrain_shape = pybullet.createCollisionShape(shapeType = pybullet.GEOM_HEIGHTFIELD,
                                                                            meshScale=[ length_per_index , length_per_index ,1],
                                                                            heightfieldTextureScaling=(numHeightfieldRows-1)/2,
                                                                            heightfieldData=heightfieldData,
                                                                            numHeightfieldRows=numHeightfieldRows,
                                                                            numHeightfieldColumns=numHeightfieldColumns,
                                                                            replaceHeightfieldIndex=height_field_terrain_shape)


        terrain_id  = pybullet.createMultiBody(baseMass = 0, baseCollisionShapeIndex = height_field_terrain_shape)
        pybullet.changeVisualShape(terrain_id, -1, rgbaColor=[.0, 1., .0, 1.])
        
    def show_desired_contact_locations(self, cnt_plan):
        
        for j in range(len(cnt_plan[0])):
            if cnt_plan[0, j, 0] == 1:
                pybullet.addUserDebugPoints([cnt_plan[0, j, 1:]], [[1, 0, 0]], pointSize=8.0, lifeTime=0.5)
