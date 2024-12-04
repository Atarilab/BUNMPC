import pybullet as p
from bullet_utils.env import BulletEnvWithGround
from robot_properties_go2.go2wrapper import Go2Robot

env = BulletEnvWithGround(p.GUI)
robot = env.add_robot(Go2Robot)