import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import os

# warnings.filterwarnings("ignore", category=UserWarning, message="You passed package dir(s) via argument geometry_model and provided package_dirs.")

class Robot_Dora2:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_model_path = os.path.join(current_dir, "urdf/dora2_erect_arm.urdf")
        mesh_dir = os.path.join(current_dir, "meshes_7Dof")
        
        self.robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())
        self.model = self.robot.model
        self.gmodel = self.robot.collision_model
        self.vmodel = self.robot.visual_model
        
        self.data = self.robot.data
        self.gdata = self.gmodel.createData()
        self.vdata = self.vmodel.createData()
        
        self.q0 = pin.neutral(self.model)
        self.q0[:3] = [0, 0, 1]
        self.v0 = np.zeros(self.model.nv)
        self.collisionPairs = []


if __name__ == "__main__":
    robot = Robot_Dora2()
    print(robot.q0)