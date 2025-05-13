import pinocchio as pin
import numpy as np
import hppfcl

class Ground:
    def __init__(self):
        ground_geometry = hppfcl.Box(10, 10, 0.1)
        ground_placement = pin.SE3(np.eye(3), np.array([0, 0, -0.05]))
        
        self.gobject = pin.GeometryObject(
            "ground",
            0,
            ground_geometry,
            ground_placement
        )
        self.gobject.meshColor = np.array([0.5, 0.5, 0.5, 1])
        
        
if __name__ == "__main__":
    ground = Ground()