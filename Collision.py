import pinocchio as pin
import numpy as np
import warnings

from Robot_Dora2 import Robot_Dora2
from Ground import Ground


warnings.filterwarnings("ignore", category=UserWarning, message="This function has been marked as deprecated and will be removed in a future release.")

class Collision:
    def __init__(self, robot:Robot_Dora2, ground:Ground):
        self.robot = robot
        self.ground = ground
        
        self.rmodel = robot.model
        self.rdata = self.rmodel.createData()
        
        self.collision_model = robot.gmodel
        self.collision_model.addGeometryObject(ground.gobject)
        self.addCollisionPairs()
        self.collison_data = self.collision_model.createData()
        
        
    def addCollisionPairs(self):
        pairs = [
            ['base_link_0', 'ground'],
            
            ['l_arm_wrist_yaw1_Link_0', 'ground'],
            ['l_arm_claw_link_0', 'ground'],
            ['l_leg_hip_yaw_Link_0', 'ground'],
            ['l_leg_hip_pitch_Link_0', 'ground'],
            ['l_leg_knee_Link_0', 'ground'],
            ['l_leg_ankle_roll_Link_0', 'ground'],
            
            ['r_arm_wrist_yaw1_Link_0', 'ground'],
            ['r_arm_claw_link_0', 'ground'],
            ['r_leg_hip_yaw_Link_0', 'ground'],
            ['r_leg_hip_pitch_Link_0', 'ground'],
            ['r_leg_knee_Link_0', 'ground'],
            ['r_leg_ankle_roll_Link_0', 'ground'],
        ]
        
        for (n1, n2) in pairs:
            collision_pair = pin.CollisionPair(self.collision_model.getGeometryId(n1), self.collision_model.getGeometryId(n2))
            self.collision_model.addCollisionPair(collision_pair)
            
    def computeCollisions(self,q,vq=None):
        res = pin.computeCollisions(self.rmodel,self.rdata,self.collision_model,self.collison_data,q,False)
        pin.computeDistances(self.rmodel,self.rdata,self.collision_model,self.collison_data,q)
        pin.computeJointJacobians(self.rmodel,self.rdata,q)
        if vq is not None:
            pin.forwardKinematics(self.rmodel,self.rdata,q,vq,0*vq)
        return res
    
    def getCollisionList(self):
        '''
        Return a list of triplets [ index,collision,result ] where 
        index:      the index of the collision pair
        colision:   gmodel.collisionPairs[index]
        result:     gdata.collisionResults[index]
        '''
        return [ [ir, self.collision_model.collisionPairs[ir], r]
                 for ir,r in enumerate(self.collison_data.collisionResults) if r.isCollision() ]
        
    def getCollisionJacobian(self, collision_pair, collision_result):
        '''
        return the jacobian of the collision point, which maps the generalized velocity to the linear velocity expressed in the local frame
        '''
        contact = collision_result.getContact(0)
        
        # get the geometry object which contacts with the ground(ground is always the second object)
        geom_obj = self.collision_model.geometryObjects[collision_pair.first]
        
        # notice contact.normal points from o1 to o2, which is the opposite direction of the desired direction
        oMc = pin.SE3(pin.Quaternion.FromTwoVectors(np.array([0, 0, 1]), -contact.normal).matrix(), contact.pos)
        
        joint = geom_obj.parentJoint
        oMj = self.rdata.oMi[joint]
        
        cMj = oMc.inverse() * oMj
        
        J_joint = pin.getJointJacobian(self.rmodel, self.rdata, joint, pin.ReferenceFrame.LOCAL)    # is local correct?
        J = cMj.action @ J_joint # Jc = [Ad_cMj] * Jj
        return J[:3, :]
    
    def getAllCollisionJacobian(self, collisions=None):
        if collisions is None:
            collisions = self.getCollisionList()
        
        if(len(collisions) == 0):
            return np.ndarray([0,self.rmodel.nv])
        
        J = np.vstack([self.getCollisionJacobian(collision_pair, collision_result) for (index,collision_pair,collision_result) in collisions])
        return J
    
    def getAllCollisonDistances(self, collisions=None):
        if collisions is None:
            collisions = self.getCollisionList()
        
        if(len(collisions) == 0):
            return np.array([])
        
        dist = np.array([self.collison_data.distanceResults[i].min_distance for (i, c, r) in collisions])
        return dist
    
    def getAllCollisionPoints(self, collisions=None):
        # get all collision points in world frame
        if collisions is None:
            collisions = self.getCollisionList()
            
        if(len(collisions) == 0):
            return np.array([])
        
        points = np.concatenate([self.collison_data.collisionResults[i].getContact(0).pos for (i, c, r) in collisions])
        return points
    
    def getCollisionInfo(self):
        # get the collision info(Yes/No) for all collision pairs
        for k in range(len(self.collision_model.collisionPairs)):
            cr = self.collison_data.collisionResults[k]
            cp = self.collision_model.collisionPairs[k]
            print(
                "collision pair:",
                self.collision_model.geometryObjects[cp.first].name,
                ",",
                self.collision_model.geometryObjects[cp.second].name,
                "- collision:",
                "Yes" if cr.isCollision() else "No",
            )
        
if __name__ == "__main__":
    robot = Robot_Dora2()
    ground = Ground()
    collision = Collision(robot, ground)
    
    q = robot.q0
    q[2] = 0.785 # 0.785时脚踝刚好接触地面
    is_contact = collision.computeCollisions(q)
    print("is_contact:", is_contact)
    
    # collision_list = collision.getCollisionList()
    # index, collision_pair, collision_result = collision_list[0]
    # contact = collision_result.getContact(0)
    # print("c1:", collision.collision_model.geometryObjects[collision_pair.first].name)
    # print("c2:", collision.collision_model.geometryObjects[collision_pair.second].name)
    # joint = collision.collision_model.geometryObjects[collision_pair.first].parentJoint
    # print("oMj:", collision.rdata.oMi[joint])
    # print("o1", contact.o1)
    # print("o2", contact.o2)
    # print("normal", contact.normal)
    # print("pos", contact.pos)
    
    #joint 7
    # Jb = pin.getJointJacobian(collision.rmodel, collision.rdata, 7, pin.ReferenceFrame.LOCAL)
    # print("Jb:", Jb)
    # Js = pin.getJointJacobian(collision.rmodel, collision.rdata, 7, pin.ReferenceFrame.WORLD)
    # print("Js:", Js)
    # sMb = collision.rdata.oMi[7]
    # print("[Ad_sMb]Jb", sMb.action @ Jb)
    distance = collision.getAllCollisonDistances()
    print("distance:", distance)
    