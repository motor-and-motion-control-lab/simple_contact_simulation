import pinocchio as pin
import numpy as np
import cvxpy as cp


from Robot_Dora2 import Robot_Dora2
from Ground import Ground
from Collision import Collision

# test
from pinocchio.visualize import MeshcatVisualizer
import meshcat

class Dynamics:
    def __init__(self, robot:Robot_Dora2, ground:Ground, h, mu):
        self.robot = robot
        self.collision = Collision(robot, ground)
        self.h = h
        self.mu = mu
        self.nv = self.robot.model.nv
        
        # state
        self.q = None
        self.v = None
        
        self.u = None
        
        # self.is_contact = False
        self.nc = 0
        self.penetr_dis = None
        
        # ref-vec 
        self.eps = 1e-6
        self.stiffness = 2*self.h
        self.R_min = 1e-6
        
        self.M_inv = None
        self.c = None
        self.Jc = None
        
        self.A = None
        self.contact_v_pre = None
        self.contact_v_ref = None
        self.R = None   # dimension: 3*nc x 3*nc
        self.K = None
        self.B = None
        
        self.f = None
        
    def calc_inv_M(self):
        self.M_inv =  pin.computeMinverse(self.robot.model ,self.robot.data, self.q)
    
    def calc_c(self):
        self.c = pin.rnea(self.robot.model, self.robot.data, self.q, self.v, np.zeros(self.robot.model.nv))
        
    def calc_collision(self):
        is_contact = self.collision.computeCollisions(self.q)
        if(is_contact):
            self.penetr_dis = self.collision.getAllCollisonDistances()
            self.nc = len(self.penetr_dis)
        return is_contact
        
    def calc_Jc(self):
        self.Jc = self.collision.getAllCollisionJacobian()
        
    def calc_A(self):
        self.A = self.Jc @ self.M_inv @ self.Jc.T
        
    def calc_contact_v_pre(self):
        # contact speed expressed in contact frame
        self.contact_v_pre = self.Jc @ (self.v + self.M_inv @ self.u * self.h)
        
    def calc_R(self):
        self.R = np.diag(
            np.maximum(
                np.diag(self.A) * self.eps,
                [self.R_min]*self.nc*3
            )
        )
        
    def calc_BK(self):
        self.B = np.diag([2*(1+self.eps)*(1/self.stiffness)] * self.nc * 3)
        
        k_diagonal = np.zeros(self.nc * 3)
        k_diagonal[2::3] = (1+self.eps) * (1/(self.stiffness ** 2))
        self.K = np.diag(k_diagonal)
         
    def calc_contact_v_ref(self):
        x_dot = self.Jc @ self.v
        x = np.zeros(self.nc * 3)
        x[2::3] = self.penetr_dis
        self.contact_v_ref = x_dot - self.h * (self.B @ x_dot + self.K @ x) 
        
    def calc_f(self):
        # notice that this f is in the contact frame
        Q = self.A + self.R
        p = self.contact_v_pre - self.contact_v_ref
        f = cp.Variable(3*self.nc)
        
        objective = cp.Minimize(cp.quad_form(f, Q) + p.T @ f)
        
        constraints = []
        for i in range(self.nc):
            fx, fy, fz = f[3*i:3*i+3]
            constraints.append(fz>=0)
            constraints.append(cp.SOC(fz/self.mu, cp.hstack([fx, fy])))
        
        prob = cp.Problem(objective, constraints)
        # prob.solve(solver = cp.MOSEK, verbose=True)
        # prob.solve(solver = cp.MOSEK)
        prob.solve(solver = cp.ECOS)
        self.f = f.value
        
    def calc_next_v(self, is_contact):
        if(is_contact):
            v_next = self.v + self.M_inv @ ((self.u - self.c) * self.h + self.Jc.T @ self.f)
        else:
            v_next = self.v + self.M_inv @ ((self.u - self.c) * self.h)
        return v_next
    
    def forward_sim(self, q, v, u):
        self.q = q
        self.v = v
        self.u = u
        
        self.calc_inv_M()
        self.calc_c()
        is_contact = self.calc_collision()
        
        if(is_contact):
            self.calc_Jc()
            self.calc_A()
            self.calc_R()
            self.calc_BK()
            self.calc_contact_v_pre()
            self.calc_contact_v_ref()
            self.calc_f()
            v_next = self.calc_next_v(is_contact)
        else:
            v_next = self.calc_next_v(is_contact)
        
        q_next = pin.integrate(self.robot.model, self.q, v_next * self.h)
        return q_next, v_next
        
    
if __name__ == "__main__":
    robot = Robot_Dora2()
    ground = Ground()
    dynamics = Dynamics(robot, ground, 0.005, 0.2)
    
    q = robot.q0
    q[2] = 2
    v = np.zeros(robot.model.nv)
    u = np.zeros(robot.model.nv)
    
    time = 0
    for i in range(100):
        q, v = dynamics.forward_sim(q, v, u)
        time += dynamics.h
        
        pin.forwardKinematics(robot.model, robot.data, q, v)
        pin.updateFramePlacements(robot.model, robot.data)
        pin.computeJointJacobians(robot.model, robot.data)
        
        J_com = pin.jacobianCenterOfMass(robot.model, robot.data, q)
        v_com = J_com @ v
        
        # print("time:", time)
        # print("q:", q)
        # print("v:", v)
        # print("f", dynamics.f)
        # print("v_com:", v_com)
        # print("-"*20)
        