""" 6 DOF equation of motion"""
import numpy as np
from . import math_function as mf
from scipy.integrate import ode


class SixDOF(object):
    """# 6 dimensional equation of motion of the rigid body.
    ## Instances
     ### x: state variables
      x[0:3]: position, inertial coordinate [m]
      x[3:6]: velocity, inertial coordinate [m/s]
      x[6:10]: quartanion [q0, q1, q2, q3]
      x[10:13]: angular velocity, body frame [rad/s]

     ### u: input variables
      u[0:3]: acceleration, inertial coordinate [m/s^2]
      u[3:6]: angular acceleration, body frame [rad/s^2]

     + t: time [s]
     + dt: step time [s]
     + integrator: integrator from scipy (inner instance)

    ## Methods
     + dxdt: derivative of the state variables
     + dqdt: derivative of the quartanion
     + post_process: called after integration
     + step: execute integration one time
     + normalize_quartanion
     ### set_input
      + set input to self.u
      + variables must be expressed in inertial coordinate
     ### set_input_body_frame
      + set input to self.u
      + variables must be expressed in body frame
     + set_position
     + set_velocity
     + set_angular_acceleration
     + set_quartanion
     + set_quartanion_from: set_quartanion form euler angles

    """

    def __init__(self, t0, dt):
        super(SixDOF, self).__init__()
        self.x = np.zeros((13), dtype=float) # pos, vel, quart, rot
        self.set_quartanion_from()
        self.u = np.zeros((6), dtype=float) # acc, angular acc
        self.t = t0
        self.dt = dt
        self.integrator = ode(f=self.dxdt)
        self.integrator.set_integrator('dopri5')
        self.integrator.set_initial_value(self.x)
        self.integrator.set_solout(self.post_process)

    def reset_state(self):
        self.x = np.zeros((13), dtype=float) # pos, vel, quart, rot
        self.set_quartanion_from()
        self.u = np.zeros((6), dtype=float) # acc, angular acc
        self.integrator = ode(f=self.dxdt)
        self.integrator.set_integrator('dopri5')
        self.integrator.set_initial_value(self.x, t=self.t + self.dt)
        self.integrator.set_solout(self.post_process)


    def dxdt(self, t, y):
        self.t = t
        dxdt =self.x[3:6]
        dvdt = self.u[0:3]
        dqdt = self.dqdt()
        dpdt = self.u[3:6]
        # print(dxdt, dvdt, dqdt, dpdt)
        return np.hstack((dxdt, dvdt, dqdt, dpdt))

    def dqdt(self):
        """calculate derivative of quartanion"""

        q = self.get_quartanion()
        w = mf.convert_vector_body_to_inertial(self.get_angular_velocity(), q)
        w_hat = 0.5 * np.array([
            [0.0,  -w[0], -w[1], -w[2]],
            [w[0],   0.0, -w[2],  w[1]],
            [w[1],  w[2],   0.0, -w[0]],
            [w[2], -w[1],  w[0],   0.0]
        ])
        return np.dot(w_hat, q)

    def post_process(self, t, y):
        self.x = self.integrator.y
        self.normalize_quartanion()

    def step(self, input):
        self.set_input(input)
        self.x = self.integrator.integrate(self.t+self.dt)

    def normalize_quartanion(self):
        self.x[6:10] /= np.linalg.norm(self.x[6:10])

    def set_time(self, t):
        self.t = t

    def set_input(self, input):
        self.u = input

    def set_input_body_frame(self, input):
        """ not used beacause gravity is inertial but force is local"""
        self.u = np.hstack((self.get_inertial_acceleration(input[0:3]), input[3:6]))

    def set_position(self, position):
        self.x[0:3] = position

    def set_velocity(self, velocity):
        self.x[3:6] = velocity

    def set_acceleration(self, acceleration):
        self.u[0:3] = acceleration

    def set_quartanion(self, quartanion):
        self.x[6:10] = quartanion

    def set_quartanion_from(self, roll=0.0, pitch=0.0, yaw=0.0):
        """set quartanion from euler angles."""
        self.x[6:10] = mf.quartanion_from(roll, pitch, yaw)

    def set_angular_velocity(self, p):
        self.x[10:13] = p

    def set_angular_acceleration(self, a):
        self.u[3:6] = a

    def get_position(self):
        return self.x[0:3]

    def get_velocity(self):
        return self.x[3:6]

    def get_acceleration(self):
        return self.u[0:3]

    def get_quartanion(self):
        return self.x[6:10]

    def get_euler_angle(self):
        return mf.euler_from(self.x[6:10])

    def get_angular_velocity(self):
        return self.x[10:13]

    def get_angular_acceleration(self):
        return self.u[3:6]

    def get_time(self):
        return self.integrator.t

    def get_status(self):
        return self.x

    def get_input(self):
        return self.u

    def get_inertial_acceleration(self, a_body_frame):
        return mf.convert_vector_body_to_inertial(a_body_frame, self.get_quartanion())
