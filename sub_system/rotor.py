"""Dynamics of the rotor."""
import numpy as np
import numpy.linalg as nl
from scipy.integrate import ode



class Dynamics(object):
    """Dynamics for Rotor."""

    def __init__(self):
        super(Dynamics, self).__init__()
        self.speed = 0.0 # radian per second
        self.direction = np.array([0.0, 0.0, -1.0]) # body frame
        self.displacement = np.array([1.0, 0.0, 0.0]) # displacement
        self.lmd = 1.0  # rotation direction: 1 = 'cw', -1 = 'ccw'
        self.ct = 0.0050 # thrust coefficient
        self.cq = 0.01  # torque coefficient
        self.m = 0.005 # weight [kg]

        self.input = 0.0  # control input: [0.0, 1.0]
        self.tau = 0.125  # time constant
        self.max_speed = 1000

        self.t = 0.0
        self.dt = 0.01
        self.integrator = ode(self.dsdt).set_integrator('dopri5')
        self.integrator.set_initial_value(0.0, 0.0)
        self.integrator.set_f_params(1.0)
        self.integrator.set_solout(self.set_int_speed)

    def show_settings(self):
        print("--- Rotor Specification ---")
        v = vars(self)
        for key, value in v.items():
            print(f'{key}: {value}')

    def integrate(self):
        self.integrator.integrate(self.integrator.t + self.dt)

    def dsdt(self, t, y, input):
        """derivative of motor speed."""
        u = 0.0
        if input > 1.0:
            u = 1.0
        elif input < 0.0:
            u = 0.0
        else:
            u = input
        return 1/self.tau * (self.max_speed * u - y)

    def get_total_force(self):
        return self.thrust() + self.drag_force()

    def get_total_torque(self):
        # print(self.torque() , self.drag_torque())
        return self.torque() + self.drag_torque()

    def thrust(self):
        return np.dot(self.ct * self.speed, self.direction)

    def torque(self):
        return np.cross(self.displacement, self.get_total_force())

    def drag_force(self):
        return np.array([0.0, 0.0, 0.0])

    def drag_torque(self): # Attention! Flapping dynamics is NOT included!
        return np.dot(self.lmd * self.cq * self.speed, self.direction)


    # setter

    def set_rotor_speed(self, speed):
        self.speed = speed

    def set_input(self, input):
        """input: [0.0, 1.0]"""
        self.input = input
        self.integrator.set_f_params(input)

    def set_displacement(self, displacement):
        self.displacement = np.array(displacement)

    def set_lambda(self, value):
        self.lmd = value

    def set_direction(self, dir):
        self.direction = dir

    def reset_speed(self):
        self.speed = 0.0
        self.input = 0.0
        self.integrator.set_initial_value(self.speed, t=self.t + self.dt)

    # getter

    def get_t(self):
        return self.integrator.t

    def get_input(self):
        return self.input

    def set_int_speed(self, t, y):
        self.t = t
        self.speed = y[0]

    def get_speed(self):
        return self.speed
