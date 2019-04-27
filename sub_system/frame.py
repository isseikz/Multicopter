"""Dynamics of the multicopter"""

import numpy as np
from . import math_function as mf
from . import equation_of_motion as em
from . import rotor

class Multicopter(object):
    """Multicopter model expression.
    Quadrotor model is default.
    """

    def __init__(self, n_rotor=4):
        super(Multicopter, self).__init__()
        self.m = 1.0
        self.I = np.eye(3)
        self.I_inv = np.linalg.inv(self.I)
        self.dynamics = em.SixDOF(0.0, 0.01)
        self.gravity = np.array([0.0, 0.0, 9.81])
        self.r = [rotor.Dynamics() for i in range(n_rotor)] # rotor instances
        self.ur = np.zeros(n_rotor, dtype=float) # inputs to the rotors
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    def read_regular_settings(self):
        for i, rotor in enumerate(self.r):
            x = np.float((i % 2)*2 -1)
            y = np.float((i // 2)*2 -1)
            rotor.set_displacement((x,y,0.0))
            rotor.set_lambda(x*y)

    def integrate(self, arr_inputs):
        self.set_inputs(arr_inputs)

        self.reset_force_and_torque()

        self.add_propeller_force_and_torque()

        self.add_gravity_force()
        self.add_aerodynamic_drag()
        self.add_gyroscopic_torque()

        inputs = np.hstack((self.force/self.m, np.dot(self.I_inv, self.torque)))
        self.dynamics.step(inputs)
        self.doesnt_sink_to_ground()

    def add_propeller_force_and_torque(self):
        for i, rotor in enumerate(self.r):
            rotor.set_input(self.ur[i])
            rotor.integrate()
            self.force += rotor.get_total_force()
            self.torque += rotor.get_total_torque()
        self.force = mf.convert_vector_body_to_inertial(
            self.force,
            self.get_quartanion()
        )

    def add_gyroscopic_torque(self):
        w = self.dynamics.get_angular_velocity()
        self.torque += np.cross(w, np.dot(self.I, w))

    def add_gravity_force(self):
        self.force += np.dot(self.m, self.gravity)

    def add_aerodynamic_drag(self): # TODO: Implement!!!
        pass

    def show_settings(self):
        print("--- Multicopter System Specification ---")
        print(f"Weight: {self.m}")
        print(f'Moment of inertia:')
        for row in self.I:
            print(row)
        print(f'Inversed matrix of Moment of inertia:')
        for row in self.I_inv:
            print(row)
        print(f'gravity: {self.gravity}')
        for rotor in self.r:
            rotor.show_settings()

    def show_status(self):
        print(f'--- Time: {self.dynamics.get_time()} ---')
        print(f'Position: {self.dynamics.get_position()}')
        print(f'Velocity: {self.dynamics.get_velocity()}')
        print(f'Acceleration: {self.dynamics.get_acceleration()}')
        print(f'Quartanion: {self.dynamics.get_quartanion()}')
        print(f'Angular_velocity: {self.dynamics.get_angular_velocity()}')
        print(f'Angular_acceleration: {self.dynamics.get_angular_acceleration()}')

    def doesnt_sink_to_ground(self):
        pos = self.get_position()
        vel = self.get_velocity()
        acc = self.get_acceleration()
        if pos[2] > 0.0:
            pos[2] = 0.0
            vel[2] = 0.0
            acc[2] = 0.0
            self.set_position(pos)
            self.set_velocity(vel)

        return

    # getter

    def get_position(self):
        return self.dynamics.get_position()

    def get_velocity(self):
        return self.dynamics.get_velocity()

    def get_acceleration(self):
        return self.dynamics.get_acceleration()

    def get_quartanion(self):
        return self.dynamics.get_quartanion()

    def get_euler_angle(self):
        return self.dynamics.get_euler_angle()

    def get_angular_velocity(self):
        return self.dynamics.get_angular_velocity()

    def get_angular_acceleration(self):
        return self.dynamics.get_angular_acceleration()

    def get_force(self):
        return self.force

    def get_torque(self):
        return self.torque


    # setter
    def reset_force_and_torque(self):
        self.force[:] = 0.0
        self.torque[:] = 0.0

    def set_inputs(self, arr_inputs):
        self.ur = arr_inputs

    def set_force(self, force):
        self.force = force

    def set_torque(self, torque):
        self.torque = torque

    def set_position(self, position):
        self.dynamics.set_position(position)

    def set_velocity(self, velocity):
        self.dynamics.set_velocity(velocity)

    def set_acceleration(self, acceleration):
        self.dynamics.set_acceleration(acceleration)

    def set_quartanion(self, quartanion):
        self.dynamics.set_quartanion(quartanion)

    def set_angular_velocity(self, angular_velocity):
        self.dynamics.set_angular_velocity(angular_velocity)

    def set_angular_acceleration(self, angular_acceleration):
        self.dynamics.set_angular_acceleration(angular_acceleration)

    def set_quartanion_from(self, roll, pitch, yaw):
        self.dynamics.set_quartanion_from(roll, pitch, yaw)
