import numpy as np
from . import frame
from . import math_function as mf

class sixDOF(object):
    """docstring for sixDOF."""

    def __init__(self):
        super(sixDOF, self).__init__()
        self.erroe_rate = 0.05

    def monitor_raw_acceleration(self, model):
        return mf.convert_vector_inertial_to_body(
            model.get_acceleration(),
            model.get_quartanion()) + mf.convert_vector_inertial_to_body(
                model.gravity,
                model.get_quartanion())

    def monitor_raw_angular_velocity(self, model):
        return model.get_angular_velocity()


    def monitor_acceleration(self, model):
        if model.get_position()[2] >= 0.0:
            return mf.convert_vector_inertial_to_body(
                model.gravity,
                model.get_quartanion())
        else:
            return mf.convert_vector_inertial_to_body(
                model.get_acceleration(),
                model.get_quartanion()) + mf.convert_vector_inertial_to_body(
                    model.gravity,
                    model.get_quartanion())

    def monitor_angular_velocity(self, model):
        if model.get_position()[2] >= 0.0:
            return np.zeros(3)
        else:
            return model.get_angular_velocity()
