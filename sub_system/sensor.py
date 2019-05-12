import numpy as np
from . import frame
from . import math_function as mf

class sixDOF(object):
    """docstring for sixDOF."""

    def __init__(self):
        super(sixDOF, self).__init__()
        self.erroe_rate = 0.05

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

    def monitor_acgular_velocity(self, model):
        return model.get_angular_velocity()
