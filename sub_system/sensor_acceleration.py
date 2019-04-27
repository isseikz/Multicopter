"""sub-system for sensor"""

from numpy.random import normal

class WithGaussianError(object):
    """docstring for Sensor."""

    def __init__(self):
        super(WithGaussianError, self).__init__()
        self.err_var = 1.0
        self.err_std = 0.05

    def get_value(self, inertial_position, gravity):
        vals = inertial_position - gravity
        val = [val * normal(self.err_var, self.err_std) for val in vals]
