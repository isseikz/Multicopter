"""Test pickled aircraft model."""

import os
import math
from sub_system import frame
from sub_system import math_function as mf
from sub_system import logger
import numpy as np
import matplotlib.pyplot as plt

import pickle
import datetime

def main():
    model = pickle.load(open('./model/20190506110221_target_model.bin', 'rb'))
    print(model)
    model.show_settings()

    arr_input = np.ones(4)

    for i in range(100):
        model.integrate(arr_input)
        print(model.get_position())
    print(model.get_status())

if __name__ == '__main__':
    main()
