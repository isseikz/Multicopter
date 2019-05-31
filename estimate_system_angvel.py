"""System estimation"""

import os
import math
from sub_system import frame
from sub_system import math_function as mf
from sub_system import logger
from sub_system import equation_of_motion as em
from sub_system import estimator_acc_angvel
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.callbacks import Callback

import pickle
import datetime

def main():
    aae = estimator_acc_angvel.Estimator()

    time = 0
    while time < (aae.terminate_time + 1): # 1 sec/ 100steps
        if time % 200 == 0:
            print(aae.model.dynamics.get_time())
            # model.show_status()
            # print(model.get_acceleration())
            #
            # for rotor in model.r:
            #     print(rotor.get_total_force())

        # logging datas
        aae.log_data()

        # --- when certain time has passed or the vihicle is in dangerous state,
        #         the vehicle will be stopped and the state will be reset ---
        if (time % 100 == 0) | any( abs(angle) > np.pi/3 for angle in aae.model.get_euler_angle()):
            print("--- Reset Vehicle Motion ---")
            aae.reset_motion()
        # --- when certain time has passed or the vihicle is in dangerous state,
        #         the vehicle will be stopped and the state will be reset ---

        # --- update input to the estimator: {x} ---
        aae.update_x()
        # --- update input to the estimator: {x} ---

        aae.predict()


        # --- log current input and estimated response to update the estimator ---
        aae.add_est_value()
        # --- log current input and estimated response to update the estimator ---

        # --- Plant ---
        #   You should NOT modify below in the loop
        #   if you are not familier with the system
        aae.integrate_plant()
        time += 1
        # --- Plant ---

        # --- save the result for the input at {time} step
        aae.save_result()
        # --- save the result for the input at {time} step

        # Don't update estimator during latter half -> for test
        if time < aae.terminate_time // 2:
            # --- update estimator at every {n_sequences} step ---
            # if row == n_sequences-1:
            aae.append_data()
            #     row = 0
            # row += 1
            # --- update estimator every {n_sequences} step ---
        # Don't update estimator during latter half


    # simulation loop has been finished!

    # --- save model and weights of the neural network
    print('save the architecture of a model')
    aae.save_estimator_model()
    # --- save model and weights of the neural network

    # Visulize machine response
    aae.vis_machine_response()
    # Visulize machine response

    # --- Visulize the performance of the estimator ---
    aae.vis_do_before_output()

    aae.vis_estimated_value()


    aae.vis_mean_squared_error()

    # compare simulation from estimated value and answer
    aae.vis_estimator_pos_att()

    plt.show()
    # --- Visulize the performance of the estimator ---


if __name__ == '__main__':
    main()
