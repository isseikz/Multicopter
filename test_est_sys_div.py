"""System estimation"""

import os
import math
from sub_system import frame
from sub_system import math_function as mf
from sub_system import logger
from sub_system import equation_of_motion as em
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.callbacks import Callback
import json

import pickle
import datetime

class GainTuner(object):
    """docstring for GainTuner."""

    def __init__(self, n_gains):
        self.n_gains = n_gains
        self.gains = np.ones(self.n_gains)
        self.reward = 0

    def get_gains(self):
        return self.gains

    def set_gains(self, gains):
        self.gains = np.array(gains)


def input_range(x):
    if x > 1.0:
        return 1.0
    elif x < 0.0:
        return 0.0
    else:
        return x


def main():
    file_date = '20190527121417'

    # --- load model ---
        #  If you want to load a pre-designed model, then uncomment.
    # with open('./model/'+file_date+'_target_model.bin', mode='rb') as f:
    #     model = pickle.load(f)
    model = frame.Multicopter()
    model.read_regular_settings_zhang()

    # --- load model ---

    model.show_settings()

    print(f'Initial position: {model.get_position()}')
    print(f'Initial velocity: {model.get_velocity()}')
    print(f'Initial acceleration: {model.get_acceleration()}')
    print(f'Initial quartanion: {model.get_quartanion()}')
    print(f'Initial angular_velocity: {model.get_angular_velocity()}')
    print(f'Initial angular_acceleration: {model.get_angular_acceleration()}')

    log = logger.Logger()
    log_pred = []

    time = 0
    integral = 0.0
    ref_pos = [1.0, -1.0, -1.0]
    ref_yaw = 0.0
    ref_final_yaw = math.atan2(ref_pos[1], ref_pos[0]) * 180 / (2 * math.pi)
    arr_inputs = np.zeros(4)

    n_input = 4 + 1 # control inputs and altitude
    n_states = 3 # 3d acceleration or 3d angular_acceleration
    n_sequences = 2
    input_dim = (n_input + n_states) * n_sequences + n_input

    # Load pre-designed estimator
    estimator_acc = load_model('./model/'+file_date+'_estimator_model_acc.h5')
    estimator_ang = load_model('./model/'+file_date+'_estimator_model_ang.h5')
    # load pre-designed estimator

    batch_x_acc = np.zeros((n_sequences, n_input+n_states))
    batch_x_ang = np.zeros((n_sequences, n_input+n_states))

    dnn_input= np.zeros(n_states)
    dnn_output_acc = np.zeros(n_input)
    dnn_output_ang = np.zeros(n_input)
    row = 0

    # --- estimated simulator ---
    pred_state = em.SixDOF(0.0, 0.01)
    # --- estimated simulator ---

    # --- Updated Params (PD gain) ---
    k_p = 1.0
    k_d = 0.99
    k_alt_p = 0.5
    k_alt_d = 0.99
    gain_tuner = GainTuner(2)
    gain_tuner.set_gains((k_p, k_d))
    # --- Updated Params (PD gain) ---

    history = []
    log_est = []
    log_ans = []
    log_est_t = []
    log_input = []
    log_est_s = []
    log_est_q = []
    log_ans_s = []
    log_ans_q = []
    log_ref = []

    while time <1001: # 1 sec/ 100steps
        if time % 200 == 0:
            print(model.dynamics.get_time())
            # model.show_status()
            # print(model.get_acceleration())
            #
            # for rotor in model.r:
            #     print(rotor.get_total_force())

        # logging datas
        log.add_data(model)

        # --- when certain time has passed or the vihicle is in dangerous state,
        #         the vehicle will be stopped and the state will be reset ---
        if any( abs(angle) > np.pi/3 for angle in model.get_euler_angle()):
            print("--- Reset Vehicle Motion ---")
            model.reset_all()
            pred_state.reset_state()
            # arr_inputs = (0.5+ np.random.rand(4) * 0.5)# * np.ones(n_input) #
            batch_x_acc = np.zeros((n_sequences, n_input+n_states))
            batch_x_ang = np.zeros((n_sequences, n_input+n_states))
        # --- when certain time has passed or the vihicle is in dangerous state,
        #         the vehicle will be stopped and the state will be reset ---

        # Calc input with PD method
        # Change nominal positoin
        # if time % 1000 == 0:
        #     ref_pos[0] = np.random.rand()*2 -1.0
        #     ref_pos[1] = np.random.rand()*2 -1.0
        #     ref_pos[2] = np.random.rand()*-1.0
        #     ref_final_yaw = math.atan2(ref_pos[1], ref_pos[0]) * 180 / (2 * math.pi)

        # Guidance: State & Nominal Position -> Nominal attitude

        position = model.get_position()
        velocity = model.get_velocity()
        quartanion = model.get_quartanion()
        diff_position_body = mf.convert_vector_inertial_to_body(
            ref_pos - position,
            quartanion
        )
        diff_velocity_body = mf.convert_vector_inertial_to_body(
            velocity,
            quartanion
        )
        ref_roll = 5.0 * (diff_position_body[1]) + 5.0 * 0.99 * (0.0 - diff_velocity_body[1])
        ref_pitch = 5.0 * -(diff_position_body[0]) + 5.0 * 0.99 * -(0.0 -diff_velocity_body[0])
        ref_yaw = ref_final_yaw

        # Control: State & Nominal attitude -> Nominal input(s)
        altitude = model.get_position()[2]
        vel_alt = model.get_velocity()[2]
        attitude = model.get_euler_angle()
        attitude_deg = np.array(attitude) * 180 / math.pi
        ang_vel = model.get_angular_velocity()

        [k_p, k_alt_p] = gain_tuner.get_gains()
        integral += (-1.0-altitude)
        inp_alt = -k_alt_p * (-1.0-altitude) - k_alt_p * k_alt_d *(-vel_alt) #-0.001 * integral
        inp_rol = k_p * ( ref_roll - attitude_deg[0])/180 - k_p * k_d * ang_vel[0]/(2*math.pi)
        inp_pit = k_p * ( ref_pitch - attitude_deg[1])/180 - k_p * k_d * ang_vel[1]/(2*math.pi)
        inp_yaw = 1.0 * ( ref_yaw - attitude_deg[2])/180 - 1.0 * 0.99 * ang_vel[2]/(2*math.pi)

        new_inputs = np.array([
            inp_alt + inp_rol - inp_pit - inp_yaw,
            inp_alt + inp_rol + inp_pit + inp_yaw,
            inp_alt - inp_rol - inp_pit + inp_yaw,
            inp_alt - inp_rol + inp_pit - inp_yaw
        ]) # 左後，左前，右後，右前
        new_inputs = np.frompyfunc(input_range, 1, 1)(new_inputs)
        arr_inputs = new_inputs
        # arr_inputs = arr_inputs * 0.5 + new_inputs * 0.5
        # print(arr_inputs, new_inputs)
        # Calc input with PD

        # --- update input to the estimator: {x} ---
        altitude = pred_state.get_position()[2]
        batch_x_acc[0:-1, :] = batch_x_acc[1:, :]
        batch_x_acc[-1, :] = np.hstack((dnn_input, dnn_output_acc))
        batch_x_ang[0:-1, :] = batch_x_ang[1:, :]
        batch_x_ang[-1, :] = np.hstack((dnn_input, dnn_output_ang))
        dnn_input = np.hstack((arr_inputs, altitude))
        x_acc = np.reshape((batch_x_acc), (n_sequences * (n_input + n_states)))
        x_acc = np.hstack((x_acc, dnn_input))
        x_acc = np.reshape(x_acc, (1,input_dim))
        x_ang = np.reshape((batch_x_ang), (n_sequences * (n_input + n_states)))
        x_ang = np.hstack((x_ang, dnn_input))
        x_ang = np.reshape(x_ang, (1,input_dim))
        # --- update input to the estimator: {x} ---

        pred_acc = estimator_acc.predict(x_acc)[0]
        pred_ang = estimator_ang.predict(x_ang)[0]
        pred_q = pred_state.get_quartanion()
        pred_acc_and_g = mf.convert_vector_body_to_inertial(pred_acc, pred_q) + np.array([0.0,0.0,9.81])
        pred_state.step_angvel(pred_acc_and_g, pred_ang)
        log_est_s.append(pred_state.get_status())
        log_est_q.append(pred_state.get_euler_angle())

        # --- log current input and estimated response to update the estimator ---
        log_est.append(np.hstack((pred_acc, pred_ang)))
        log_est_t.append(model.dynamics.get_time())
        log_input.append(dnn_input)
        # --- log current input and estimated response to update the estimator ---

        # --- Plant ---
        #   You should NOT modify below in the loop
        #   if you are not familier with the system
        model.integrate(arr_inputs) # 左後，左前，右後，右前
        time += 1
        # --- Plant ---

        # --- save the result for the input at {time} step
        sensor_acc = model.get_sensor_acceleration()
        dnn_output_acc = (sensor_acc
            - mf.convert_vector_inertial_to_body(
                model.gravity,
                model.get_quartanion())) / 9.81
        dnn_output_ang = model.get_angular_velocity()

        log_ans.append(np.hstack((dnn_output_acc, dnn_output_ang)))
        log_ans_s.append(model.get_position())
        log_ans_q.append(model.get_euler_angle())
        log_ref.append(ref_pos)
        # --- save the result for the input at {time} step

    # simulation loop has been finished!


    # Visulize machine response
    log.visualize_data(save=False, show=False, filename=file_date)
    # Visulize machine response

    # --- Visulize the performance of the estimator ---
    test_object = '_test_pid'

    log_error = np.array(log_est) -np.array(log_ans)
    log_mse_acc = np.zeros(log_error.shape[0])
    log_mse_ang = np.zeros(log_error.shape[0])
    for row in range(log_error.shape[0]):
        log_mse_acc[row] = (np.dot(log_error[row, 0:3], log_error[row, 0:3].T))
        log_mse_ang[row] = (np.dot(log_error[row, 3:], log_error[row, 3:].T))
    log_mse_acc = np.sqrt(log_mse_acc)
    log_mse_ang = np.sqrt(log_mse_ang)

    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['lines.linewidth'] = 2

    log_est = np.array(log_est)
    log_ans = np.array(log_ans)

    log_est[3:6] *= 180.0 / np.pi
    log_ans[3:6] *= 180.0 / np.pi

    fig0, ax = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
    ax[0].plot(log_est_t,np.array(log_est)[:,0], label='est_x')
    ax[0].plot(log_est_t,np.array(log_est)[:,1], label='est_y')
    ax[0].plot(log_est_t,np.array(log_est)[:,2], label='est_z')
    ax[0].plot(log_est_t,np.array(log_ans)[:,0], label='ans_x')
    ax[0].plot(log_est_t,np.array(log_ans)[:,1], label='ans_y')
    ax[0].plot(log_est_t,np.array(log_ans)[:,2], label='ans_z')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_ylim([-2, 2])
    ax[0].set_ylabel(r'$Acceleration\ [G]$')

    ax[1].plot(log_est_t,np.array(log_est)[:,3], label='est_x')
    ax[1].plot(log_est_t,np.array(log_est)[:,4], label='est_y')
    ax[1].plot(log_est_t,np.array(log_est)[:,5], label='est_z')
    ax[1].plot(log_est_t,np.array(log_ans)[:,3], label='ans_x')
    ax[1].plot(log_est_t,np.array(log_ans)[:,4], label='ans_y')
    ax[1].plot(log_est_t,np.array(log_ans)[:,5], label='ans_z')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylim([-15, 15])
    ax[1].set_ylabel(r'$Angular\ Velocity\ [deg/s]$')

    ax[1].set_xlabel(r'$Time[s]$')
    ax_time = time
    while ax_time > 500:
        ax[1].set_xlim([(ax_time-500)*0.01, ax_time*0.01])
        fig0.savefig('./model/'+file_date+'_log_est_'+str(ax_time)+'.png')
        ax_time -= 1000
    fig0.savefig('./model/'+file_date+'_log_est'+test_object+'.png')


    fig1, ax1 = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
    ax1[0].plot(log_est_t, log_mse_acc, label='mse acceleration [m]')
    ax1[0].plot(log_est_t, log_mse_ang, label='mse angular velocity [rad/s]')
    ax1[0].legend()
    ax1[0].grid()
    ax1[0].set_ylabel(r'$\sqrt{Mean\ Squared\ Error}$')

    ax1[1].plot(log_est_t, np.array(log_input), label='input')
    ax1[1].legend()
    ax1[1].grid()
    ax1[1].set_ylabel(r'$Control\ input$')

    ax1[1].set_xlabel(r'$Time[s]$')
    fig1.savefig('./model/'+file_date+'_log_mse'+test_object+'.png')

    # compare simulation from estimated value and answer
    log_est_s = np.array(log_est_s)
    log_ans_s = np.array(log_ans_s)
    log_est_q = np.array(log_est_q) * 180 / np.pi
    log_ans_q = np.array(log_ans_q) * 180 / np.pi
    log_ref = np.array(log_ref)
    fig2, ax2 = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
    ax2[0].plot(log_est_t, log_est_s[:,0], label='est x')
    ax2[0].plot(log_est_t, log_est_s[:,1], label='est y')
    ax2[0].plot(log_est_t, log_est_s[:,2], label='est z')
    ax2[0].plot(log_est_t, log_ans_s[:,0], label='ans x')
    ax2[0].plot(log_est_t, log_ans_s[:,1], label='ans y')
    ax2[0].plot(log_est_t, log_ans_s[:,2], label='ans z')
    ax2[0].plot(log_est_t, log_ref[:,0], label='ref x')
    ax2[0].plot(log_est_t, log_ref[:,1], label='ref y')
    ax2[0].plot(log_est_t, log_ref[:,2], label='ref z')
    ax2[0].set_ylabel(r'$Position\ [m]$')
    ax2[0].legend()
    ax2[0].grid()
    ax2[1].plot(log_est_t, log_est_q[:,0], label='est roll')
    ax2[1].plot(log_est_t, log_est_q[:,1], label='est pitch')
    ax2[1].plot(log_est_t, log_est_q[:,2], label='est yaw')
    ax2[1].plot(log_est_t, log_ans_q[:,0], label='ans roll')
    ax2[1].plot(log_est_t, log_ans_q[:,1], label='ans pitch')
    ax2[1].plot(log_est_t, log_ans_q[:,2], label='ans yaw')
    ax2[1].set_xlabel(r'$Time\ [s]$')
    ax2[1].set_ylabel(r'$Euler\ Angles\ [deg]$')
    ax2[1].legend()
    ax2[1].grid()
    while time > 500:
        ax2[1].set_xlim([(time-500)*0.01, time*0.01])
        ax2[0].set_ylim([-3, 3])
        fig2.savefig('./model/'+file_date+'_comp_sim_ans_'+str(time)+test_object+'.png')
        time -= 1000
    ax2[1].set_xlim([0.0, 5.0])
    ax2[0].set_ylim([-3, 3])
    fig2.savefig('./model/'+file_date+'_comp_sim_ans_'+str(time)+test_object+'.png')

    plt.show()
    # --- Visulize the performance of the estimator ---



if __name__ == '__main__':
    main()
