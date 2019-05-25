"""System estimation"""

import os
import math
from sub_system import frame
from sub_system import math_function as mf
from sub_system import logger
from sub_system import equation_of_motion as em
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.callbacks import Callback
from keras import regularizers

import pickle
import datetime

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def loss_function(y_true, y_label):
    return (y_true - y_label)**2

class MyGenerator(Sequence):
    """Custom generator"""

    def __init__(self, batch_size=1, input_dim=104, output_dim=6):
        """construction
        """
        self.data = []
        self.label = []
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __getitem__(self, idx):
        """Get batch data
        """
        x = np.reshape(np.array(self.data), (self.batch_size, self.input_dim))
        y = np.reshape(np.array(self.label), (self.batch_size, self.output_dim))
        return x, y

    def __len__(self):
        """Batch length"""
        self.batch_size = len(self.data)
        return self.batch_size

    def __set_data_label__(self, x, y):
        self.data.append(x)
        self.label.append(y)

    def on_epoch_end(self):
        """Task when end of epoch"""
        pass

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def main():
    tz_jst = datetime.timezone(datetime.timedelta(hours=9))
    dt_now = datetime.datetime.now(tz=tz_jst)
    file_date = dt_now.strftime('%Y%m%d%H%M%S')
    terminate_time = 20000

    # --- load model ---
        #  If you want to load a pre-designed model, then uncomment.
    # with open('./model/Seminar 190513/20190516000814_target_model.bin', mode='rb', ) as f:
    #     model = pickle.load(f)
        # If you want to load a pre-designed model.

        # If you want to create a new model, then uncomment.
    model = frame.Multicopter()
    model.read_regular_settings_zhang()
    model.ground_cond = True
        # If you want to create a new model.
    # --- load model ---

    model.show_settings()
    # with open('./model/'+file_date+'_target_model.bin', 'wb') as f: # save model
    #     pickle.dump(model, f, protocol=4)

    print(f'Initial position: {model.get_position()}')
    print(f'Initial velocity: {model.get_velocity()}')
    print(f'Initial acceleration: {model.get_acceleration()}')
    print(f'Initial quartanion: {model.get_quartanion()}')
    print(f'Initial angular_velocity: {model.get_angular_velocity()}')
    print(f'Initial angular_acceleration: {model.get_angular_acceleration()}')

    log = logger.Logger()
    log_pred = []

    time = 0
    step_time = 0
    integral = 0.0
    ref_pos = [0.5, -1.0, -1.0]
    ref_yaw = 0.0
    ref_final_yaw = math.atan2(ref_pos[1], ref_pos[0]) * 180 / (2 * math.pi)
    arr_inputs = np.zeros(4)

    n_input = 5 # 4 inputs and altitude
    n_states = 3 # 3d acceleration, 3d angular_acceleration
    n_sequences = 2
    input_dim = (n_input + n_states) * n_sequences + n_input
    output_dim = n_states

    estimator_acc = Sequential()
    estimator_acc.add(Dense(units=10, kernel_regularizer=regularizers.l1(0.0001), activation='relu', input_dim=input_dim))
    estimator_acc.add(Dense(units=10, kernel_regularizer=regularizers.l1(0.0001), activation='relu'))
    estimator_acc.add(Dense(units=n_states, activation='linear'))
    estimator_acc.compile(optimizer=Adam(lr=0.0001), loss='mse')

    myGenerator_acc = MyGenerator(input_dim=input_dim, output_dim=output_dim)
    loss = LossHistory()

    estimator_ang = Sequential()
    estimator_ang.add(Dense(units=10, kernel_regularizer=regularizers.l1(0.0001), activation='relu', input_dim=input_dim))
    estimator_ang.add(Dense(units=10, kernel_regularizer=regularizers.l1(0.0001), activation='relu'))
    estimator_ang.add(Dense(units=n_states, activation='linear'))
    estimator_ang.compile(optimizer=Adam(lr=0.0001), loss='mse')

    myGenerator_ang = MyGenerator(input_dim=input_dim, output_dim=output_dim)
    loss = LossHistory()

    batch_x_acc = np.zeros((n_sequences, n_input+n_states))
    batch_y_acc = np.zeros((n_sequences, n_states))
    batch_x_ang = np.zeros((n_sequences, n_input+n_states))
    batch_y_ang = np.zeros((n_sequences, n_states))

    dnn_input = np.zeros(n_input)
    dnn_output_acc = np.zeros(n_states)
    dnn_output_ang = np.zeros(n_states)

    row = 0

    # --- estimated simulator ---
    pred_state = em.SixDOF(0.0, 0.01)
    # --- estimated simulator ---

    history = []
    log_est = []
    log_ans = []
    log_est_t = []
    log_input = []
    log_est_s = []
    log_est_q = []
    log_ans_s = []
    log_ans_q = []

    while time <terminate_time + 1: # 1 sec/ 100steps
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
        if (time % 200 == 0) | any( abs(angle) > np.pi/3 for angle in model.get_euler_angle()[:2]):
            if (time > 0) & (myGenerator_acc.__len__() > 0):
                hist = estimator_acc.fit_generator(myGenerator_acc, epochs=1, verbose=1)
                hist = estimator_ang.fit_generator(myGenerator_ang, epochs=1, verbose=1)
                history.append(hist.history['loss'])
                myGenerator_acc.data = []
                myGenerator_acc.label = []
                myGenerator_ang.data = []
                myGenerator_ang.label = []

            print("--- Reset Vehicle Motion ---")
            model.reset_all()
            pred_state.reset_state()
            arr_inputs = 0.5+ np.random.rand(4) * 0.5 # * np.ones(n_input) #
            batch_x_acc = np.zeros((n_sequences, n_input+n_states))
            # batch_y_acc = np.zeros((n_sequences, n_states))
            batch_x_ang = np.zeros((n_sequences, n_input+n_states))
            # batch_y_ang = np.zeros((n_sequences, n_states))
            step_time = 0
        # --- when certain time has passed or the vihicle is in dangerous state,
        #         the vehicle will be stopped and the state will be reset ---

        # --- update input to the estimator: {x} ---
        altitude = model.get_position()[2]
        batch_x_acc[0:-1, :] = batch_x_acc[1:, :]
        batch_x_acc[-1, :] = np.hstack((dnn_input, dnn_output_acc))
        batch_x_ang[0:-1, :] = batch_x_ang[1:, :]
        batch_x_ang[-1, :] = np.hstack((dnn_input, dnn_output_ang))
        # print(batch_x)
        dnn_input = np.hstack((arr_inputs, np.tanh(altitude)))
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
        pred_acc_and_g = mf.convert_vector_body_to_inertial(pred_acc, pred_q) * 9.81 + np.array([0.0,0.0,9.81])
        # pred_state.step(np.hstack((pred_acc_and_g, pred_ang)))
        pred_state.step_angvel(pred_acc_and_g, pred_ang)
        if step_time > 0:
            log_est_s.append(pred_state.get_status())
            log_est_q.append(pred_state.get_euler_angle())

        # --- log current input and estimated response to update the estimator ---
        if step_time > 0:
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
            + mf.convert_vector_inertial_to_body(
                model.gravity,
                model.get_quartanion())) / 9.81
        dnn_output_ang = model.get_sensor_angular_velocity()

        # batch_y_acc[0:-1, :] = batch_y_acc[1:, :]
        # batch_y_acc[-1,:] = dnn_output_acc
        # batch_y_ang[0:-1, :] = batch_y_ang[1:, :]
        # batch_y_ang[-1,:] = dnn_output_ang
        y_acc = np.reshape(dnn_output_acc, (1, n_states))
        y_ang = np.reshape(dnn_output_ang, (1, n_states))
        if step_time > 0:
            log_ans.append(np.hstack((dnn_output_acc,dnn_output_ang)))
            log_ans_s.append(model.get_position())
            log_ans_q.append(model.get_euler_angle())
        # --- save the result for the input at {time} step

        # If you want to check the acceleration estimation, then comment in.
        e = pred_acc - dnn_output_acc
        print(f'[{pred_acc[0]:.3f}, {pred_acc[1]:.3f}, {pred_acc[2]:.3f}]')
        print(f'[{dnn_output_acc[0]:.3f}, {dnn_output_acc[1]:.3f}, {dnn_output_acc[2]:.3f}]')
        print(f'{np.dot(e, e.T):.3f}')
        # If you want to check the acceleration estimation, then comment in.


        # Don't update estimator during latter half -> for test
        if (time < terminate_time * 9/10 ):# // 2:
            # --- update estimator at every {n_sequences} step ---
            # if row == n_sequences-1:
            myGenerator_acc.__set_data_label__(x_acc, y_acc)
            myGenerator_ang.__set_data_label__(x_ang, y_ang)
            #     row = 0
            # row += 1
            # --- update estimator every {n_sequences} step ---
        # Don't update estimator during latter half

        step_time += 1


    # simulation loop has been finished!

    # --- save model and weights of the neural network
    print('save the architecture of a model')
    f_model = './model/'
    estimator_acc.save(f_model+file_date+'_estimator_model_acc.h5')
    estimator_ang.save(f_model+file_date+'_estimator_model_ang.h5')
    # --- save model and weights of the neural network


    # Output the weights of an estimator
    print("--- weights of acc estimator ---")
    layers = estimator_acc.layers
    for i, layer in enumerate(layers):
        print(f'{i}: {layer}')
        weights = layer.get_weights()[0]
        influence = []
        for weight in weights:
            print(weight)
            influence.append(np.dot(weight, weight.T))
        print(influence)
        print(layer.get_weights()[1])

    print("--- weights of ang estimator ---")
    layers = estimator_ang.layers
    for i, layer in enumerate(layers):
        print(f'{i}: {layer}')
        weights = layer.get_weights()[0]
        influence = []
        for weight in weights:
            print(weight)
            influence.append(np.dot(weight, weight.T))
        print(influence)
        print(layer.get_weights()[1])
    # Output the weights of an estimator


    # Visulize machine response
    # log.visualize_data(save=False, show=False, filename=file_date)
    # Visulize machine response

    # --- Visulize the performance of the estimator ---
    test_object = '_est_'

    log_est = np.array(log_est)
    log_ans = np.array(log_ans)

    log_est[3:6] *= 180.0 / np.pi
    log_ans[3:6] *= 180.0 / np.pi

    log_error = log_est - log_ans
    log_mse_acc = np.zeros(log_error.shape[0])
    log_mse_ang = np.zeros(log_error.shape[0])
    for row in range(log_error.shape[0]):
        log_mse_acc[row] = (np.dot(log_error[row, 0:3], log_error[row, 0:3].T))
        log_mse_ang[row] = (np.dot(log_error[row, 3:], log_error[row, 3:].T))
    log_mse_acc = np.sqrt(log_mse_acc)
    log_mse_ang = np.sqrt(log_mse_ang)
    log_mse_ang_avg = [np.median(log_mse_ang[i:i+5]) for i in range(log_mse_ang.shape[0]-5)]
    log_mse_ang_avg = np.hstack((np.zeros(5), np.array(log_mse_ang_avg)))

    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['lines.linewidth'] = 2

    fig0, ax = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
    ax[0].plot(log_est_t,np.array(log_est)[:,0], label='est_x')
    ax[0].plot(log_est_t,np.array(log_est)[:,1], label='est_y')
    ax[0].plot(log_est_t,np.array(log_est)[:,2], label='est_z')
    ax[0].plot(log_est_t,np.array(log_ans)[:,0], label='ans_x')
    ax[0].plot(log_est_t,np.array(log_ans)[:,1], label='ans_y')
    ax[0].plot(log_est_t,np.array(log_ans)[:,2], label='ans_z')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_ylim([-5.0, 5.0])
    ax[0].set_ylabel(r'$Acceleration\ [G]$')

    ax[1].plot(log_est_t,np.array(log_est)[:,3], label='est_x')
    ax[1].plot(log_est_t,np.array(log_est)[:,4], label='est_y')
    ax[1].plot(log_est_t,np.array(log_est)[:,5], label='est_z')
    ax[1].plot(log_est_t,np.array(log_ans)[:,3], label='ans_x')
    ax[1].plot(log_est_t,np.array(log_ans)[:,4], label='ans_y')
    ax[1].plot(log_est_t,np.array(log_ans)[:,5], label='ans_z')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylim([-180, 180])
    ax[1].set_ylabel(r'$Angular\ Acceleration\ [deg/s^{2}]$')

    ax[1].set_xlabel(r'$Time[s]$')
    ax_time = time
    while ax_time > 500:
        ax[1].set_xlim([(ax_time-500)*0.01, ax_time*0.01])
        fig0.savefig('./model/'+file_date+test_object+str(ax_time)+'.png')
        ax_time -= 1000
    fig0.savefig('./model/'+file_date+test_object+'.png')
    ax[1].set_xlim([0.0, log_est_t[-1]])


    fig1, ax1 = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
    ax1[0].plot(log_est_t, log_mse_acc, label=r'$mse\ acceleration$')
    ax1[0].plot(log_est_t, log_mse_ang, label=r'$mse angular\ velocity$')
    ax1[0].legend()
    ax1[0].grid()
    ax1[0].set_ylabel(r'$\sqrt{Mean\ Squared\ Error}$')

    ax1[1].plot(log_est_t, np.array(log_input), label='input')
    ax1[1].legend()
    ax1[1].grid()
    ax1[1].set_ylabel(r'$Control\ input$')

    ax1[1].set_xlabel(r'$Time[s]$')
    fig1.savefig('./model/'+file_date+test_object+'mse.png')
    ax1[1].set_xlim([0.0, log_est_t[-1]])

    # compare simulation from estimated value and answer
    log_est_s = np.array(log_est_s)
    log_ans_s = np.array(log_ans_s)
    log_est_q = np.array(log_est_q) * 180 / np.pi
    log_ans_q = np.array(log_ans_q) * 180 / np.pi
    fig2, ax2 = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
    ax2[0].plot(log_est_t, log_est_s[:,0], label='est x')
    ax2[0].plot(log_est_t, log_est_s[:,1], label='est y')
    ax2[0].plot(log_est_t, log_est_s[:,2], label='est z')
    ax2[0].plot(log_est_t, log_ans_s[:,0], label='ans x')
    ax2[0].plot(log_est_t, log_ans_s[:,1], label='ans y')
    ax2[0].plot(log_est_t, log_ans_s[:,2], label='ans z')
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
        ax2[0].set_ylim([-1, 1])
        fig2.savefig('./model/'+file_date+test_object+'comp_sim_ans_'+str(time)+'.png')
        time -= 1000
    ax2[1].set_xlim([0.0, 5.0])
    ax2[0].set_ylim([-1, 1])
    fig2.savefig('./model/'+file_date+test_object+'comp_sim_ans_'+str(time)+'.png')
    ax2[1].set_xlim([0.0, log_est_t[-1]])

    plt.show()
    # --- Visulize the performance of the estimator ---



if __name__ == '__main__':
    main()
