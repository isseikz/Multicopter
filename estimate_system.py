"""System estimation"""

import os
import math
from sub_system import frame
from sub_system import math_function as mf
from sub_system import logger
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

class MyGenerator(Sequence):
    """Custom generator"""

    def __init__(self, batch_size=1):
        """construction
        """
        self.batch_size = batch_size

    def __getitem__(self, idx):
        """Get batch data
        """
        return self.data, self.label

    def __len__(self):
        """Batch length"""
        return 1

    def __set_data_label__(self, x, y):
        self.data = x
        self.label = y

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

    model = frame.Multicopter()
    model.read_regular_settings()
    model.show_settings()
    with open('./model/'+file_date+'_target_model.bin', 'wb') as f:
        pickle.dump(model, f, protocol=4)
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
    ref_pos = [0.5, -1.0, -1.0]
    ref_yaw = 0.0
    ref_final_yaw = math.atan2(ref_pos[1], ref_pos[0]) * 180 / (2 * math.pi)
    arr_inputs = np.zeros(4)

    n_input = 4
    n_states = 6 # acceleration, angular_acceleration
    n_sequences = 10
    input_dim = (n_input + n_states) * n_sequences + n_input

    estimator = Sequential()
    estimator.add(Dense(units=32, activation='relu', input_dim=input_dim))
    estimator.add(Dense(units=32, activation='relu'))
    estimator.add(Dense(units=n_states, activation='linear'))
    estimator.compile(optimizer='rmsprop',loss='mse')

    myGenerator = MyGenerator()
    loss = LossHistory()

    batch_x = np.zeros((n_sequences, n_input+n_states))
    batch_y = np.zeros((n_sequences, n_states))

    dnn_input = np.zeros(n_input)
    dnn_output= np.zeros(n_states)
    row = 0

    history = []
    log_est = []
    log_ans = []
    log_est_t = []
    log_input = []

    while time < 5000: # 1 sec/ 100steps
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
        if (time % 100 == 0) | any( abs(angle) > np.pi/3 for angle in model.get_euler_angle()):
            print("--- Reset Vehicle Motion ---")
            model.reset_all()
            arr_inputs = (0.5+ np.random.rand(4) * 0.5)# * np.ones(n_input) #
            batch_x = np.zeros((n_sequences, n_input+n_states))
            batch_y = np.zeros((n_sequences, n_states))
        # --- when certain time has passed or the vihicle is in dangerous state,
        #         the vehicle will be stopped and the state will be reset ---

        # --- update input to the estimator: {x} ---
        batch_x[0:-2, :] = batch_x[1:-1, :]
        batch_x[-1, :] = np.hstack((dnn_input, dnn_output))
        dnn_input = arr_inputs
        x = np.reshape((batch_x), (n_sequences * (n_input + n_states)))
        x = np.hstack((x, dnn_input))
        x = np.reshape(x, (1,input_dim))
        # --- update input to the estimator: {x} ---

        # --- log current input and estimated response to update the estimator ---
        log_est.append(np.reshape(estimator.predict(x), 6))
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
        dnn_output = np.hstack((
            sensor_acc
            - mf.convert_vector_inertial_to_body(
                model.gravity,
                model.get_quartanion())
            ,
            model.get_angular_acceleration() ))

        batch_y[0:-2, :] = batch_y[1:-1, :]
        batch_y[-1,:] = dnn_output
        y = np.reshape(dnn_output, (1, n_states))
        log_ans.append(np.reshape(dnn_output, 6))
        # --- save the result for the input at {time} step

        # --- update estimator at every {n_sequences} step ---
        # if row == n_sequences-1:
        myGenerator.__set_data_label__(x, y)
        hist = estimator.fit_generator(myGenerator, epochs=1)
        history.append(hist.history['loss'])
        #     row = 0
        # row += 1
        # --- update estimator every {n_sequences} step ---


    # simulation loop has been finished!

    # --- save model and weights of the neural network
    print('save the architecture of a model')
    f_model = './model'
    json_string = estimator.to_json()
    open(os.path.join(f_model,file_date+'_est_model.json'), 'w').write(json_string)
    print('save weights')
    estimator.save_weights(os.path.join(f_model,file_date+'_est_model_weights.hdf5'))
    # --- save model and weights of the neural network

    # Visulize machine response
    log.visualize_data(save=True, filename=file_date)
    # Visulize machine response

    # --- Visulize the performance of the estimator ---
    log_error = np.array(log_est) -np.array(log_ans)
    log_mse_acc = np.zeros(log_error.shape[0])
    log_mse_ang = np.zeros(log_error.shape[0])
    for row in range(log_error.shape[0]):
        log_mse_acc[row] = (np.dot(log_error[row, 0:3], log_error[row, 0:3].T))
        log_mse_ang[row] = (np.dot(log_error[row, 3:], log_error[row, 3:].T))
    log_mse_acc = np.sqrt(log_mse_acc / np.max(log_mse_acc))
    log_mse_ang = np.sqrt(log_mse_ang / np.max(log_mse_ang))

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
    ax[0].set_ylabel(r'$Acceleration\ [m/s^{2}]$')

    ax[1].plot(log_est_t,np.array(log_est)[:,3], label='est_x')
    ax[1].plot(log_est_t,np.array(log_est)[:,4], label='est_y')
    ax[1].plot(log_est_t,np.array(log_est)[:,5], label='est_z')
    ax[1].plot(log_est_t,np.array(log_ans)[:,3], label='ans_x')
    ax[1].plot(log_est_t,np.array(log_ans)[:,4], label='ans_y')
    ax[1].plot(log_est_t,np.array(log_ans)[:,5], label='ans_z')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylabel(r'$Angular\ Acceleration\ [deg/s^{2}]$')

    ax[1].set_xlabel(r'$Time[s]$')
    fig0.savefig('./model/'+file_date+'_log_est.png')


    fig1, ax1 = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
    ax1[0].plot(log_est_t, log_mse_acc, label='mse acceleration')
    ax1[0].plot(log_est_t, log_mse_ang, label='mse angular_acceleration')
    ax1[0].legend()
    ax1[0].grid()
    ax1[0].set_ylabel(r'$\sqrt{\frac{Mean Squared Error}{Max(MSE)}}$')

    ax1[1].plot(log_est_t, np.array(log_input), label='input')
    ax1[1].legend()
    ax1[1].grid()
    ax1[1].set_ylabel(r'$Control input$')

    ax1[1].set_xlabel(r'$Time[s]$')
    fig1.savefig('./model/'+file_date+'_log_mse.png')

    plt.show()
    # --- Visulize the performance of the estimator ---



if __name__ == '__main__':
    main()