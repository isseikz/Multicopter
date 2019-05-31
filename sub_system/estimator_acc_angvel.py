"""System estimation"""

import os
import math
from . import frame
from . import math_function as mf
from . import logger
from . import equation_of_motion as em
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

def loss_function(y_true, y_label):
    return (y_true - y_label)**2

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


class Estimator(object):
    """docstring for Estimator."""

    def __init__(self, terminate_time=100):
        tz_jst = datetime.timezone(datetime.timedelta(hours=9))
        dt_now = datetime.datetime.now(tz=tz_jst)
        self.file_date = dt_now.strftime('%Y%m%d%H%M%S')
        self.terminate_time = terminate_time

        # --- load model ---
            #  If you want to load a pre-designed model, then uncomment.
        # with open('./model/Seminar 190513/20190516000814_target_model.bin', mode='rb', ) as f:
        #     model = pickle.load(f)
            # If you want to load a pre-designed model.

            # If you want to create a new model, then uncomment.
        self.model = frame.Multicopter()
        self.model.read_regular_settings_zhang()
            # If you want to create a new model.
        # --- load model ---

        self.model.show_settings()
        with open('./model/'+self.file_date+'_target_model.bin', 'wb') as f: # save model
            pickle.dump(self.model, f, protocol=4)

        print(f'Initial position: {self.model.get_position()}')
        print(f'Initial velocity: {self.model.get_velocity()}')
        print(f'Initial acceleration: {self.model.get_acceleration()}')
        print(f'Initial quartanion: {self.model.get_quartanion()}')
        print(f'Initial angular_velocity: {self.model.get_angular_velocity()}')
        print(f'Initial angular_acceleration: {self.model.get_angular_acceleration()}')

        self.log = logger.Logger()
        self.log_pred = []

        self.time = 0
        self.integral = 0.0
        self.ref_pos = [0.5, -1.0, -1.0]
        self.ref_yaw = 0.0
        self.ref_final_yaw = math.atan2(self.ref_pos[1], self.ref_pos[0]) * 180 / (2 * math.pi)
        self.arr_inputs = np.zeros(4)

        self.n_input = 4
        self.n_states = 6 # 3d acceleration, 3d angular_acceleration
        self.n_sequences = 10
        self.input_dim = (self.n_input + self.n_states) * self.n_sequences + self.n_input

        self.estimator = Sequential()
        self.estimator.add(Dense(units=32, activation='relu', input_dim=self.input_dim))
        self.estimator.add(Dense(units=32, activation='relu'))
        self.estimator.add(Dense(units=self.n_states, activation='linear'))
        self.estimator.compile(optimizer='rmsprop', loss='mse')

        self.myGenerator = MyGenerator()
        self.loss = LossHistory()

        self.batch_x = np.zeros((self.n_sequences, self.n_input+self.n_states))
        self.batch_y = np.zeros((self.n_sequences, self.n_states))

        self.dnn_input = np.zeros(self.n_input)
        self.dnn_output= np.zeros(self.n_states)
        self.row = 0

        # --- estimated simulator ---
        self.pred_state = em.SixDOF(0.0, 0.01)
        # --- estimated simulator ---

        self.history = []
        self.log_est = []
        self.log_ans = []
        self.log_est_t = []
        self.log_input = []
        self.log_est_s = []
        self.log_est_q = []
        self.log_ans_s = []
        self.log_ans_q = []

    def reset_motion(self):
        self.model.reset_all()
        self.pred_state.reset_state()
        self.arr_inputs = (0.5+ np.random.rand(4) * 0.5)# * np.ones(n_input) #
        self.batch_x = np.zeros((self.n_sequences, self.n_input+self.n_states))
        self.batch_y = np.zeros((self.n_sequences, self.n_states))

    def update_x(self):
        self.batch_x[0:-2, :] = self.batch_x[1:-1, :]
        self.batch_x[-1, :] = np.hstack((self.dnn_input, self.dnn_output))
        self.dnn_input = self.arr_inputs
        self.x = np.reshape((self.batch_x), (self.n_sequences * (self.n_input + self.n_states)))
        self.x = np.hstack((self.x, self.dnn_input))
        self.x = np.reshape(self.x, (1,self.input_dim))

    def predict(self):
        self.pred_accang = self.estimator.predict(self.x)[0]
        self.pred_acc = self.pred_accang[0:3]
        self.pred_ang = self.pred_accang[3:6]
        self.pred_q = self.pred_state.get_quartanion()
        self.pred_acc_and_g = mf.convert_vector_body_to_inertial(self.pred_acc, self.pred_q) + np.array([0.0,0.0,9.81])
        self.pred_state.step_angvel(self.pred_acc_and_g, self.pred_ang)
        self.log_est_s.append(self.pred_state.get_status())
        self.log_est_q.append(self.pred_state.get_euler_angle())

    def add_est_value(self):
        self.log_est.append(np.reshape(self.pred_accang, 6))
        self.log_est_t.append(self.model.dynamics.get_time())
        self.log_input.append(self.dnn_input)

    def integrate_plant(self):
        self.model.integrate(self.arr_inputs)         # 左後，左前，右後，右前

    def save_result(self):
        self.sensor_acc = self.model.get_sensor_acceleration()
        self.dnn_output = np.hstack((
            (self.sensor_acc
            - mf.convert_vector_inertial_to_body(
                self.model.gravity,
                self.model.get_quartanion())) / 9.81
            ,
            self.model.get_angular_velocity()))

        self.batch_y[0:-2, :] = self.batch_y[1:-1, :]
        self.batch_y[-1,:] = self.dnn_output
        self.y = np.reshape(self.dnn_output, (1, self.n_states))
        self.log_ans.append(np.reshape(self.dnn_output, 6))
        self.log_ans_s.append(self.model.get_position())
        self.log_ans_q.append(self.model.get_euler_angle())

    def append_data(self):
        self.myGenerator.__set_data_label__(self.x, self.y)
        self.hist = self.estimator.fit_generator(self.myGenerator, epochs=1, verbose=0)
        self.history.append(self.hist.history['loss'])

    def log_data(self):
        self.log.add_data(self.model)

    def save_estimator_model(self):
        f_model = './model/'
        self.estimator.save(f_model+self.file_date+'_estimator_model.h5')

    def vis_machine_response(self):
        self.log.visualize_data(save=True, show=False, filename=self.file_date)

    def vis_do_before_output(self):
        self.test_object = '_test_RAND'

        self.log_est = np.array(self.log_est)
        self.log_ans = np.array(self.log_ans)

        self.log_est[3:6] *= 180.0 / np.pi
        self.log_ans[3:6] *= 180.0 / np.pi

        self.log_error = self.log_est - self.log_ans
        self.log_mse_acc = np.zeros(self.log_error.shape[0])
        self.log_mse_ang = np.zeros(self.log_error.shape[0])


        for row in range(self.log_error.shape[0]):
            self.log_mse_acc[row] = (np.dot(self.log_error[row, 0:3], self.log_error[row, 0:3].T))
            self.log_mse_ang[row] = (np.dot(self.log_error[row, 3:], self.log_error[row, 3:].T))
        self.log_mse_acc = np.sqrt(self.log_mse_acc)
        self.log_mse_ang = np.sqrt(self.log_mse_ang)
        self.log_mse_ang_avg = [np.median(self.log_mse_ang[i:i+5]) for i in range(self.log_mse_ang.shape[0]-5)]
        self.log_mse_ang_avg = np.hstack((np.zeros(5), np.array(self.log_mse_ang_avg)))

        self.log_est_s = np.array(self.log_est_s)
        self.log_ans_s = np.array(self.log_ans_s)
        self.log_est_q = np.array(self.log_est_q) * 180 / np.pi
        self.log_ans_q = np.array(self.log_ans_q) * 180 / np.pi

        plt.rcParams['font.size'] = 20
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['lines.linewidth'] = 2

    def vis_estimated_value(self):
        fig0, ax = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
        ax[0].plot(self.log_est_t,np.array(self.log_est)[:,0], label='est_x')
        ax[0].plot(self.log_est_t,np.array(self.log_est)[:,1], label='est_y')
        ax[0].plot(self.log_est_t,np.array(self.log_est)[:,2], label='est_z')
        ax[0].plot(self.log_est_t,np.array(self.log_ans)[:,0], label='ans_x')
        ax[0].plot(self.log_est_t,np.array(self.log_ans)[:,1], label='ans_y')
        ax[0].plot(self.log_est_t,np.array(self.log_ans)[:,2], label='ans_z')
        ax[0].legend()
        ax[0].grid()
        ax[0].set_ylabel(r'$Acceleration\ [G]$')

        ax[1].plot(self.log_est_t,np.array(self.log_est)[:,3], label='est_x')
        ax[1].plot(self.log_est_t,np.array(self.log_est)[:,4], label='est_y')
        ax[1].plot(self.log_est_t,np.array(self.log_est)[:,5], label='est_z')
        ax[1].plot(self.log_est_t,np.array(self.log_ans)[:,3], label='ans_x')
        ax[1].plot(self.log_est_t,np.array(self.log_ans)[:,4], label='ans_y')
        ax[1].plot(self.log_est_t,np.array(self.log_ans)[:,5], label='ans_z')
        ax[1].legend()
        ax[1].grid()
        ax[1].set_ylabel(r'$Angular\ Acceleration\ [deg/s^{2}]$')

        ax[1].set_xlabel(r'$Time[s]$')
        ax_time = self.terminate_time
        while ax_time > 500:
            ax[1].set_xlim([(ax_time-500)*0.01, ax_time*0.01])
            fig0.savefig('./model/'+self.file_date+'_log_est_'+str(ax_time)+'.png')
            ax_time -= 1000
        fig0.savefig('./model/'+self.file_date+'_log_est'+self.test_object+'.png')


    def vis_mean_squared_error(self):
        fig1, ax1 = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
        ax1[0].plot(self.log_est_t, self.log_mse_acc, label=r'$Acceleration\ [G]$')
        ax1[0].plot(self.log_est_t, self.log_mse_ang, label=r'$Angular\ acceleration\ [deg/s^{2}]$')
        ax1[0].plot(self.log_est_t, self.log_mse_ang_avg, label=r'$Angular\ acceleration\ median [deg/s^{2}]$')
        ax1[0].legend()
        ax1[0].grid()
        ax1[0].set_ylabel(r'$\sqrt{Mean\ Squared\ Error}$')
        ax1[0].set_ylim([0.0, 20.0])

        ax1[1].plot(self.log_est_t, np.array(self.log_input), label='input')
        ax1[1].legend()
        ax1[1].grid()
        ax1[1].set_ylabel(r'$Control\ input$')

        ax1[1].set_xlabel(r'$Time[s]$')
        fig1.savefig('./model/'+self.file_date+'_log_mse'+self.test_object+'.png')

    def vis_estimator_pos_att(self):
        fig2, ax2 = plt.subplots(nrows=2, sharex=True, figsize=(16,12))
        ax2[0].plot(self.log_est_t, self.log_est_s[:,0], label='est x')
        ax2[0].plot(self.log_est_t, self.log_est_s[:,1], label='est y')
        ax2[0].plot(self.log_est_t, self.log_est_s[:,2], label='est z')
        ax2[0].plot(self.log_est_t, self.log_ans_s[:,0], label='ans x')
        ax2[0].plot(self.log_est_t, self.log_ans_s[:,1], label='ans y')
        ax2[0].plot(self.log_est_t, self.log_ans_s[:,2], label='ans z')
        ax2[0].set_ylabel(r'$Position\ [m]$')
        ax2[0].legend()
        ax2[0].grid()
        ax2[1].plot(self.log_est_t, self.log_est_q[:,0], label='est roll')
        ax2[1].plot(self.log_est_t, self.log_est_q[:,1], label='est pitch')
        ax2[1].plot(self.log_est_t, self.log_est_q[:,2], label='est yaw')
        ax2[1].plot(self.log_est_t, self.log_ans_q[:,0], label='ans roll')
        ax2[1].plot(self.log_est_t, self.log_ans_q[:,1], label='ans pitch')
        ax2[1].plot(self.log_est_t, self.log_ans_q[:,2], label='ans yaw')
        ax2[1].set_xlabel(r'$Time\ [s]$')
        ax2[1].set_ylabel(r'$Euler\ Angles\ [deg]$')
        ax2[1].legend()
        ax2[1].grid()
        time = self.terminate_time
        while time > 500:
            ax2[1].set_xlim([(time-500)*0.01, time*0.01])
            ax2[0].set_ylim([-1, 1])
            fig2.savefig('./model/'+self.file_date+'_comp_sim_ans_'+str(time)+self.test_object+'.png')
            time -= 1000
        ax2[1].set_xlim([0.0, 5.0])
        ax2[0].set_ylim([-1, 1])
        fig2.savefig('./model/'+self.file_date+'_comp_sim_ans_'+str(time)+self.test_object+'.png')



def main():
    aae = Estimator()

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
