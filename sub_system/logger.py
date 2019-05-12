"""Logger"""

import numpy as np
import matplotlib.pyplot as plt
import math

class Logger(object):
    """docstring for Logger."""

    def __init__(self):
        super(Logger, self).__init__()
        self.log_tim = []
        self.log_pos = []
        self.log_vel = []
        self.log_acc = []
        self.log_ang = []
        self.log_avl = []
        self.log_aac = []
        self.log_rof = []
        self.log_rom = []
        self.log_fm  = []

    def add_data(self, model):
        self.log_tim.append(model.dynamics.get_time())
        self.log_pos.append(model.get_position())
        self.log_vel.append(model.get_velocity())
        self.log_acc.append(model.get_acceleration())
        self.log_ang.append([angle * 180 / math.pi for angle in model.get_euler_angle()]) #self.log_ang.append(model.get_quartanion())#
        self.log_avl.append(model.get_angular_velocity())
        self.log_aac.append(model.get_angular_acceleration())
        self.log_rof.append([rotor.get_total_force() for rotor in model.r])
        self.log_rom.append([rotor.get_total_torque() for rotor in model.r])
        self.log_fm.append(np.hstack((model.get_force(), model.get_torque())))

    def visualize_data(self, save=False, filename=''):
        # Visulize datas
        fig, ax = plt.subplots(3,3, sharex='col', figsize=(12,9))
        lineobj = ax[0,0].plot(self.log_tim, self.log_pos)
        ax[0,0].legend(iter(lineobj), ['x','y','z'])
        ax[0,0].set_title('CoG position [m]')
        ax[0,0].grid()

        lineobj = ax[1,0].plot(self.log_tim, self.log_vel)
        ax[1,0].legend(iter(lineobj), ['dxdt','dydt','dzdt'])
        ax[1,0].set_title('CoG velocity [m/s]')
        ax[1,0].grid()

        lineobj = ax[2,0].plot(self.log_tim, self.log_acc)
        ax[2,0].legend(iter(lineobj), ['d2xdt2','dy2dt2','dz2dt2'])
        ax[2,0].set_title('CoG acceleration [m/s2]')
        ax[2,0].set_xlabel('time [s]')
        ax[2,0].grid()

        lineobj = ax[0,1].plot(self.log_tim, self.log_ang)
        ax[0,1].legend(iter(lineobj), ['roll','pitch','yaw'])
        ax[0,1].set_title('Attitude angle [deg]')
        ax[0,1].grid()

        lineobj = ax[1,1].plot(self.log_tim, self.log_avl)
        ax[1,1].legend(iter(lineobj), ['p','q','r'])
        ax[1,1].set_title('Anguler velocity [deg/s]')
        ax[1,1].grid()

        lineobj = ax[2,1].plot(self.log_tim, self.log_aac)
        ax[2,1].legend(iter(lineobj), ['dpdt','dqdt','drdt'])
        ax[2,1].set_title('Angular acceleration [deg/s2]')
        ax[2,1].set_xlabel('time [s]')
        ax[2,1].grid()

        lineobj = ax[0,2].plot(self.log_tim, [[log[2] for log in log_rotor] for log_rotor in self.log_rof])
        ax[0,2].legend(iter(lineobj), ['rotor1','rotor2','rotor3','rotor4'])
        ax[0,2].set_title('propeller force [N]')
        ax[0,2].grid()

        lineobj = ax[1,2].plot(self.log_tim, [[log[2] for log in log_rotor] for log_rotor in self.log_rom])
        ax[1,2].legend(iter(lineobj), ['rotor1','rotor2','rotor3','rotor4'])
        ax[1,2].set_title('propeller torque [NM]')
        ax[1,2].grid()

        lineobj = ax[2,2].plot(self.log_tim, self.log_fm)
        ax[2,2].legend(iter(lineobj), ['fx','fy','fz','mx','my','mz'])
        ax[2,2].set_title('CoG Force / Moment')
        ax[2,2].set_xlabel('time [s]')
        ax[2,2].grid()

        plt.savefig('./model/'+filename+'_log.png')

        plt.show()
