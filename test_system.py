"""Test code for the test. """
import math
import matplotlib.pyplot as plt
from sub_system import frame
from sub_system import math_function as mf
import numpy as np

def main():
    model = frame.Multicopter()
    model.read_regular_settings()
    model.show_settings()
    print(f'Initial position: {model.get_position()}')
    print(f'Initial velocity: {model.get_velocity()}')
    print(f'Initial acceleration: {model.get_acceleration()}')
    print(f'Initial quartanion: {model.get_quartanion()}')
    print(f'Initial angular_velocity: {model.get_angular_velocity()}')
    print(f'Initial angular_acceleration: {model.get_angular_acceleration()}')

    log_tim = []
    log_pos = []
    log_vel = []
    log_acc = []
    log_ang = []
    log_avl = []
    log_aac = []
    log_rof = []
    log_rom = []
    log_fm  = []

    time = 0
    integral = 0.0
    ref_pos = [0.5, -1.0, -1.0]
    ref_yaw = 0.0
    ref_final_yaw = math.atan2(ref_pos[1], ref_pos[0]) * 180 / (2 * math.pi)
    while time < 3000:
        if time % 200 == 0:
            print(model.dynamics.get_time())
            # model.show_status()
            # print(model.get_acceleration())
            #
            # for rotor in model.r:
            #     print(rotor.get_total_force())

        # logging datas
        log_tim.append(model.dynamics.get_time())
        log_pos.append(model.get_position())
        log_vel.append(model.get_velocity())
        log_acc.append(model.get_acceleration())
        log_ang.append([angle * 180 / math.pi for angle in model.get_euler_angle()]) #log_ang.append(model.get_quartanion())#
        log_avl.append(model.get_angular_velocity())
        log_aac.append(model.get_angular_acceleration())
        log_rof.append([rotor.get_total_force() for rotor in model.r])
        log_rom.append([rotor.get_total_torque() for rotor in model.r])
        log_fm.append(np.hstack((model.get_force(), model.get_torque())))

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

        integral += (-1.0-altitude)
        inp_alt = -0.5 * (-1.0-altitude) -0.001 * integral - 0.5*(-vel_alt)
        inp_rol = 4.0 * ( ref_roll - attitude_deg[0])/180 - 4 *0.99 * ang_vel[0]/(2*math.pi)
        inp_pit = 4.0 * ( ref_pitch - attitude_deg[1])/180 - 4 *0.99 * ang_vel[1]/(2*math.pi)
        inp_yaw = 1.0 * ( ref_yaw - attitude_deg[2])/180 - 1.0 * 0.99 * ang_vel[2]/(2*math.pi)

        arr_inputs = [
            inp_alt + inp_rol - inp_pit - inp_yaw,
            inp_alt + inp_rol + inp_pit + inp_yaw,
            inp_alt - inp_rol - inp_pit + inp_yaw,
            inp_alt - inp_rol + inp_pit - inp_yaw
        ] # 左後，左前，右後，右前

        # Plant:
        #   You should NOT modify below in the loop
        #   if you are not familier with the system
        model.integrate(arr_inputs) # 左後，左前，右後，右前
        time += 1

    # Visulize datas
    fig, ax = plt.subplots(3,3, sharex='col')
    lineobj = ax[0,0].plot(log_tim, log_pos)
    ax[0,0].legend(iter(lineobj), ['x','y','z'])
    ax[0,0].set_title('CoG position [m]')
    ax[0,0].grid()

    lineobj = ax[1,0].plot(log_tim, log_vel)
    ax[1,0].legend(iter(lineobj), ['dxdt','dydt','dzdt'])
    ax[1,0].set_title('CoG velocity [m/s]')
    ax[1,0].grid()

    lineobj = ax[2,0].plot(log_tim, log_acc)
    ax[2,0].legend(iter(lineobj), ['d2xdt2','dy2dt2','dz2dt2'])
    ax[2,0].set_title('CoG acceleration [m/s2]')
    ax[2,0].set_xlabel('time [s]')
    ax[2,0].grid()

    lineobj = ax[0,1].plot(log_tim, log_ang)
    ax[0,1].legend(iter(lineobj), ['roll','pitch','yaw'])
    ax[0,1].set_title('Attitude angle [deg]')
    ax[0,1].grid()

    lineobj = ax[1,1].plot(log_tim, log_avl)
    ax[1,1].legend(iter(lineobj), ['p','q','r'])
    ax[1,1].set_title('Anguler velocity [deg/s]')
    ax[1,1].grid()

    lineobj = ax[2,1].plot(log_tim, log_aac)
    ax[2,1].legend(iter(lineobj), ['dpdt','dqdt','drdt'])
    ax[2,1].set_title('Angular acceleration [deg/s2]')
    ax[2,1].set_xlabel('time [s]')
    ax[2,1].grid()

    lineobj = ax[0,2].plot(log_tim, [[log[2] for log in log_rotor] for log_rotor in log_rof])
    ax[0,2].legend(iter(lineobj), ['rotor1','rotor2','rotor3','rotor4'])
    ax[0,2].set_title('propeller force [N]')
    ax[0,2].grid()

    lineobj = ax[1,2].plot(log_tim, [[log[2] for log in log_rotor] for log_rotor in log_rom])
    ax[1,2].legend(iter(lineobj), ['rotor1','rotor2','rotor3','rotor4'])
    ax[1,2].set_title('propeller torque [NM]')
    ax[1,2].grid()

    lineobj = ax[2,2].plot(log_tim, log_fm)
    ax[2,2].legend(iter(lineobj), ['fx','fy','fz','mx','my','mz'])
    ax[2,2].set_title('CoG Force / Moment')
    ax[2,2].set_xlabel('time [s]')
    ax[2,2].grid()

    plt.show()

if __name__ == '__main__':
    main()
