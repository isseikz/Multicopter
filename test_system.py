"""Test code for the test. """
import math
from sub_system import frame
from sub_system import math_function as mf
from sub_system import logger
import numpy as np

def main():
    model = frame.Multicopter()
    # model.read_regular_settings()
    model.read_regular_settings_omari()
    model.show_settings()
    print(f'Initial position: {model.get_position()}')
    print(f'Initial velocity: {model.get_velocity()}')
    print(f'Initial acceleration: {model.get_acceleration()}')
    print(f'Initial quartanion: {model.get_quartanion()}')
    print(f'Initial angular_velocity: {model.get_angular_velocity()}')
    print(f'Initial angular_acceleration: {model.get_angular_acceleration()}')

    log = logger.Logger()

    time = 0
    integral = 0.0
    ref_pos = [1.0, -1.0, -1.0]
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

        k_p = 5.0
        k_d = 0.99
        integral += (-1.0-altitude)
        inp_alt = -0.5 * (-1.0-altitude) -0.001 * integral - 0.5*(-vel_alt)
        inp_rol = k_p * ( ref_roll - attitude_deg[0])/180 - k_p * k_d * ang_vel[0]/(2*math.pi)
        inp_pit = k_p * ( ref_pitch - attitude_deg[1])/180 - k_p * k_d * ang_vel[1]/(2*math.pi)
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
        # logging datas
        # print(model.get_status())
        log.add_data(model)
        time += 1

    # Visulize datas
    log.visualize_data()

if __name__ == '__main__':
    main()
