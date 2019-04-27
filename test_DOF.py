"""Test code for 'equation_of_motion.py'. """
import math
import matplotlib.pyplot as plt
from sub_system import equation_of_motion
import numpy as np

def output_state(r):
    # print(f'--- Time: {r.t} ---')
    # print(f'position: {r.get_position()}')
    # print(f'velocity: {r.get_velocity()}')
    # print(f'quartanion: {r.get_quartanion()}')
    print(f'euler angle: {r.get_euler_angle()}')
    # print(f'ang velocity: {r.get_angular_velocity()}')

def test_single(id_input):
    r = equation_of_motion.SixDOF(t0=0.0, dt=0.01)
    tf = 1.0

    # log = []

    r.set_quartanion_from()
    r.set_angular_velocity([0.0, 0.0, 0.0])
    output_state(r)

    test_item = ('forward', 'right', 'left', 'rolling', 'pitching', 'yawing')

    input = np.zeros(6, dtype=float)

    print(f"--- {test_item[id_input]} ---")
    while r.integrator.t < 2*math.pi:
        az = np.sin(r.integrator.t)
        input[id_input] = az
        r.step(input)
        output_state(r)
        print(r.get_time())

def test_dual(id_input):
    r = equation_of_motion.SixDOF(t0=0.0, dt=0.01)
    tf = 1.0

    # log = []

    r.set_quartanion_from()
    r.set_angular_velocity([0.0, 0.0, 0.0])
    output_state(r)

    test_item = ('forward', 'right', 'left', 'rolling', 'pitching', 'yawing')

    input = np.zeros(6, dtype=float)

    print(f"--- {test_item[id_input[0]]} {test_item[id_input[1]]} ---")
    while r.integrator.t < 2*math.pi:
        a1 = np.sin(r.integrator.t)
        a2 = np.cos(r.integrator.t)
        input[id_input[0]] = a1
        input[id_input[1]] = a2
        r.step(input)
        output_state(r)
        print(r.get_time())

def test_triple(id_input):
    r = equation_of_motion.SixDOF(t0=0.0, dt=0.01)
    tf = 1.0

    # log = []

    r.set_quartanion_from()
    r.set_angular_velocity([0.0, 0.0, 0.0])
    output_state(r)

    test_item = ('forward', 'right', 'left', 'rolling', 'pitching', 'yawing')

    input = np.zeros(6, dtype=float)

    print(f"--- {test_item[i] for i in id_input} ---")
    while r.integrator.t < 2*math.pi:
        a1 = np.sin(r.integrator.t)
        a2 = np.cos(r.integrator.t)
        a3 = np.sin(r.integrator.t)
        input[id_input[0]] = a1
        input[id_input[1]] = a2
        input[id_input[2]] = a3
        r.step(input)
        output_state(r)
        print(r.get_time())

def main():
    test_single(3)
    test_single(4)
    test_single(5)
    test_dual((3,4))
    test_dual((4,5))
    test_dual((3,5))
    test_triple((3,4,5))



    # plt.figure()
    # plt.plot(log)
    # plt.show()


if __name__ == '__main__':
    main()
