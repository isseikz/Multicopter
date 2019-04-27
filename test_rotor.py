"""Test code for 'rotor.py'. """
import math
import matplotlib.pyplot as plt
from sub_system import rotor

def main():
    r = rotor.Dynamics()
    tf = 5.0
    u_ref = 1.0

    log = []

    print('Time, Input, Speed')
    while r.integrator.successful() and r.integrator.t < tf:
        u_ref =math.sin( r.get_t() * 1.0 )
        r.set_input(u_ref)
        r.integrate()
        print(r.get_t(),', ', r.get_input(),',',r.get_speed(), r.get_total_force(), r.get_total_torque())
        log.append([r.get_t(), r.get_input()*r.max_speed, r.get_speed()])

    plt.figure()
    plt.plot(log)
    plt.show()


if __name__ == '__main__':
    main()
