# Multicopter Simulation Tools for System Design
====

## Overview
+ sub_system/equation_of_motion - 6 DOF rigid body equation of motion
+ sub_system/frame - Multicopter frame model
+ sub_system/math_function - self-made mathmatical functions used here
+ sub_system/rotor - rotor model
+ sub_system/sensor_acceleration.py - accelometer model

## Demo
Simulation sample:
+ ref_pos = [0.5, -1.0, -1.0]
+ ref_yaw =  math.atan2(ref_pos[1], ref_pos[0]) * 180 / (2 * math.pi)
+ Guidance: Propotional Derivative Control (PD)
+ Control: Propotional Derivative Control (PD)
![result](/datas/test_system_output.png)

## Requirement
+ NumPy
+ Matplotlib
+ SciPy

## Usage
Sample code is written in [test code](/test_system.py)
`your console$ python test_system.py`

NOTE: z-axis is opposite to altitude.
ex. (altitude) = 10 <=> z = -10

## Install
`git clone`


## Licence
MIT
