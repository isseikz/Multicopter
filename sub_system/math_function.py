# -*- coding:utf-8 -*-
"""Mathmatical functions for aerospace dynamics calculation."""

import numpy as np
from numpy import sin, cos, arcsin, arctan2


# 回転行列
def R_roll(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def R_pitch(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])


def R_yaw(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


def getYXZRotationMatrixFrom(roll=0.0, pitch=0.0, yaw=0.0):
    """オイラー角から回転行列を作成する.

    * 回転順序がヨー→ロール→ピッチであることに注意.
    """
    return np.dot(R_pitch(pitch), np.dot(R_roll(roll), R_yaw(pitch)))

def getXYZRotationMatrixFrom(roll=0.0, pitch=0.0, yaw=0.0):
    """オイラー角から回転行列を作成する.

    * 回転順序がヨー→ピッチ→ロールであることに注意.
    """
    return np.dot(R_roll(roll), np.dot(R_pitch(pitch), R_yaw(pitch)))


def quartanion_from(roll, pitch, yaw):
    """オイラー角をクオータニオンに変換する."""
    cosR_2 = np.cos(roll/2)
    sinR_2 = np.sin(roll/2)
    cosP_2 = np.cos(pitch/2)
    sinP_2 = np.sin(pitch/2)
    cosY_2 = np.cos(yaw/2)
    sinY_2 = np.sin(yaw/2)

    q0 = cosR_2 * cosP_2 * cosY_2 + sinR_2 * sinP_2 * sinY_2
    q1 = sinR_2 * cosP_2 * cosY_2 - cosR_2 * sinP_2 * sinY_2
    q2 = cosR_2 * sinP_2 * cosY_2 + sinR_2 * cosP_2 * sinY_2
    q3 = cosR_2 * cosP_2 * sinY_2 - sinR_2 * sinP_2 * cosY_2

    return np.array([q0, q1, q2, q3])

def euler_from(quartanion):
    q0 = quartanion[0]
    q1 = quartanion[1]
    q2 = quartanion[2]
    q3 = quartanion[3]

    roll = np.arctan2(2*(q0 * q1 + q2 * q3), (1 -2 * (q1**2 + q2**2)))
    pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
    yaw = np.arctan2(2*(q0 * q3 + q1 * q2), (1 -2 * (q2**2 + q3**2)))
    return roll, pitch, yaw


def convert_vector_inertial_to_body(r, q):
    """慣性系上のベクトルを機体座標系で表す.
    input:
      r: vector, inertial axis
      q: quartanion q0, q1, q2, q3
    """
    if len(r) != 3:
        raise RuntimeError("Position vector must be three dimentional.")

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    A = np.array([
        [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
        [2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3+q0*q1)],
        [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2]
    ])
    rRotated = np.dot(A, r)

    return rRotated

def R_body_to_inertial(q):
    """Rotation matrix """
    A11 = q[0]**2+q[1]**2-q[2]**2-q[3]**2
    A12 = 2*(q[1]*q[2]-q[0]*q[3])
    A13 = 2*(q[1]*q[3]+q[0]*q[2])

    A21 = 2*(q[1]*q[2]+q[0]*q[3])
    A22 = q[0]**2-q[1]**2+q[2]**2-q[3]**2
    A23 = 2*(q[2]*q[3]-q[0]*q[1])

    A31 = 2*(q[1]*q[3]-q[0]*q[2])
    A32 = 2*(q[2]*q[3]+q[0]*q[1])
    A33 = q[0]**2-q[1]**2-q[2]**2+q[3]**2

    R = np.array([
        [A11, A12, A13],
        [A21, A22, A23],
        [A31, A32, A33]
    ])
    return R


def convert_vector_body_to_inertial(r, q):
    """位置ベクトルを回転クオータニオンに基づき回転させる.

    クオータニオンが機体の姿勢変化を表しているなら、機体座標上の位置ベクトルを慣性座標系の位置ベクトルに変換する
    """
    if len(r) != 3:
        raise RuntimeError("Inputted vector must be three dimentional.",len(r))

    R = R_body_to_inertial(q)
    rRotated = np.dot(R, r)

    return rRotated
