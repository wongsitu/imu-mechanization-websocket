'''
This code is adapted by Adam Smith from the madgwick algorithm provided in the AHRS 
python library available at https://github.com/Mayitzin/ahrs for improved computation speed.
'''

'''
Madgwick Orientation Filter
===========================

This is an orientation filter applicable to IMUs consisting of tri-axial
gyroscopes and accelerometers, and MARG arrays, which also include tri-axial
magnetometers, proposed by Sebastian Madgwick [Madgwick]_.

The filter employs a quaternion representation of orientation to describe the
nature of orientations in three-dimensions and is not subject to the
singularities associated with an Euler angle representation, allowing
accelerometer and magnetometer data to be used in an analytically derived and
optimised gradient-descent algorithm to compute the direction of the gyroscope
measurement error as a quaternion derivative.

Innovative aspects of this filter include:

- A single adjustable parameter defined by observable systems characteristics.
- An analytically derived and optimised gradient-descent algorithm enabling
  performance at low sampling rates.
- On-line magnetic distortion compensation algorithm.
- Gyroscope bias drift compensation.

Rewritten in Python from the `original implementation <https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/>`_
conceived by Sebastian Madgwick.

References
----------
.. [Madgwick] Sebastian Madgwick. An efficient orientation filter for inertial
    and inertial/magnetic sensor arrays. April 30, 2010.
    http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/
'''

import numpy as np
from math import sqrt


def updateIMUFast(q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Quaternion Estimation with IMU architecture.

    Parameters
    ----------
    q : numpy.ndarray
        A-priori quaternion.
    gyr : numpy.ndarray
        Sample of tri-axial Gyroscope in rad/s
    acc : numpy.ndarray
        Sample of tri-axial Accelerometer in m/s^2

    Returns
    -------
    q : numpy.ndarray
        Estimated quaternion.
    """

    if gyr is None:
        return q
    qDot = 0.5 * q_prod(q, [0, *gyr])  # (eq. 12)

    a = acc / norm3(acc)
    qw, qx, qy, qz = q / norm4(q)
    qw2 = 2 * qw
    qx2 = 2 * qx
    qy2 = 2 * qy
    qz2 = 2 * qz

    # Gradient objective function (eq. 25) and Jacobian (eq. 26)
    f = np.array(
        [
            qx2 * qz - qw2 * qy - a[0],
            qw2 * qx + qy2 * qz - a[1],
            2.0 * (0.5 - qx * qx - qy * qy) - a[2],
        ]
    )  # (eq. 25)
    J = np.array(
        [
            [-qy2, qz2, -qw2, qx2],
            [qx2, qw2, qz2, qy2],
            [0.0, -4.0 * qx, -4.0 * qy, 0.0],
        ]
    )  # (eq. 26)
    # Objective Function Gradient
    gradient = J.T @ f  # (eq. 34)
    gradient /= norm4(gradient)
    qDot -= 0.033 * gradient  # (eq. 33)  # gain = 0.033

    q += qDot * dt  # (eq. 13)
    q /= norm4(q)
    return q


def updateMARGFast(q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Quaternion Estimation with a MARG architecture.

    Parameters
    ----------
    q : numpy.ndarray
        A-priori quaternion.
    gyr : numpy.ndarray
        Sample of tri-axial Gyroscope in rad/s
    acc : numpy.ndarray
        Sample of tri-axial Accelerometer in m/s^2
    mag : numpy.ndarray
        Sample of tri-axial Magnetometer in nT

    Returns
    -------
    q : numpy.ndarray
        Estimated quaternion.
    """

    if gyr is None:
        return q
    if mag is None:
        return updateIMUFast(q, gyr, acc, dt=dt)
    qDot = 0.5 * q_prod(q, [0, *gyr])  # (eq. 12)

    a = acc / norm3(acc)
    m = mag / norm3(mag)
    # Rotate normalized magnetometer measurements
    h = q_prod(q, q_prod([0, *m], q_conj(q)))  # (eq. 45)
    bx = sqrt(h[1] * h[1] + h[2] * h[2])  # (eq. 46)
    bz = h[3]
    qw, qx, qy, qz = q / norm4(q)

    # Gradient objective function (eq. 31) and Jacobian (eq. 32)
    qxqz = qx * qz
    qyqz = qy * qz
    qwqx = qw * qx
    qwqy = qw * qy

    qxsqpqysq = qx * qx + qy * qy

    bx2 = 2.0 * bx
    bz2 = 2.0 * bz

    qx2 = 2.0 * qx
    qy2 = 2.0 * qy
    qz2 = 2.0 * qz
    qw2 = 2.0 * qw

    qx4 = 2.0 * qx2
    qy4 = 2.0 * qy2

    bx2qx = bx2 * qx
    bx2qy = bx2 * qy
    bx2qz = bx2 * qz
    bx2qw = bx2 * qw

    bz2qx = bz2 * qx
    bz2qy = bz2 * qy
    bz2qz = bz2 * qz
    bz2qw = bz2 * qw

    qwqxpqyqz = qwqx + qyqz
    qxqzmqwqy = qxqz - qwqy

    f = np.array(
        [
            2.0 * qxqzmqwqy - a[0],
            2.0 * qwqxpqyqz - a[1],
            2.0 * (0.5 - qxsqpqysq) - a[2],
            bx - bx2qy * qy - bx2qz * qz + bz2 * qxqzmqwqy - m[0],
            bx2qx * qy - bx2qw * qz + bz2 * qwqxpqyqz - m[1],
            bx2 * (qwqy + qxqz) + bz2 * (0.5 - qxsqpqysq) - m[2],
        ]
    )  # (eq. 31)
    J = np.array(
        [
            [-qy2, qz2, -qw2, qx2],
            [qx2, qw2, qz2, qy2],
            [0.0, -qx4, -qy4, 0.0],
            [-bz2qy, bz2qz, -bx * qy4 - bz2qw, -2.0 * bx2qz + bz2qx],
            [-bx2qz + bz2qx, bx2qy + bz2qw, bx2qx + bz2qz, -bx2qw + bz2qy],
            [bx2qy, bx2qz - bz * qx4, bx2qw - bz * qy4, bx2qx],
        ]
    )  # (eq. 32)
    gradient = J.T @ f  # (eq. 34)
    gradient /= norm4(gradient)
    qDot -= 0.041 * gradient  # (eq. 33)  # gain = 0.041

    q += qDot * dt  # (eq. 13)
    return q / norm4(q)


def norm3(x):
    return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])


def norm4(x):
    return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])


def q_conj(q: np.ndarray) -> np.ndarray:
    """
    Conjugate of unit quaternion

    Parameters
    ----------
    q : numpy.ndarray
        Unit quaternion or 2D array of Quaternions.

    Returns
    -------
    q_conj : numpy.ndarray
        Conjugated quaternion or 2D array of conjugated Quaternions.
    """
    return np.array([q[0], -q[1], -q[2], -q[2]])


def q_prod(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Product of two unit quaternions.

    Parameters
    ----------
    p : numpy.ndarray
        First quaternion to multiply
    q : numpy.ndarray
        Second quaternion to multiply

    Returns
    -------
    pq : numpy.ndarray
        Product of both quaternions
    """
    pq = np.zeros(4)
    pq[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]
    pq[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2]
    pq[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
    pq[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
    return pq
