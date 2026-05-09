import numpy as np


def rpy_from_quaternion_xyzw(quat):
    q = np.asarray(quat, dtype=np.float64)
    x, y, z, w = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1).astype(np.float32)


def quaternion_xyzw_from_rpy(rpy):
    r, p, y = np.moveaxis(np.asarray(rpy, dtype=np.float64), -1, 0) * 0.5
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    ).astype(np.float32)
