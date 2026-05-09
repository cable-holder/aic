import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Pose, Vector3, Wrench
from std_msgs.msg import Header

from ch_inference.rotations import quaternion_xyzw_from_rpy


def pose_from_action(action):
    qx, qy, qz, qw = quaternion_xyzw_from_rpy(action[3:6])
    pose = Pose()
    pose.position.x = float(action[0])
    pose.position.y = float(action[1])
    pose.position.z = float(action[2])
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)
    return pose


def motion_update(pose, stamp, frame_id="base_link"):
    return MotionUpdate(
        header=Header(frame_id=frame_id, stamp=stamp),
        pose=pose,
        target_stiffness=np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten(),
        target_damping=np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten(),
        feedforward_wrench_at_tip=Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        ),
        wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        trajectory_generation_mode=TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        ),
    )
