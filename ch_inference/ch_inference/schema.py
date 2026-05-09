import numpy as np
from PIL import Image

from ch_inference.rotations import rpy_from_quaternion_xyzw


IMAGE_FIELDS = {
    "left_camera": "left_image",
    "center_camera": "center_image",
    "right_camera": "right_image",
}
IMAGE_SHAPE = (256, 288, 3)
STATE_NAMES = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.roll",
    "tcp_pose.orientation.pitch",
    "tcp_pose.orientation.yaw",
    "tcp_velocity.linear.x",
    "tcp_velocity.linear.y",
    "tcp_velocity.linear.z",
    "tcp_velocity.angular.x",
    "tcp_velocity.angular.y",
    "tcp_velocity.angular.z",
    "tcp_error.x",
    "tcp_error.y",
    "tcp_error.z",
    "tcp_error.rx",
    "tcp_error.ry",
    "tcp_error.rz",
    "joint_positions.0",
    "joint_positions.1",
    "joint_positions.2",
    "joint_positions.3",
    "joint_positions.4",
    "joint_positions.5",
    "joint_positions.6",
]
ACTION_NAMES = [
    "target_tcp_pose.position.x",
    "target_tcp_pose.position.y",
    "target_tcp_pose.position.z",
    "target_tcp_pose.orientation.roll",
    "target_tcp_pose.orientation.pitch",
    "target_tcp_pose.orientation.yaw",
]


def task_prompt(task):
    return (
        f"insert {task.plug_type} plug {task.plug_name} "
        f"into {task.port_type} port {task.target_module_name}/{task.port_name}"
    )


def observation_frame(observation, image_shape=IMAGE_SHAPE):
    return {
        "observation.state": state_vector(observation),
        **{
            f"observation.images.{name}": image_from_msg(
                getattr(observation, field), image_shape
            )
            for name, field in IMAGE_FIELDS.items()
        },
    }


def state_vector(observation):
    state = observation.controller_state
    pose = state.tcp_pose
    velocity = state.tcp_velocity
    quat = (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            *rpy_from_quaternion_xyzw(quat),
            velocity.linear.x,
            velocity.linear.y,
            velocity.linear.z,
            velocity.angular.x,
            velocity.angular.y,
            velocity.angular.z,
            *state.tcp_error,
            *observation.joint_states.position[:7],
        ],
        dtype=np.float32,
    )


def image_from_msg(msg, image_shape=IMAGE_SHAPE):
    target_h, target_w, _ = image_shape
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    return np.asarray(Image.fromarray(image).resize((target_w, target_h)))
