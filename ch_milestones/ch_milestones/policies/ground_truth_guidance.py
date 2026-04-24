import numpy as np

from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from transforms3d.quaternions import quat2mat


class GroundTruthGuide:
    def __init__(self, node, task):
        self.node = node
        self.task = task
        self.tip_error_i = np.zeros(2)

    def wait_for(self, target_frame: str, source_frame: str, timeout_sec=10.0):
        start = self.node.get_clock().now()
        timeout = Duration(seconds=timeout_sec)
        while (self.node.get_clock().now() - start) < timeout:
            if self.node._tf_buffer.can_transform(target_frame, source_frame, Time()):
                return
            self.node.get_clock().sleep_for(Duration(seconds=0.1))
        raise TimeoutError(f"Missing transform {source_frame} -> {target_frame}")

    def transform(self, target_frame: str, source_frame: str) -> Transform:
        return self.node._tf_buffer.lookup_transform(
            target_frame, source_frame, Time()
        ).transform

    def gripper_pose(
        self,
        orientation_ref: Transform,
        position_ref: Transform,
        stage: str,
        slerp_fraction=1.0,
        position_fraction=1.0,
        z_offset=0.1,
        reset_xy_integrator=False,
        xy_integral_gain=0.15,
        xy_integrator_limit=0.05,
    ) -> Pose:
        plug = self.transform(
            "base_link", f"{self.task.cable_name}/{self.task.plug_name}_link"
        )
        gripper = self.transform("base_link", "gripper/tcp")

        q_gripper = self._quat(gripper)
        q_start = q_gripper
        q_target = self.target_quat(orientation_ref, plug, gripper)
        q_target = self.same_hemisphere(q_target, q_start)
        q_blend = quaternion_slerp(q_start, q_target, slerp_fraction)
        r_gripper = quat2mat(q_gripper)
        r_blend = quat2mat(q_blend)

        desired_plug = self.desired_plug_position(
            orientation_ref, position_ref, z_offset
        )
        plug_xyz = np.array(
            [plug.translation.x, plug.translation.y, plug.translation.z]
        )
        grip_xyz = np.array(
            [gripper.translation.x, gripper.translation.y, gripper.translation.z]
        )
        tcp_to_plug = r_gripper.T @ (plug_xyz - grip_xyz)
        tip_error = desired_plug[:2] - plug_xyz[:2]
        if reset_xy_integrator:
            self.reset_integrator()
        elif xy_integral_gain:
            self.tip_error_i = np.clip(
                self.tip_error_i + tip_error,
                -xy_integrator_limit,
                xy_integrator_limit,
            )

        desired_plug = desired_plug.copy()
        desired_plug[:2] += xy_integral_gain * self.tip_error_i

        target = desired_plug - r_blend @ tcp_to_plug
        start_xyz = grip_xyz
        xyz = position_fraction * target + (1.0 - position_fraction) * start_xyz
        self.node.get_logger().info(
            f"stage={stage} z={z_offset:.4f} "
            f"xy_error={tip_error[0]:.4f},{tip_error[1]:.4f}"
        )

        return Pose(
            position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
            orientation=Quaternion(
                w=float(q_blend[0]),
                x=float(q_blend[1]),
                y=float(q_blend[2]),
                z=float(q_blend[3]),
            ),
        )

    def reset_integrator(self):
        self.tip_error_i = np.zeros(2)

    def desired_plug_position(
        self,
        orientation_ref: Transform,
        position_ref: Transform,
        z_offset,
    ):
        del orientation_ref
        return self._xyz(position_ref) + np.array([0.0, 0.0, z_offset])

    def plug_xy_error(
        self,
        orientation_ref: Transform,
        position_ref: Transform,
        z_offset,
    ):
        plug = self.transform(
            "base_link", f"{self.task.cable_name}/{self.task.plug_name}_link"
        )
        desired_plug = self.desired_plug_position(
            orientation_ref, position_ref, z_offset
        )
        return desired_plug[:2] - self._xyz(plug)[:2]

    def target_quat(
        self,
        orientation_ref: Transform,
        plug: Transform,
        gripper: Transform,
    ):
        q_port = self._quat(orientation_ref)
        q_plug = self._quat(plug)
        q_gripper = self._quat(gripper)
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        return quaternion_multiply(
            quaternion_multiply(q_port, q_plug_inv),
            q_gripper,
        )

    def same_hemisphere(self, quat, reference):
        quat = np.array(quat, dtype=float)
        reference = np.array(reference, dtype=float)
        if np.dot(quat, reference) < 0.0:
            quat = -quat
        return tuple(quat)

    def _quat(self, transform: Transform):
        q = transform.rotation
        return (q.w, q.x, q.y, q.z)

    def _xyz(self, transform: Transform):
        t = transform.translation
        return np.array([t.x, t.y, t.z])

    def approach_offset(self, orientation_ref, position_ref, source_frame):
        del orientation_ref
        source = self.transform("base_link", source_frame)
        return float(source.translation.z - position_ref.translation.z)
