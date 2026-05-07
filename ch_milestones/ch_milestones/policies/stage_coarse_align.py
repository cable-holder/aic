from math import ceil

from geometry_msgs.msg import Point, Pose, Quaternion
import numpy as np
from transforms3d._gohlketransforms import quaternion_slerp
from transforms3d.quaternions import quat2mat

from ch_milestones.policies.cartesian_trajectory import (
    minimum_jerk_fractions,
    pose_from_transform,
)
from ch_milestones.policies.oracle_stage_base import OracleStage


class CoarseAlignStage(OracleStage):
    stage = "coarse_align"

    def run(self):
        self.begin()
        step_meters = (
            self.param("oracle_coarse_align_step_meters") * self.policy.speed_scale()
        )
        current_tcp, current_plug = self.motion.current_tcp_and_plug()
        start = pose_from_transform(current_tcp)
        goal_quat = self.orientation_goal(current_tcp, current_plug)
        goal_orientation = Quaternion(
            w=float(goal_quat[0]),
            x=float(goal_quat[1]),
            y=float(goal_quat[2]),
            z=float(goal_quat[3]),
        )
        rotation = self.motion.rotation_distance(start.orientation, goal_orientation)
        radius = self.motion.transform_distance(current_tcp, current_plug)
        rotation_step = (
            self.param("oracle_coarse_align_rotation_step_radians")
            * self.policy.speed_scale()
        )
        steps = max(
            1,
            2 * ceil(radius * rotation / step_meters),
            2 * ceil(rotation / rotation_step),
        )

        self.policy.guide.reset_integrator()
        for pose in self.pivot_trajectory(
            current_tcp,
            current_plug,
            goal_quat,
            steps,
        ):
            self.command_pose(pose, target_plug=current_plug)

    def orientation_goal(self, current_tcp, current_plug):
        quat = self.policy.guide.target_quat(
            self.frames.orientation_ref,
            current_plug,
            current_tcp,
        )
        return self.policy.guide.same_hemisphere(
            quat,
            (
                current_tcp.rotation.w,
                current_tcp.rotation.x,
                current_tcp.rotation.y,
                current_tcp.rotation.z,
            ),
        )

    def pivot_trajectory(self, current_tcp, current_plug, goal_quat, steps):
        start_quat = (
            current_tcp.rotation.w,
            current_tcp.rotation.x,
            current_tcp.rotation.y,
            current_tcp.rotation.z,
        )
        tcp_xyz = self.xyz(current_tcp)
        plug_xyz = self.xyz(current_plug)
        tcp_to_plug = quat2mat(start_quat).T @ (plug_xyz - tcp_xyz)

        for fraction in minimum_jerk_fractions(steps):
            quat = quaternion_slerp(start_quat, goal_quat, fraction)
            xyz = plug_xyz - quat2mat(quat) @ tcp_to_plug
            yield Pose(
                position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                orientation=Quaternion(
                    w=float(quat[0]),
                    x=float(quat[1]),
                    y=float(quat[2]),
                    z=float(quat[3]),
                ),
            )

    def xyz(self, transform):
        return np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=float,
        )
