from math import ceil, sqrt

from ch_milestones.policies.cartesian_trajectory import (
    interpolate_pose,
    pose_from_transform,
    pose_trajectory,
)


class OracleMotionCommander:
    def __init__(self, policy):
        self.policy = policy

    def command_pose(self, pose):
        current = self.policy.guide.transform("base_link", "gripper/tcp")
        self.policy.get_logger().info(
            f"stage={self.policy.stage} "
            f"current_tcp_p={self.fmt_xyz(current.translation)} "
            f"current_tcp_q={self.fmt_quat(current.rotation)} "
            f"target_tcp_p={self.fmt_xyz(pose.position)} "
            f"target_tcp_q={self.fmt_quat(pose.orientation)}"
        )
        self.policy.set_pose_target(
            self.policy.move_robot,
            pose,
            stiffness=self.policy.param("oracle_cartesian_stiffness"),
            damping=self.policy.param("oracle_cartesian_damping"),
        )
        self.policy.sleep_for(self.policy.command_period())

    def execute_pose_trajectory(self, goal, steps):
        start = pose_from_transform(
            self.policy.guide.transform("base_link", "gripper/tcp")
        )
        self.policy.get_logger().info(
            f"stage={self.policy.stage} planned_cartesian_trajectory steps={steps} "
            f"start_tcp_p={self.fmt_xyz(start.position)} "
            f"goal_tcp_p={self.fmt_xyz(goal.position)}"
        )
        for pose in pose_trajectory(start, goal, steps):
            self.command_pose(pose)

    def execute_live_step_trajectory(self, goal_fn, steps, max_translation_step):
        if max_translation_step <= 0.0:
            raise ValueError("max_translation_step must be positive")

        self.policy.get_logger().info(
            f"stage={self.policy.stage} live_step_trajectory steps={steps} "
            f"max_translation_step={max_translation_step:.4f}"
        )
        for _ in range(steps):
            if self.command_step_toward(goal_fn(), max_translation_step):
                return True
        return False

    def command_step_toward(self, goal, max_translation_step):
        current = pose_from_transform(
            self.policy.guide.transform("base_link", "gripper/tcp")
        )
        distance = self.translation_distance(current, goal)
        fraction = 1.0
        if distance > max_translation_step:
            fraction = max_translation_step / distance
        self.command_pose(interpolate_pose(current, goal, fraction))
        return distance <= max_translation_step

    def hold_pose(self, pose, duration):
        for _ in range(ceil(duration / self.policy.command_period())):
            self.command_pose(pose)

    def hold_live_pose(self, pose_fn, duration):
        for _ in range(ceil(duration / self.policy.command_period())):
            self.command_pose(pose_fn())

    def fmt_xyz(self, xyz):
        return f"({xyz.x:.4f},{xyz.y:.4f},{xyz.z:.4f})"

    def fmt_quat(self, quat):
        return f"({quat.w:.4f},{quat.x:.4f},{quat.y:.4f},{quat.z:.4f})"

    def translation_distance(self, start, goal):
        dx = goal.position.x - start.position.x
        dy = goal.position.y - start.position.y
        dz = goal.position.z - start.position.z
        return sqrt(dx * dx + dy * dy + dz * dz)
