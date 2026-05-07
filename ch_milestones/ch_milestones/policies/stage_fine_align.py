from ch_milestones.policies.cartesian_trajectory import (
    minimum_jerk_pose_trajectory,
    pose_from_transform,
)
from ch_milestones.policies.stage_alignment import AlignmentStage


class FineAlignStage(AlignmentStage):
    stage = "fine_align"

    def run(self):
        self.begin()
        z_offset = self.param("oracle_alignment_fine_align_z_offset")
        steps = int(self.param("oracle_alignment_fine_align_steps"))
        command_period = self.param("oracle_alignment_command_period")
        start = pose_from_transform(
            self.policy.guide.transform("base_link", "gripper/tcp")
        )
        goal = self.target_pose(z_offset, reset_xy_integrator=True)

        for pose in minimum_jerk_pose_trajectory(start, goal, steps):
            self.command_pose(pose, command_period)

        for _ in range(
            self.scaled_steps("oracle_fine_align_hold_steps", allow_zero=True)
        ):
            pose = self.target_pose(z_offset)
            self.command_pose(pose, command_period)
