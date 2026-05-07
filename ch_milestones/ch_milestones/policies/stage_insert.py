from math import ceil

from ch_milestones.policies.cartesian_trajectory import (
    pose_from_transform,
    pose_trajectory,
)
from ch_milestones.policies.stage_alignment import AlignmentStage


class InsertStage(AlignmentStage):
    stage = "insert"

    def run(self):
        self.begin()
        z_offset = self.param("oracle_alignment_fine_align_z_offset")
        end_z_offset = self.param("oracle_insert_end_z_offset")
        segment_m = self.param("oracle_insert_step_meters")
        command_step_m = self.param("oracle_alignment_insert_step_meters")
        command_period = self.param("oracle_alignment_command_period")
        start = pose_from_transform(
            self.policy.guide.transform("base_link", "gripper/tcp")
        )

        while z_offset > end_z_offset:
            if self.policy.insertion_completed():
                return
            next_z_offset = max(end_z_offset, z_offset - segment_m)
            steps = ceil((z_offset - next_z_offset) / command_step_m)
            goal = self.target_pose(next_z_offset)

            for pose in pose_trajectory(start, goal, steps):
                if self.policy.insertion_completed():
                    return
                self.command_pose(pose, command_period)

            start = goal
            z_offset = next_z_offset
