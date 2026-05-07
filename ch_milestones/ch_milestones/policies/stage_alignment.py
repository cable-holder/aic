from ch_milestones.policies.oracle_stage_base import OracleStage


class AlignmentStage(OracleStage):
    def target_pose(self, z_offset, reset_xy_integrator=False):
        self.policy.debug_frames.publish_reference_frame(
            self.frames.orientation_ref,
            self.frames.port_ref,
            z_offset,
        )
        return self.policy.guide.alignment_gripper_pose(
            self.frames.orientation_ref,
            self.frames.port_ref,
            slerp_fraction=1.0,
            position_fraction=1.0,
            z_offset=z_offset,
            reset_xy_integrator=reset_xy_integrator,
        )
