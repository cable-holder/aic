from math import ceil


class OracleStageSet:
    def __init__(self, policy):
        self.approach = ApproachStage(policy)
        self.coarse_align = CoarseAlignStage(policy)
        self.fine_align = FineAlignStage(policy)
        self.insert = InsertStage(policy)


class OracleStage:
    stage = None

    def __init__(self, policy):
        self.policy = policy

    @property
    def frames(self):
        return self.policy.frames

    @property
    def motion(self):
        return self.policy.motion

    def begin(self):
        self.policy.set_stage(self.stage)

    def param(self, name):
        return self.policy.param(name)

    def scaled_steps(self, name, allow_zero=False):
        steps = self.param(name)
        if allow_zero and steps == 0:
            return 0
        return max(1, ceil(steps / self.policy.speed_scale()))

    def alignment_gain(self):
        return self.param("oracle_alignment_integral_gain")

    def integrator_limit(self):
        return self.param("oracle_alignment_integrator_limit")

    def progress(self, step, steps):
        return step / steps


class ApproachStage(OracleStage):
    stage = "approach"

    def run(self):
        self.begin()
        steps = self.scaled_steps("oracle_approach_steps")
        z_offset = self.param("oracle_approach_z_offset")
        step_meters = self.param("oracle_approach_step_meters") * self.policy.speed_scale()

        self.policy.guide.reset_integrator()
        reached = self.motion.execute_live_step_trajectory(
            lambda: self.live_approach_pose(z_offset),
            steps,
            step_meters,
        )
        if not reached:
            self.policy.get_logger().warning(
                "Approach did not reach the live staged target before max steps"
            )

    def live_approach_pose(self, z_offset):
        return self.policy.gripper_pose(
            slerp_fraction=0.0,
            position_fraction=1.0,
            position_ref=self.frames.offset_ref(z_offset),
            z_offset=0.0,
            reset_xy_integrator=True,
            xy_integral_gain=0.0,
        )


class CoarseAlignStage(OracleStage):
    stage = "coarse_align"

    def run(self):
        self.begin()
        steps = self.scaled_steps("oracle_coarse_align_steps")
        z_offset = self.param("oracle_approach_z_offset")
        self.policy.guide.reset_integrator()

        goal = self.policy.gripper_pose(
            slerp_fraction=1.0,
            position_fraction=1.0,
            position_ref=self.frames.offset_ref(z_offset),
            z_offset=0.0,
            reset_xy_integrator=True,
            xy_integral_gain=0.0,
        )
        self.motion.execute_pose_trajectory(goal, steps)


class FineAlignStage(OracleStage):
    stage = "fine_align"

    def run(self):
        self.begin()
        steps = self.scaled_steps("oracle_fine_align_steps")
        end_z = self.param("oracle_hover_z_offset")

        self.policy.guide.reset_integrator()
        goal = self.policy.gripper_pose(
            slerp_fraction=1.0,
            position_fraction=1.0,
            position_ref=self.frames.offset_ref(end_z),
            z_offset=0.0,
            xy_integral_gain=0.0,
        )
        self.motion.execute_pose_trajectory(goal, steps)

        for _ in range(self.scaled_steps("oracle_fine_align_hold_steps", True)):
            self.motion.command_pose(self.live_alignment_pose(end_z))

    def live_alignment_pose(self, z_offset):
        return self.policy.gripper_pose(
            slerp_fraction=1.0,
            position_fraction=1.0,
            position_ref=self.frames.offset_ref(z_offset),
            z_offset=0.0,
            xy_integral_gain=self.alignment_gain(),
            xy_integrator_limit=self.integrator_limit(),
        )


class InsertStage(OracleStage):
    stage = "insert"

    def run(self):
        self.begin()
        speed = self.policy.speed_scale()
        z_offset = max(
            self.policy.current_z_offset(self.frames.port_ref),
            self.param("oracle_hover_z_offset"),
        )
        end_z = self.param("oracle_insert_end_z_offset")
        step_m = self.param("oracle_insert_step_meters") * speed

        while z_offset > end_z:
            z_offset = max(end_z, z_offset - step_m)
            pose = self.policy.gripper_pose(
                slerp_fraction=1.0,
                position_fraction=1.0,
                position_ref=self.frames.offset_ref(z_offset),
                z_offset=0.0,
                xy_integral_gain=self.alignment_gain(),
                xy_integrator_limit=self.integrator_limit(),
            )
            self.motion.command_pose(pose)
