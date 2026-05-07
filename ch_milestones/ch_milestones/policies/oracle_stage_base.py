from math import ceil
from time import monotonic


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
        self.policy.handoff_target_to_current()

    def param(self, name):
        return self.policy.param(name)

    def scaled_steps(self, name, allow_zero=False):
        steps = self.param(name)
        if allow_zero and steps == 0:
            return 0
        return max(1, ceil(steps / self.policy.speed_scale()))

    def timeout_seconds(self):
        return float(self.param("oracle_stage_timeout_seconds"))

    def timeout_start(self):
        return monotonic()

    def timed_out(self, start):
        return monotonic() - start >= self.timeout_seconds()

    def timeout_error(self):
        raise TimeoutError(
            f"{self.stage} did not succeed within "
            f"{self.timeout_seconds():.1f}s wall time"
        )

    def alignment_gain(self):
        return self.param("oracle_alignment_integral_gain")

    def integrator_limit(self):
        return self.param("oracle_alignment_integrator_limit")

    def progress(self, step, steps):
        return step / steps

    def command_pose(self, pose, command_period=None, target_plug=None):
        explicit_target_plug = target_plug is not None
        current_tcp, current_plug = self.motion.current_tcp_and_plug()
        if target_plug is None:
            target_plug = self.policy.debug_frames.predicted_child_transform(
                current_tcp,
                current_plug,
                pose,
            )
        goal_plug = None if explicit_target_plug else self.goal_plug()
        if goal_plug is not None:
            self.policy.debug_frames.publish_goal_plug_frame(goal_plug)
        self.policy.debug_frames.publish_tcp_frames(
            current_tcp,
            pose,
            current_plug,
            target_plug,
        )
        self.policy.set_pose_target(
            move_robot=self.policy.move_robot,
            pose=pose,
            stiffness=self.param("oracle_cartesian_stiffness"),
            damping=self.param("oracle_cartesian_damping"),
        )
        self.policy.sleep_for(
            self.policy.command_period()
            if command_period is None
            else command_period
        )
        self.policy.divergence_guard.check(goal_plug or target_plug)

    def goal_plug(self):
        debug = self.policy.guide.last_gripper_pose_debug
        if debug is None:
            return None
        return debug.get("goal_plug") or debug.get("desired_plug")
