from math import sqrt


class StageDivergenceError(RuntimeError):
    pass


class OracleDivergenceGuard:
    def __init__(self, policy):
        self.policy = policy
        self.reset(None)

    def reset(self, stage):
        self.stage = stage
        self.best_distance = None
        self.bad_steps = 0

    def check(self, goal_plug):
        if not self.policy.param("oracle_divergence_check_enabled"):
            return
        if goal_plug is None:
            return

        plug = self.policy.guide.transform("base_link", self.policy.frames.plug_frame)
        distance = self.distance(plug, goal_plug)
        if self.best_distance is None or distance <= self.best_distance:
            self.best_distance = distance
            self.bad_steps = 0
            return

        tolerance = self.policy.param("oracle_divergence_tolerance_meters")
        if distance <= self.best_distance + tolerance:
            self.bad_steps = 0
            return

        self.bad_steps += 1
        limit = int(self.policy.param("oracle_divergence_consecutive_steps"))
        if self.bad_steps >= limit:
            raise StageDivergenceError(
                f"{self.stage} diverged from target: "
                f"distance={distance:.4f} best={self.best_distance:.4f}"
            )

    def distance(self, start, goal):
        dx = goal.translation.x - start.translation.x
        dy = goal.translation.y - start.translation.y
        dz = goal.translation.z - start.translation.z
        return sqrt(dx * dx + dy * dy + dz * dz)
