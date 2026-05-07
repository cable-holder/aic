from types import SimpleNamespace

from geometry_msgs.msg import Transform
import pytest

from ch_milestones.policies.oracle_divergence import (
    OracleDivergenceGuard,
    StageDivergenceError,
)


def transform(x):
    value = Transform()
    value.translation.x = x
    return value


class Guide:
    def __init__(self, positions):
        self.positions = list(positions)

    def transform(self, target_frame, source_frame):
        del target_frame, source_frame
        return transform(self.positions.pop(0))


def policy(positions):
    params = {
        "oracle_divergence_check_enabled": True,
        "oracle_divergence_tolerance_meters": 0.1,
        "oracle_divergence_consecutive_steps": 2,
    }
    return SimpleNamespace(
        guide=Guide(positions),
        frames=SimpleNamespace(plug_frame="plug"),
        param=lambda name: params[name],
    )


def test_divergence_guard_raises_after_consecutive_bad_steps():
    guard = OracleDivergenceGuard(policy([0.0, 0.05, 0.2, 0.25]))
    guard.reset("fine_align")
    goal = transform(0.0)

    guard.check(goal)
    guard.check(goal)
    guard.check(goal)
    with pytest.raises(StageDivergenceError):
        guard.check(goal)
