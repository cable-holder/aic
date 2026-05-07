from types import SimpleNamespace

from geometry_msgs.msg import Transform
from pytest import approx

from ch_milestones.policies.ground_truth_guidance import GroundTruthGuide
from ch_milestones.policies.oracle_frames import OracleFrames


class Logger:
    def info(self, *_):
        pass


class Node:
    def get_logger(self):
        return Logger()


def transform(x, y, z):
    value = Transform()
    value.translation.x = x
    value.translation.y = y
    value.translation.z = z
    value.rotation.w = 1.0
    return value


class StaticGuide(GroundTruthGuide):
    def __init__(self, plug, gripper):
        params = {
            "oracle_alignment_integral_gain": 0.0,
            "oracle_alignment_integrator_limit": 0.05,
        }
        task = SimpleNamespace(cable_name="cable", plug_name="plug")
        super().__init__(Node(), task, param=params.__getitem__)
        self.plug = plug
        self.gripper = gripper

    def synchronized_transforms(self, *_):
        return self.plug, self.gripper


def test_alignment_gripper_pose_uses_position_ref_for_xy():
    guide = StaticGuide(
        plug=transform(1.0, 2.0, 3.0),
        gripper=transform(0.8, 1.7, 3.4),
    )

    pose = guide.alignment_gripper_pose(
        orientation_ref=transform(10.0, 20.0, 30.0),
        position_ref=transform(4.0, 5.0, 6.0),
        z_offset=0.2,
        reset_xy_integrator=True,
    )
    goal = guide.last_gripper_pose_debug["goal_plug"].translation

    assert pose.position.x == approx(3.8)
    assert pose.position.y == approx(4.7)
    assert pose.position.z == approx(5.8)
    assert (goal.x, goal.y, goal.z) == approx((4.0, 5.0, 6.2))


def test_oracle_frames_uses_port_link_for_alignment_position():
    task = SimpleNamespace(
        target_module_name="module",
        port_name="port",
        cable_name="cable",
        plug_name="plug",
    )
    frames = {
        "task_board/module/port_link": transform(1.0, 2.0, 3.0),
    }
    guide = SimpleNamespace(
        wait_for=lambda *_: None,
        transform=lambda _, source: frames[source],
    )

    loaded = OracleFrames.load(Node(), guide, task)

    assert loaded.orientation_ref.translation.x == 1.0
    assert loaded.port_ref.translation.x == 1.0
