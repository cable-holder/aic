from geometry_msgs.msg import Point, Pose, Quaternion

from ch_milestones.policies.cartesian_trajectory import minimum_jerk_pose_trajectory


def pose(x):
    return Pose(
        position=Point(x=x, y=0.0, z=0.0),
        orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )


def test_minimum_jerk_pose_trajectory_uses_fixed_endpoints():
    trajectory = list(minimum_jerk_pose_trajectory(pose(0.0), pose(10.0), 3))

    assert [point.position.x for point in trajectory] == [0.0, 5.0, 10.0]


def test_minimum_jerk_pose_trajectory_single_step_reaches_goal():
    trajectory = list(minimum_jerk_pose_trajectory(pose(0.0), pose(10.0), 1))

    assert [point.position.x for point in trajectory] == [10.0]
