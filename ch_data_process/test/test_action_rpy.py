import json

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ch_data_process.action_rpy import ACTION_NAMES, convert_action_to_rpy
from ch_data_process.rotations import (
    quaternion_xyzw_from_rpy,
    rpy_from_quaternion_xyzw,
)


FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["x", "y", "z"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "target_tcp_pose.position.x",
            "target_tcp_pose.position.y",
            "target_tcp_pose.position.z",
            "target_tcp_pose.orientation.x",
            "target_tcp_pose.orientation.y",
            "target_tcp_pose.orientation.z",
            "target_tcp_pose.orientation.w",
        ],
    },
}


class Args:
    overwrite = False
    media_mode = "link"


def test_quaternion_rpy_round_trip():
    rpy = np.array([[0.2, -0.1, 0.3], [0.0, 0.4, -0.5]], dtype=np.float32)
    quat = quaternion_xyzw_from_rpy(rpy)
    np.testing.assert_allclose(rpy_from_quaternion_xyzw(quat), rpy, atol=1e-6)


def test_convert_action_to_rpy_updates_data_and_metadata(tmp_path):
    src = tmp_path / "quat"
    dst = tmp_path / "rpy"
    write_dataset(src)

    args = Args()
    args.input_root = src
    args.output_root = dst
    convert_action_to_rpy(args)

    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["features"]["action"]["shape"] == [6]
    assert info["features"]["action"]["names"] == ACTION_NAMES

    dataset = LeRobotDataset("local/rpy", root=dst)
    np.testing.assert_allclose(
        dataset[0]["action"].numpy(),
        np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32),
        atol=1e-6,
    )
    stats = json.loads((dst / "meta" / "stats.json").read_text())
    assert len(stats["action"]["mean"]) == 6


def test_convert_action_to_rpy_unwraps_episode_angles(tmp_path):
    src = tmp_path / "quat"
    dst = tmp_path / "rpy"
    write_dataset(src, rpy_values=[(0.0, 0.0, 3.12), (0.0, 0.0, -3.12)])

    args = Args()
    args.input_root = src
    args.output_root = dst
    convert_action_to_rpy(args)

    dataset = LeRobotDataset("local/rpy", root=dst)
    assert dataset[1]["action"][5] > 3.12


def write_dataset(root, rpy_values=None):
    rpy_values = rpy_values or [(0.1, 0.2, 0.3), (0.2, 0.1, 0.4)]
    dataset = LeRobotDataset.create(
        repo_id="local/quat",
        root=root,
        fps=20,
        robot_type="aic_controller",
        features=FEATURES,
        use_videos=False,
    )
    try:
        for index, rpy in enumerate(rpy_values):
            quat = quaternion_xyzw_from_rpy(np.array(rpy, dtype=np.float32))
            dataset.add_frame(
                {
                    "observation.state": np.zeros(3, dtype=np.float32),
                    "action": np.concatenate(
                        (np.array([1, 2, 3], dtype=np.float32), quat)
                    ),
                    "task": "insert cable",
                }
            )
        dataset.save_episode()
    finally:
        dataset.finalize()
