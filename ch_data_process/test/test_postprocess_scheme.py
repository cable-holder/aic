import json

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ch_data_process.postprocess_scheme import postprocess_scheme
from ch_data_process.rotations import quaternion_xyzw_from_rpy


FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (26,),
        "names": [
            "tcp_pose.position.x",
            "tcp_pose.position.y",
            "tcp_pose.position.z",
            "tcp_pose.orientation.x",
            "tcp_pose.orientation.y",
            "tcp_pose.orientation.z",
            "tcp_pose.orientation.w",
            "tcp_velocity.linear.x",
            "tcp_velocity.linear.y",
            "tcp_velocity.linear.z",
            "tcp_velocity.angular.x",
            "tcp_velocity.angular.y",
            "tcp_velocity.angular.z",
            "tcp_error.x",
            "tcp_error.y",
            "tcp_error.z",
            "tcp_error.rx",
            "tcp_error.ry",
            "tcp_error.rz",
            "joint_positions.0",
            "joint_positions.1",
            "joint_positions.2",
            "joint_positions.3",
            "joint_positions.4",
            "joint_positions.5",
            "joint_positions.6",
        ],
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
    "task.message_json": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
}


class Args:
    overwrite = False
    media_mode = "link"


def test_postprocess_scheme_converts_merged_dataset(tmp_path):
    src = tmp_path / "merged"
    dst = tmp_path / "vla"
    write_dataset(src)

    args = Args()
    args.input_root = src
    args.output_root = dst
    postprocess_scheme(args)

    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["total_tasks"] == 1
    assert "task.message_json" not in info["features"]
    assert info["features"]["observation.state"]["shape"] == [25]
    assert info["features"]["action"]["shape"] == [6]

    tasks = pd.read_parquet(dst / "meta" / "tasks.parquet")
    assert tasks.index.tolist() == ["insert cable"]

    dataset = LeRobotDataset("local/vla", root=dst)
    assert "task.message_json" not in dataset[0]
    assert dataset[0]["task"] == "insert cable"
    np.testing.assert_allclose(
        dataset[0]["observation.state"][:6].numpy(),
        np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        dataset[0]["action"].numpy(),
        np.array([4.0, 5.0, 6.0, 0.4, 0.5, 0.6], dtype=np.float32),
        atol=1e-6,
    )

    stats = json.loads((dst / "meta" / "stats.json").read_text())
    assert len(stats["observation.state"]["mean"]) == 25
    assert len(stats["action"]["mean"]) == 6
    assert stats["task_index"]["max"] == [0.0]


def write_dataset(root):
    dataset = LeRobotDataset.create(
        repo_id="local/merged",
        root=root,
        fps=20,
        robot_type="aic_controller",
        features=FEATURES,
        use_videos=False,
    )
    try:
        add_frame(dataset, "insert cable; stage: approach", [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        add_frame(dataset, "insert cable; stage: insert", [0.2, 0.1, 0.4], [0.5, 0.4, 0.7])
        dataset.save_episode()
    finally:
        dataset.finalize()


def add_frame(dataset, task, state_rpy, action_rpy):
    state = np.concatenate(
        (
            np.array([1, 2, 3], dtype=np.float32),
            quaternion_xyzw_from_rpy(np.array(state_rpy, dtype=np.float32)),
            np.arange(19, dtype=np.float32),
        )
    )
    action = np.concatenate(
        (
            np.array([4, 5, 6], dtype=np.float32),
            quaternion_xyzw_from_rpy(np.array(action_rpy, dtype=np.float32)),
        )
    )
    dataset.add_frame(
        {
            "observation.state": state,
            "action": action,
            "task.message_json": "{}",
            "task": task,
        }
    )
