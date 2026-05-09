import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch_data_process.rotations import rpy_from_quaternion_xyzw


OBSERVATION_NAMES = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.roll",
    "tcp_pose.orientation.pitch",
    "tcp_pose.orientation.yaw",
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
]
ACTION_NAMES = [
    "target_tcp_pose.position.x",
    "target_tcp_pose.position.y",
    "target_tcp_pose.position.z",
    "target_tcp_pose.orientation.roll",
    "target_tcp_pose.orientation.pitch",
    "target_tcp_pose.orientation.yaw",
]
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def main():
    postprocess_scheme(parse_args())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a merged ch_milestones dataset to the VLA training scheme."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--media-mode", choices=("link", "copy"), default="link")
    return parser.parse_args()


def postprocess_scheme(args):
    src = args.input_root.expanduser()
    dst = args.output_root.expanduser()
    task_map, tasks = task_mapping(src)
    copy_dataset(src, dst, args)
    stats_by_key, episode_stats = convert_data(dst, task_map)
    update_info(dst, tasks)
    update_tasks(dst, tasks)
    update_stats(dst, stats_by_key)
    update_episode_stats(dst, episode_stats)
    print(f"wrote VLA training dataset to {dst}")


def task_mapping(root):
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    base_by_old = {
        int(row.task_index): base_task(task)
        for task, row in tasks.iterrows()
    }
    unique = list(OrderedDict.fromkeys(base_by_old.values()))
    index_by_task = {task: index for index, task in enumerate(unique)}
    return {old: index_by_task[task] for old, task in base_by_old.items()}, unique


def base_task(task):
    return str(task).split("; stage: ", 1)[0]


def copy_dataset(src, dst, args):
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"LeRobot dataset metadata not found: {src}")
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {dst}")
        shutil.rmtree(dst)
    if src.resolve() in dst.resolve().parents:
        raise ValueError("output-root must not be inside input-root")
    for path in tqdm(sorted(src.rglob("*")), desc="copying", unit="file"):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.relative_to(src).parts[0] in {"videos", "images"} and args.media_mode == "link":
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def convert_data(root, task_map):
    batches = {"observation.state": [], "action": [], "task_index": []}
    episode_batches = {}
    for path in tqdm(sorted((root / "data").glob("*/*.parquet")), desc="converting", unit="file"):
        df = pd.read_parquet(path)
        state = convert_state(np.stack(df["observation.state"].to_numpy()))
        action = convert_action(np.stack(df["action"].to_numpy()))
        task_index = df["task_index"].map(task_map).to_numpy(np.int64)
        unwrap_by_episode(state, action, df["episode_index"].to_numpy())
        df["observation.state"] = list(state)
        df["action"] = list(action)
        df["task_index"] = task_index
        df = df.drop(columns=["task.message_json"])
        df.to_parquet(path, index=False)
        batches["observation.state"].append(state)
        batches["action"].append(action)
        batches["task_index"].append(task_index[:, None])
        for episode_index in np.unique(df["episode_index"].to_numpy()):
            mask = df["episode_index"].to_numpy() == episode_index
            episode_batches.setdefault(int(episode_index), {
                "observation.state": [],
                "action": [],
                "task_index": [],
            })
            episode_batches[int(episode_index)]["observation.state"].append(state[mask])
            episode_batches[int(episode_index)]["action"].append(action[mask])
            episode_batches[int(episode_index)]["task_index"].append(task_index[mask, None])
    return {key: stats(np.concatenate(value)) for key, value in batches.items()}, {
        episode: {key: stats(np.concatenate(values)) for key, values in data.items()}
        for episode, data in episode_batches.items()
    }


def convert_state(state):
    if state.shape[1] != 26:
        raise ValueError("observation.state must be 26D quaternion TCP state")
    return np.concatenate(
        (
            state[:, :3],
            rpy_from_quaternion_xyzw(state[:, 3:7]),
            state[:, 7:],
        ),
        axis=1,
    ).astype(np.float32)


def convert_action(action):
    if action.shape[1] != 7:
        raise ValueError("action must be 7D absolute Cartesian quaternion pose")
    return np.concatenate(
        (action[:, :3], rpy_from_quaternion_xyzw(action[:, 3:])), axis=1
    ).astype(np.float32)


def unwrap_by_episode(state, action, episode_indices):
    for episode_index in np.unique(episode_indices):
        mask = episode_indices == episode_index
        state[mask, 3:6] = np.unwrap(state[mask, 3:6], axis=0)
        action[mask, 3:6] = np.unwrap(action[mask, 3:6], axis=0)


def update_info(root, tasks):
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    info["total_tasks"] = len(tasks)
    info["features"]["observation.state"]["shape"] = [25]
    info["features"]["observation.state"]["names"] = OBSERVATION_NAMES
    info["features"]["action"]["shape"] = [6]
    info["features"]["action"]["names"] = ACTION_NAMES
    info["features"].pop("task.message_json")
    path.write_text(json.dumps(info, indent=4) + "\n")


def update_tasks(root, tasks):
    df = pd.DataFrame(
        {"task_index": list(range(len(tasks)))},
        index=pd.Index(tasks, name="task"),
    )
    df.to_parquet(root / "meta" / "tasks.parquet")


def update_stats(root, stats_by_key):
    path = root / "meta" / "stats.json"
    all_stats = json.loads(path.read_text())
    all_stats.update(stats_by_key)
    path.write_text(json.dumps(all_stats, indent=4) + "\n")


def update_episode_stats(root, episode_stats):
    for path in tqdm(sorted((root / "meta" / "episodes").glob("*/*.parquet")), desc="episodes", unit="file"):
        df = pd.read_parquet(path)
        for row, episode_index in enumerate(df["episode_index"].to_numpy()):
            values = episode_stats[int(episode_index)]
            task = int(values["task_index"]["min"][0])
            df.at[row, "tasks"] = [base_task(df.at[row, "tasks"][0])]
            for key, key_stats in values.items():
                for name, value in key_stats.items():
                    df.at[row, f"stats/{key}/{name}"] = value
            df.at[row, "stats/task_index/min"] = [task]
            df.at[row, "stats/task_index/max"] = [task]
        df.to_parquet(path, index=False)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    result = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    result.update(
        {key: np.quantile(values, q, axis=0).tolist() for key, q in QUANTILES.items()}
    )
    return result


if __name__ == "__main__":
    main()
