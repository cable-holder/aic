import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch_data_process.rotations import rpy_from_quaternion_xyzw


ACTION_NAMES = [
    "target_tcp_pose.position.x",
    "target_tcp_pose.position.y",
    "target_tcp_pose.position.z",
    "target_tcp_pose.roll",
    "target_tcp_pose.pitch",
    "target_tcp_pose.yaw",
]
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def main():
    convert_action_to_rpy(parse_args())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert absolute Cartesian quaternion actions to roll-pitch-yaw."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--media-mode", choices=("link", "copy"), default="link")
    return parser.parse_args()


def convert_action_to_rpy(args):
    src = args.input_root.expanduser()
    dst = args.output_root.expanduser()
    copy_dataset(src, dst, args)
    action_stats, episode_stats = convert_data(dst)
    update_info(dst)
    update_stats(dst, action_stats)
    update_episode_stats(dst, episode_stats)
    print(f"wrote RPY action dataset to {dst}")


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


def convert_data(root):
    action_batches = []
    episode_batches = {}
    files = sorted((root / "data").glob("*/*.parquet"))
    for path in tqdm(files, desc="converting", unit="file"):
        df = pd.read_parquet(path)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        if action.shape[1] != 7:
            raise ValueError(f"{path} action must be [x,y,z,qx,qy,qz,qw]")
        converted = np.concatenate(
            (action[:, :3], rpy_from_quaternion_xyzw(action[:, 3:])), axis=1
        ).astype(np.float32)
        unwrap_by_episode(converted, df["episode_index"].to_numpy())
        df["action"] = list(converted)
        df.to_parquet(path, index=False)
        action_batches.append(converted)
        for episode_index in np.unique(df["episode_index"].to_numpy()):
            mask = df["episode_index"].to_numpy() == episode_index
            episode_batches[int(episode_index)] = converted[mask]
    return stats(np.concatenate(action_batches)), {
        episode: stats(values) for episode, values in episode_batches.items()
    }


def unwrap_by_episode(action, episode_indices):
    for episode_index in np.unique(episode_indices):
        mask = episode_indices == episode_index
        action[mask, 3:6] = np.unwrap(action[mask, 3:6], axis=0)


def update_info(root):
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    info["features"]["action"]["shape"] = [6]
    info["features"]["action"]["names"] = ACTION_NAMES
    path.write_text(json.dumps(info, indent=4) + "\n")


def update_stats(root, action_stats):
    path = root / "meta" / "stats.json"
    all_stats = json.loads(path.read_text())
    all_stats["action"] = action_stats
    path.write_text(json.dumps(all_stats, indent=4) + "\n")


def update_episode_stats(root, episode_stats):
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    for path in tqdm(files, desc="episodes", unit="file"):
        df = pd.read_parquet(path)
        for row_index, episode_index in enumerate(df["episode_index"].to_numpy()):
            values = episode_stats[int(episode_index)]
            for name, value in values.items():
                df.at[row_index, f"stats/action/{name}"] = value
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
