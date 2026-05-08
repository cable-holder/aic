import argparse
import csv
import heapq
from itertools import count
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def main():
    anomalies, top = check_dataset(parse_args())
    print_summary(anomalies, top)
    if anomalies:
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find sudden inter-frame image changes in a LeRobot dataset."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--episodes")
    parser.add_argument("--cameras")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def check_dataset(args):
    root = args.root.expanduser()
    validate_root(root)
    dataset = LeRobotDataset(
        args.repo_id or local_repo_id(root),
        root=root,
        video_backend=args.video_backend,
    )
    cameras = camera_keys(dataset, args.cameras)
    episodes = episode_indices(dataset.meta.total_episodes, args.episodes)
    anomalies = []
    top = []
    serial = count()

    with tqdm(
        total=transition_count(dataset, episodes),
        desc="checking",
        unit="transition",
        disable=args.no_progress,
    ) as progress:
        for episode_index in episodes:
            scan_episode(
                dataset,
                episode_index,
                cameras,
                args,
                anomalies,
                top,
                serial,
                progress,
            )

    if args.report is not None:
        write_report(args.report.expanduser(), anomalies)
    return anomalies, sorted((item[2] for item in top), key=lambda x: x["score"], reverse=True)


def scan_episode(dataset, episode_index, cameras, args, anomalies, top, serial, progress):
    episode = dataset.meta.episodes[episode_index]
    start = episode["dataset_from_index"]
    stop = episode["dataset_to_index"]
    if stop - start < 2:
        return

    previous = dataset[start]
    for index in range(start + 1, stop):
        current = dataset[index]
        for camera in cameras:
            score = frame_difference(previous[camera], current[camera])
            entry = {
                "episode": episode_index,
                "frame": scalar(current["frame_index"]),
                "previous_frame": scalar(previous["frame_index"]),
                "index": index,
                "previous_index": index - 1,
                "camera": camera,
                "score": score,
            }
            if score >= args.threshold:
                anomalies.append(entry)
                tqdm.write(format_entry(entry))
            keep_top(top, entry, args.top_k, serial)
        previous = current
        progress.update(1)


def frame_difference(a, b):
    a = image_array(a)
    b = image_array(b)
    if a.shape != b.shape:
        raise ValueError(f"Frame shapes differ: {a.shape} != {b.shape}")
    return float(np.mean(np.abs(a - b)))


def image_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def keep_top(top, entry, limit, serial):
    if limit <= 0:
        return
    item = (entry["score"], next(serial), entry)
    if len(top) < limit:
        heapq.heappush(top, item)
    elif entry["score"] > top[0][0]:
        heapq.heapreplace(top, item)


def camera_keys(dataset, spec):
    keys = list(dataset.meta.camera_keys)
    if spec is None:
        if not keys:
            raise ValueError("Dataset has no camera features")
        return keys
    selected = [name.strip() for name in spec.split(",") if name.strip()]
    missing = sorted(set(selected) - set(keys))
    if missing:
        raise ValueError(f"Unknown camera keys: {missing}")
    return selected


def episode_indices(total, spec):
    if spec is None:
        return list(range(total))
    if ":" in spec:
        start, stop = spec.split(":", 1)
        return list(range(int(start or 0), int(stop or total)))
    return [int(index) for index in spec.split(",")]


def transition_count(dataset, episodes):
    return sum(
        max(
            0,
            dataset.meta.episodes[index]["dataset_to_index"]
            - dataset.meta.episodes[index]["dataset_from_index"]
            - 1,
        )
        for index in episodes
    )


def write_report(path, anomalies):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(anomalies)


def print_summary(anomalies, top):
    print(f"found {len(anomalies)} abrupt frame changes")
    if top:
        print("largest changes:")
        for entry in top:
            print(format_entry(entry))


def format_entry(entry):
    return (
        f"episode={entry['episode']} frame={entry['frame']} "
        f"camera={entry['camera']} score={entry['score']:.4f}"
    )


def scalar(value):
    return int(value.item()) if hasattr(value, "item") else int(value)


def validate_root(root):
    info = root / "meta" / "info.json"
    if not info.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {info}")


def local_repo_id(path):
    return f"local/{Path(path).expanduser().name}"


FIELDNAMES = (
    "episode",
    "frame",
    "previous_frame",
    "index",
    "previous_index",
    "camera",
    "score",
)


if __name__ == "__main__":
    main()
