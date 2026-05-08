import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from tqdm import tqdm


STAGES = ("approach", "coarse_align", "fine_align", "insert")
CAMERA_DTYPES = {"image", "video"}
SCHEMA_KEYS = ("dtype", "shape", "names")
FILE_SIZE_PER_EPISODE_MB = 1e-9


def main():
    merge_phase_datasets(parse_args())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge ch_milestones phase datasets into complete episodes."
    )
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--approach", type=Path)
    parser.add_argument("--coarse-align", type=Path)
    parser.add_argument("--fine-align", type=Path)
    parser.add_argument("--insert", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--task-mode", choices=("base", "keep"), default="base")
    parser.add_argument("--task-prompt")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--episodes")
    parser.add_argument("--max-position-jump", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def merge_phase_datasets(args):
    paths = phase_paths(args)
    validate_paths(paths)
    datasets = {
        stage: LeRobotDataset(
            local_repo_id(path), root=path, video_backend=args.video_backend
        )
        for stage, path in paths.items()
    }
    validate_inputs(datasets)

    out_root = args.output_root.expanduser()
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {out_root}")
        shutil.rmtree(out_root)

    first = datasets[STAGES[0]]
    features = user_features(first)
    output = LeRobotDataset.create(
        repo_id=args.repo_id or local_repo_id(out_root),
        root=out_root,
        fps=first.meta.fps,
        robot_type=first.meta.robot_type,
        features=features,
        use_videos=has_videos(features),
        video_backend=args.video_backend,
        batch_encoding_size=1,
        vcodec=args.vcodec,
    )
    output.meta.update_chunk_settings(
        data_files_size_in_mb=FILE_SIZE_PER_EPISODE_MB,
        video_files_size_in_mb=FILE_SIZE_PER_EPISODE_MB,
    )

    try:
        episodes = episode_indices(first.meta.total_episodes, args.episodes)
        skipped = []
        for episode_index in tqdm(episodes, desc="merging", unit="episode"):
            reason = corrupted_reason(datasets, episode_index, args)
            if reason is not None:
                skipped.append((episode_index, reason))
                tqdm.write(f"skipped episode {episode_index}: {reason}")
                continue
            merge_episode(output, datasets, episode_index, args)
            output.save_episode()
    finally:
        output.finalize()

    print(f"wrote {output.meta.total_episodes} episodes to {out_root}")
    if skipped:
        print(f"skipped {len(skipped)} corrupted episodes")


def phase_paths(args):
    if args.input_root is not None:
        root = args.input_root.expanduser()
        return {stage: root / stage for stage in STAGES}

    paths = {
        "approach": args.approach,
        "coarse_align": args.coarse_align,
        "fine_align": args.fine_align,
        "insert": args.insert,
    }
    missing = [stage for stage, path in paths.items() if path is None]
    if missing:
        raise ValueError(
            "--input-root or all four phase paths are required; missing "
            + ", ".join(missing)
        )
    return {stage: path.expanduser() for stage, path in paths.items()}


def episode_indices(total, spec):
    if spec is None:
        return list(range(total))
    if ":" in spec:
        start, stop = spec.split(":", 1)
        return list(range(int(start or 0), int(stop or total)))
    return [int(index) for index in spec.split(",")]


def validate_inputs(datasets):
    first = datasets[STAGES[0]]
    first_schema = schema(user_features(first))
    counts = {}
    for stage, dataset in datasets.items():
        counts[stage] = dataset.meta.total_episodes
        if dataset.meta.fps != first.meta.fps:
            raise ValueError(f"{stage} fps differs from approach")
        if schema(user_features(dataset)) != first_schema:
            raise ValueError(f"{stage} feature schema differs from approach")
    if len(set(counts.values())) != 1:
        raise ValueError(f"Phase episode counts differ: {counts}")
    if first.meta.total_episodes == 0:
        raise ValueError(f"Phase datasets contain no saved episodes: {counts}")


def validate_paths(paths):
    for stage, path in paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{stage} dataset directory not found: {path}")
        info = path / "meta" / "info.json"
        if not info.exists():
            raise FileNotFoundError(f"{stage} dataset metadata not found: {info}")


def corrupted_reason(datasets, episode_index, args):
    task_messages = set()
    previous_position = None
    for stage in STAGES:
        dataset = datasets[stage]
        episode = dataset.meta.episodes[episode_index]
        for index in range(
            episode["dataset_from_index"], episode["dataset_to_index"]
        ):
            row = dataset.get_raw_item(index)
            task_messages.add(row["task.message_json"])
            position = np.asarray(row["observation.state"])[:3]
            if previous_position is not None:
                jump = np.linalg.norm(position - previous_position)
                if jump > args.max_position_jump:
                    return f"TCP position jump {jump:.3f} m before {stage}:{index}"
            previous_position = position
    if len(task_messages) != 1:
        return f"multiple task messages ({len(task_messages)})"
    return None


def merge_episode(output, datasets, episode_index, args):
    for stage in STAGES:
        dataset = datasets[stage]
        episode = dataset.meta.episodes[episode_index]
        for index in range(
            episode["dataset_from_index"], episode["dataset_to_index"]
        ):
            row = dataset[index]
            output.add_frame(frame_from_row(dataset, row, args))


def frame_from_row(dataset, row, args):
    frame = {}
    for key, feature in user_features(dataset).items():
        frame[key] = value_for_writer(row[key], feature)
    frame["task"] = task_for(row["task"], args)
    return frame


def task_for(task, args):
    if args.task_prompt is not None:
        return args.task_prompt
    if args.task_mode == "base":
        return task.split("; stage: ", 1)[0]
    return task


def value_for_writer(value, feature):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if feature["dtype"] not in CAMERA_DTYPES:
        expected = tuple(feature["shape"])
        if hasattr(value, "shape") and value.shape != expected:
            value = np.asarray(value).reshape(expected)
        return value

    expected = tuple(feature["shape"])
    if value.shape == expected:
        return value
    transposed = value.transpose(1, 2, 0)
    if transposed.shape == expected:
        return transposed
    return value


def user_features(dataset):
    defaults = set(DEFAULT_FEATURES)
    return {
        key: clean_feature(feature)
        for key, feature in dataset.meta.features.items()
        if key not in defaults
    }


def clean_feature(feature):
    return {key: feature[key] for key in SCHEMA_KEYS if key in feature}


def schema(features):
    return json.dumps(features, sort_keys=True)


def has_videos(features):
    return any(feature["dtype"] == "video" for feature in features.values())


def local_repo_id(path):
    return f"local/{Path(path).expanduser().name}"


if __name__ == "__main__":
    main()
