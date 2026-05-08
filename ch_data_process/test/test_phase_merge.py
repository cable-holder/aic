import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ch_data_process.phase_merge import STAGES, merge_phase_datasets


FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["x", "y", "z"],
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["target_a", "target_b"],
    },
    "task.message_json": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
}


class Args:
    input_root = None
    task_mode = "base"
    task_prompt = None
    repo_id = "local/complete"
    vcodec = "h264"
    video_backend = "pyav"
    episodes = None
    max_position_jump = 0.10
    overwrite = False


def test_merge_phase_datasets_appends_matching_episodes(tmp_path):
    args = Args()
    args.output_root = tmp_path / "complete"

    lengths = {}
    for stage_index, stage in enumerate(STAGES):
        root = tmp_path / stage
        lengths[stage] = stage_index + 1
        write_phase_dataset(root, stage, lengths[stage])
    args.approach = tmp_path / "approach"
    args.coarse_align = tmp_path / "coarse_align"
    args.fine_align = tmp_path / "fine_align"
    args.insert = tmp_path / "insert"

    merge_phase_datasets(args)

    merged = LeRobotDataset("local/complete", root=args.output_root)
    assert merged.meta.total_episodes == 2
    assert merged.meta.total_frames == 2 * sum(lengths.values())
    assert merged.meta.episodes[0]["length"] == sum(lengths.values())
    assert merged[0]["task"] == "insert cable"
    assert len(list((args.output_root / "data").glob("*/*.parquet"))) == 2


def test_merge_skips_corrupted_episode(tmp_path):
    args = Args()
    args.output_root = tmp_path / "complete"

    for stage in STAGES:
        write_phase_dataset(tmp_path / stage, stage, 2, corrupt=stage == "insert")
    args.approach = tmp_path / "approach"
    args.coarse_align = tmp_path / "coarse_align"
    args.fine_align = tmp_path / "fine_align"
    args.insert = tmp_path / "insert"

    merge_phase_datasets(args)

    merged = LeRobotDataset("local/complete", root=args.output_root)
    assert merged.meta.total_episodes == 1


def write_phase_dataset(root, stage, frame_count, corrupt=False):
    dataset = LeRobotDataset.create(
        repo_id=f"local/{stage}",
        root=root,
        fps=20,
        robot_type="aic_controller",
        features=FEATURES,
        use_videos=False,
    )
    try:
        for episode_index in range(2):
            for frame_index in range(frame_count):
                position = np.array(
                    [frame_index * 0.001, 0.0, 0.0], dtype=np.float32
                )
                if corrupt and episode_index == 1 and frame_index == 1:
                    position = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                dataset.add_frame(
                    {
                        "observation.state": position,
                        "action": np.array(
                            [frame_index, frame_count], dtype=np.float32
                        ),
                        "task.message_json": "{}",
                        "task": f"insert cable; stage: {stage.replace('_', ' ')}",
                    }
                )
            dataset.save_episode()
    finally:
        dataset.finalize()
