from types import SimpleNamespace

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ch_data_process.frame_similarity import check_dataset


FEATURES = {
    "observation.images.test_camera": {
        "dtype": "image",
        "shape": (8, 8, 3),
        "names": ["height", "width", "channels"],
    },
}


def test_frame_similarity_finds_scene_jump(tmp_path):
    root = tmp_path / "dataset"
    write_image_dataset(root)

    anomalies, top = check_dataset(
        SimpleNamespace(
            root=root,
            repo_id="local/images",
            threshold=0.5,
            video_backend="pyav",
            episodes=None,
            cameras=None,
            top_k=3,
            report=None,
            no_progress=True,
        )
    )

    assert len(anomalies) == 1
    assert anomalies[0]["episode"] == 1
    assert anomalies[0]["frame"] == 2
    assert anomalies[0]["camera"] == "observation.images.test_camera"
    assert top[0]["score"] == anomalies[0]["score"]


def write_image_dataset(root):
    dataset = LeRobotDataset.create(
        repo_id="local/images",
        root=root,
        fps=20,
        robot_type="aic_controller",
        features=FEATURES,
        use_videos=False,
    )
    try:
        for values in ((0, 8, 16), (0, 0, 255)):
            for value in values:
                dataset.add_frame(
                    {
                        "observation.images.test_camera": np.full(
                            (8, 8, 3), value, dtype=np.uint8
                        ),
                        "task": "test",
                    }
                )
            dataset.save_episode()
    finally:
        dataset.finalize()
