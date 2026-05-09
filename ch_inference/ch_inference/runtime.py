from contextlib import nullcontext
from copy import copy
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


class LeRobotRuntime:
    def __init__(self, policy_path, device="auto"):
        self.path = self.policy_path(policy_path)
        target = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.device = torch.device("cpu" if target == "auto" else target)
        self.config = PreTrainedConfig.from_pretrained(self.path)
        self.config.pretrained_path = self.path
        self.config.device = str(self.device)
        policy_class = get_policy_class(self.config.type)
        self.policy = policy_class.from_pretrained(self.path, config=self.config)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=self.path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    @staticmethod
    def policy_path(policy_path):
        raw = str(policy_path)
        path = Path(raw).expanduser()
        if path.exists():
            LeRobotRuntime.checkpoint_files(path)
            return str(path)
        if path.is_absolute() or raw.startswith(("~", ".")):
            raise FileNotFoundError(f"LeRobot checkpoint not found: {path}")
        return raw

    @staticmethod
    def checkpoint_files(path):
        missing = [
            name
            for name in (
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            )
            if not (path / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete LeRobot checkpoint {path}: missing {', '.join(missing)}"
            )

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict(self, frame, task, robot_type):
        observation = copy(frame)
        context = (
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), context:
            observation = prepare_observation_for_inference(
                observation, self.device, task, robot_type
            )
            action = self.policy.select_action(self.preprocessor(observation))
            action = self.postprocessor(action)
        return np.asarray(action.squeeze(0).cpu(), dtype=np.float32)
