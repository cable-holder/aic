import threading
import time

from aic_model.policy import Policy
from aic_task_interfaces.msg import Task
from rclpy.node import Node

from ch_inference.commands import motion_update, pose_from_action
from ch_inference.completion import InsertionCompletion
from ch_inference.runtime import LeRobotRuntime
from ch_inference.schema import IMAGE_SHAPE, observation_frame, task_prompt


class VLAPolicy(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.policy_path = self.param("vla_policy_path", "")
        if not self.policy_path:
            raise ValueError("vla_policy_path is required")
        self.device = self.param("vla_device", "auto")
        self.rate_hz = float(self.param("vla_rate_hz", 20.0))
        self.timeout = float(self.param("vla_timeout_seconds", 170.0))
        self.frame_id = self.param("vla_command_frame", "base_link")
        self.robot_type = self.param("vla_robot_type", "aic")
        self.prompt = self.param("vla_task_prompt", "")
        self.configure_wait = float(self.param("vla_configure_wait_seconds", 45.0))
        self.activate_wait = float(self.param("vla_activate_wait_seconds", 55.0))
        self.stop_on_event = bool(self.param("vla_stop_on_insertion_event", True))
        self.event_topic = self.param(
            "vla_insertion_event_topic", "/scoring/insertion_event"
        )
        width = int(self.param("vla_image_width", IMAGE_SHAPE[1]))
        height = int(self.param("vla_image_height", IMAGE_SHAPE[0]))
        self.image_shape = (height, width, 3)
        self.runtime = None
        self.load_error = None
        self.load_done = threading.Event()
        threading.Thread(target=self.load_runtime, daemon=True).start()
        self.wait_ready(self.configure_wait, required=False)

    def param(self, name, default):
        self._parent_node.declare_parameter(name, default)
        return self._parent_node.get_parameter(name).value

    def load_runtime(self):
        try:
            self.runtime = LeRobotRuntime(self.policy_path, self.device)
            self.get_logger().info(f"Loaded VLA policy from {self.policy_path}")
        except Exception as exc:
            self.load_error = exc
        self.load_done.set()

    def wait_ready(self, timeout, required=True):
        self.load_done.wait(timeout)
        if self.load_error is not None:
            raise self.load_error
        if required and self.runtime is None:
            raise TimeoutError(f"VLA policy did not load within {timeout:.1f}s")

    def activate(self):
        self.wait_ready(self.activate_wait)

    def insert_cable(self, task: Task, get_observation, move_robot, send_feedback):
        self.wait_ready(self.timeout)
        self.runtime.reset()
        prompt = self.prompt or task_prompt(task)
        completion = (
            InsertionCompletion(self._parent_node, self.event_topic)
            if self.stop_on_event
            else None
        )
        deadline = time.monotonic() + min(self.timeout, float(task.time_limit or self.timeout))
        period = 1.0 / self.rate_hz
        send_feedback("stage:vla")

        while time.monotonic() < deadline:
            if completion is not None and completion.completed():
                completion.close(self._parent_node)
                send_feedback("completed:insertion_event")
                return True

            start = time.monotonic()
            observation = get_observation()
            if observation is None:
                raise RuntimeError("No observation available for VLA inference")
            action = self.runtime.predict(
                observation_frame(observation, self.image_shape),
                prompt,
                self.robot_type,
            )
            pose = pose_from_action(action)
            move_robot(
                motion_update=motion_update(
                    pose, self.get_clock().now().to_msg(), self.frame_id
                )
            )
            send_feedback("in progress")
            time.sleep(max(0.0, period - (time.monotonic() - start)))

        if completion is not None:
            completion.close(self._parent_node)
        send_feedback("timeout")
        return False
