import sys
import threading
import time
from pathlib import Path

import rclpy
from aic_task_interfaces.action import InsertCable
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ch_milestones.config.task_config import (
    declare_task_parameters,
    task_from_parameters,
)
from ch_milestones.environment.reset_client import ResetClient
from ch_milestones.recording.stage_episode_recorder import StageEpisodeRecorder


class MilestoneDataProducer(Node):
    def __init__(self):
        super().__init__("ch_milestone_data_producer")
        self.declare_parameter(
            "dataset_root", str(Path.home() / "aic_results" / "ch_milestones_lerobot")
        )
        self.declare_parameter("repo_id", "local/ch_milestones")
        self.declare_parameter("fps", 20)
        self.declare_parameter("task_prompt", "")
        self.declare_parameter("image_width", 1152)
        self.declare_parameter("image_height", 1024)
        self.declare_parameter("image_scale", 0.25)
        self.declare_parameter("use_videos", True)
        self.declare_parameter("vcodec", "h264")
        self.declare_parameter("recording_clock", "wall")
        self.declare_parameter("observation_timeout_seconds", 30.0)
        self.declare_parameter("settle_seconds", 1.0)
        self.declare_parameter("action_timeout_padding_seconds", 15.0)
        self.declare_parameter("episode_count", 1)
        self.declare_parameter("reset_before_episode", True)
        self.declare_parameter("clear_after_episodes", True)
        self.declare_parameter("reset_service_timeout_seconds", 300.0)
        self.declare_parameter("insertion_event_topic", "/scoring/insertion_event")
        self.declare_parameter("trim_on_insertion_event", True)
        self.declare_parameter("post_insertion_frames", 10)
        self.declare_parameter("require_insertion_event", True)
        declare_task_parameters(self)
        self.change_state = self.create_client(ChangeState, "/aic_model/change_state")
        self.insert_cable = ActionClient(self, InsertCable, "/insert_cable")
        self.resetter = ResetClient(self)
        self.current_recorder = None

    def run(self):
        for episode in range(self.get_parameter("episode_count").value):
            self.get_logger().info(f"Starting milestone episode {episode}")
            self.run_episode()
        if self.get_parameter("clear_after_episodes").value:
            self.resetter.clear(
                self.get_parameter("reset_service_timeout_seconds").value
            )
        self.transition(Transition.TRANSITION_UNCONFIGURED_SHUTDOWN)

    def run_episode(self):
        task = task_from_parameters(self)
        recorder = self.create_recorder()
        self.current_recorder = recorder
        if self.get_parameter("reset_before_episode").value:
            self.resetter.reset(
                self.get_parameter("reset_service_timeout_seconds").value
            )
        self.transition(Transition.TRANSITION_CONFIGURE)
        try:
            self.transition(Transition.TRANSITION_ACTIVATE)
            recorder.start(task)
            recorder.wait_for_observation(
                self.get_parameter("observation_timeout_seconds").value
            )
            result = self.run_task(task)
            time.sleep(self.get_parameter("settle_seconds").value)
            recorder.finish(result)
        finally:
            try:
                recorder.stop()
            finally:
                self.current_recorder = None
                self.transition(Transition.TRANSITION_DEACTIVATE)
                self.transition(Transition.TRANSITION_CLEANUP)
        self.get_logger().info(f"LeRobot episodes written to {recorder.path}")

    def create_recorder(self):
        scale = self.get_parameter("image_scale").value
        image_shape = (
            int(self.get_parameter("image_height").value * scale),
            int(self.get_parameter("image_width").value * scale),
            3,
        )
        return StageEpisodeRecorder(
            self,
            self.get_parameter("dataset_root").value,
            self.get_parameter("repo_id").value,
            self.get_parameter("fps").value,
            self.get_parameter("task_prompt").value,
            image_shape,
            self.get_parameter("use_videos").value,
            self.get_parameter("vcodec").value,
            self.get_parameter("insertion_event_topic").value,
            self.get_parameter("trim_on_insertion_event").value,
            self.get_parameter("post_insertion_frames").value,
            self.get_parameter("require_insertion_event").value,
            self.get_parameter("recording_clock").value,
        )

    def transition(self, transition_id: int):
        if not self.change_state.wait_for_service(timeout_sec=30.0):
            raise TimeoutError("/aic_model/change_state is not available")
        request = ChangeState.Request()
        request.transition.id = transition_id
        response = self.await_future(
            self.change_state.call_async(request), 60.0, "lifecycle transition"
        )
        if not response.success:
            raise RuntimeError(f"Lifecycle transition {transition_id} failed")

    def run_task(self, task):
        if not self.insert_cable.wait_for_server(timeout_sec=30.0):
            raise TimeoutError("/insert_cable action server is not available")
        goal = InsertCable.Goal()
        goal.task = task
        goal_handle = self.await_future(
            self.insert_cable.send_goal_async(goal, feedback_callback=self.feedback),
            30.0,
            "send InsertCable goal",
        )
        if not goal_handle.accepted:
            raise RuntimeError("InsertCable goal was rejected")
        timeout = (
            task.time_limit
            + self.get_parameter("action_timeout_padding_seconds").value
        )
        wrapped = self.await_future(
            goal_handle.get_result_async(), timeout, "InsertCable result"
        )
        result = {
            "status": wrapped.status,
            "success": wrapped.result.success,
            "message": wrapped.result.message,
        }
        if not wrapped.result.success:
            raise RuntimeError(f"InsertCable failed: {wrapped.result.message}")
        return result

    def feedback(self, msg):
        message = msg.feedback.message
        self.get_logger().info(message)
        if message.startswith("stage:") and self.current_recorder is not None:
            self.current_recorder.set_stage(message.removeprefix("stage:"))

    def await_future(self, future, timeout_sec: float, label: str):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise TimeoutError(f"Timed out waiting for {label}")
        return future.result()


def ros_args_with_defaults(args):
    argv = list(sys.argv if args is None else args)
    joined = " ".join(argv)
    defaults = []
    if "use_sim_time:=" not in joined:
        defaults += ["-p", "use_sim_time:=true"]
    if not defaults:
        return argv
    return argv + defaults if "--ros-args" in argv else argv + ["--ros-args"] + defaults


def main(args=None):
    rclpy.init(args=ros_args_with_defaults(args))
    producer = MilestoneDataProducer()
    executor = MultiThreadedExecutor()
    executor.add_node(producer)
    errors = []

    def worker():
        try:
            producer.run()
        except BaseException as exc:
            errors.append(exc)
            producer.get_logger().error(f"{type(exc).__name__}: {exc}")
        finally:
            executor.shutdown()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    thread.join()
    producer.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    if errors:
        raise errors[0]


if __name__ == "__main__":
    main()
