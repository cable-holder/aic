from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def argument(name, default):
    return DeclareLaunchArgument(name, default_value=str(default).lower())


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    return LaunchDescription(
        [
            argument("use_sim_time", "true"),
            argument("vla_policy_path", ""),
            argument("vla_device", "auto"),
            argument("vla_rate_hz", 20.0),
            argument("vla_timeout_seconds", 170.0),
            argument("vla_command_frame", "base_link"),
            argument("vla_robot_type", "aic"),
            argument("vla_task_prompt", ""),
            argument("vla_configure_wait_seconds", 45.0),
            argument("vla_activate_wait_seconds", 55.0),
            argument("vla_image_width", 288),
            argument("vla_image_height", 256),
            argument("vla_stop_on_insertion_event", "true"),
            argument("vla_insertion_event_topic", "/scoring/insertion_event"),
            Node(
                package="aic_model",
                executable="aic_model",
                name="aic_model",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": typed("use_sim_time", bool),
                        "policy": "ch_inference.policies.VLAPolicy",
                        "vla_policy_path": LaunchConfiguration("vla_policy_path"),
                        "vla_device": LaunchConfiguration("vla_device"),
                        "vla_rate_hz": typed("vla_rate_hz", float),
                        "vla_timeout_seconds": typed("vla_timeout_seconds", float),
                        "vla_command_frame": LaunchConfiguration("vla_command_frame"),
                        "vla_robot_type": LaunchConfiguration("vla_robot_type"),
                        "vla_task_prompt": LaunchConfiguration("vla_task_prompt"),
                        "vla_configure_wait_seconds": typed(
                            "vla_configure_wait_seconds", float
                        ),
                        "vla_activate_wait_seconds": typed(
                            "vla_activate_wait_seconds", float
                        ),
                        "vla_image_width": typed("vla_image_width", int),
                        "vla_image_height": typed("vla_image_height", int),
                        "vla_stop_on_insertion_event": typed(
                            "vla_stop_on_insertion_event", bool
                        ),
                        "vla_insertion_event_topic": LaunchConfiguration(
                            "vla_insertion_event_topic"
                        ),
                    }
                ],
            ),
        ]
    )
