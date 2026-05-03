from types import SimpleNamespace

from ch_milestones.config.randomization_config import (
    DISTRACTOR_BOARD_PART_DEFAULTS,
    RANDOMIZATION_DEFAULTS,
    SC_CABLE_RANDOMIZATION_PREFIX,
    SC_TASK_BOARD_RANDOMIZATION_PREFIX,
)
from ch_milestones.config.scene_config import (
    BOARD_PART_DEFAULTS,
    CABLE_POSE_PRESETS,
)
from ch_milestones.config.task_options import (
    SC_TARGET_MODULES,
    SFP_TARGET_MODULES,
)
from ch_milestones.environment.randomization import SceneRandomizer


class Parameter:
    def __init__(self, value):
        self.value = value


class Node:
    def __init__(self, values):
        self.values = values

    def get_parameter(self, name):
        return Parameter(self.values[name])


def test_sc_ports_have_shared_randomization_parameters():
    assert "sc_port_translation_min" in RANDOMIZATION_DEFAULTS
    assert "sc_port_translation_max" in RANDOMIZATION_DEFAULTS
    assert "sc_port_yaw_min" in RANDOMIZATION_DEFAULTS
    assert "sc_port_yaw_max" in RANDOMIZATION_DEFAULTS

    assert "sc_port_0_translation_min" not in RANDOMIZATION_DEFAULTS
    assert "sc_port_1_translation_min" not in RANDOMIZATION_DEFAULTS


def test_sc_targets_have_separate_board_and_cable_jitter_parameters():
    assert "sc_task_board_x_min" in RANDOMIZATION_DEFAULTS
    assert "sc_task_board_yaw_max" in RANDOMIZATION_DEFAULTS
    assert "sc_cable_z_min" in RANDOMIZATION_DEFAULTS
    assert "sc_cable_yaw_max" in RANDOMIZATION_DEFAULTS

    expected_sc_cable_z = CABLE_POSE_PRESETS["sc"]["cable_z"]
    assert RANDOMIZATION_DEFAULTS["sc_cable_z_min"] == expected_sc_cable_z
    assert RANDOMIZATION_DEFAULTS["sc_cable_z_max"] == expected_sc_cable_z


def test_distractor_board_part_presence_parameters_are_declared():
    for name, value in DISTRACTOR_BOARD_PART_DEFAULTS.items():
        assert RANDOMIZATION_DEFAULTS[name] == value


def test_randomizer_uses_sc_pose_randomization_keys_for_sc_targets():
    randomizer = SceneRandomizer(Node({"random_seed": -1}))

    assert randomizer.pose_randomization_key(
        "task_board", SimpleNamespace(port_type="sc")
    ) == SC_TASK_BOARD_RANDOMIZATION_PREFIX
    assert randomizer.pose_randomization_key(
        "cable", SimpleNamespace(port_type="sc")
    ) == SC_CABLE_RANDOMIZATION_PREFIX
    assert (
        randomizer.pose_randomization_key(
            "task_board", SimpleNamespace(port_type="sfp")
        )
        == "task_board"
    )


def test_randomizer_uses_shared_sc_port_randomization_keys():
    randomizer = SceneRandomizer(Node({"random_seed": -1}))

    assert randomizer.randomization_key("sc_port_1", "translation") == (
        "sc_port_translation"
    )
    assert randomizer.randomization_key("sc_port_1", "yaw") == "sc_port_yaw"
    assert randomizer.randomization_key("nic_card_mount_1", "translation") == (
        "nic_card_mount_1_translation"
    )


def board_parts():
    parts = {}
    for name, values in BOARD_PART_DEFAULTS.items():
        present, translation, roll, pitch, yaw = values
        parts[f"{name}_present"] = present
        parts[f"{name}_translation"] = translation
        parts[f"{name}_roll"] = roll
        parts[f"{name}_pitch"] = pitch
        parts[f"{name}_yaw"] = yaw
    return parts


def presence_node(**overrides):
    values = {
        "random_seed": 1,
        "randomize_distractor_board_part_presence": True,
        "nic_card_mount_distractor_presence_probability": 0.0,
        "sc_port_distractor_presence_probability": 0.0,
    }
    values.update(overrides)
    return Node(values)


def test_randomizer_keeps_sfp_target_mount_present_and_can_hide_all_distractors():
    task = SimpleNamespace(
        port_type="sfp",
        target_module_name="nic_card_mount_4",
    )
    randomizer = SceneRandomizer(presence_node())
    randomized = randomizer.randomized_distractor_board_part_presence(
        board_parts(), task
    )

    assert randomized["nic_card_mount_4_present"] is True
    for name in (*SFP_TARGET_MODULES, *SC_TARGET_MODULES):
        if name == "nic_card_mount_4":
            continue
        assert randomized[f"{name}_present"] is False


def test_randomizer_keeps_sc_target_port_present_and_can_spawn_all_distractors():
    task = SimpleNamespace(
        port_type="sc",
        target_module_name="sc_port_1",
    )
    randomized = SceneRandomizer(
        presence_node(
            nic_card_mount_distractor_presence_probability=1.0,
            sc_port_distractor_presence_probability=1.0,
        )
    ).randomized_distractor_board_part_presence(
        board_parts(),
        task,
    )

    for name in (*SFP_TARGET_MODULES, *SC_TARGET_MODULES):
        assert randomized[f"{name}_present"] is True


def test_randomizer_can_leave_board_part_presence_fixed():
    parts = board_parts()
    task = SimpleNamespace(port_type="sc", target_module_name="sc_port_1")
    randomized = SceneRandomizer(
        presence_node(randomize_distractor_board_part_presence=False)
    ).randomized_distractor_board_part_presence(
        parts,
        task,
    )

    expected = dict(parts)
    expected["sc_port_1_present"] = True
    assert randomized == expected
