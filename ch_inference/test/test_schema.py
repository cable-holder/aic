from types import SimpleNamespace

import numpy as np

from ch_inference.rotations import quaternion_xyzw_from_rpy, rpy_from_quaternion_xyzw
from ch_inference.schema import image_from_msg, state_vector, task_prompt


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_rpy_roundtrip():
    rpy = np.array([0.2, -0.3, 1.1], dtype=np.float32)
    got = rpy_from_quaternion_xyzw(quaternion_xyzw_from_rpy(rpy))
    np.testing.assert_allclose(got, rpy, atol=1e-6)


def test_state_vector_uses_rpy_scheme():
    obs = ns(
        controller_state=ns(
            tcp_pose=ns(
                position=ns(x=1, y=2, z=3),
                orientation=ns(x=0, y=0, z=0, w=1),
            ),
            tcp_velocity=ns(
                linear=ns(x=4, y=5, z=6),
                angular=ns(x=7, y=8, z=9),
            ),
            tcp_error=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ),
        joint_states=ns(position=list(range(7))),
    )
    state = state_vector(obs)
    assert state.shape == (25,)
    np.testing.assert_allclose(state[:6], [1, 2, 3, 0, 0, 0])
    np.testing.assert_allclose(state[18:], list(range(7)))


def test_image_resize():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    msg = ns(height=2, width=3, data=image.tobytes())
    assert image_from_msg(msg, (4, 6, 3)).shape == (4, 6, 3)


def test_task_prompt():
    task = ns(
        plug_type="sfp",
        plug_name="sfp_tip",
        port_type="sfp",
        target_module_name="nic_card_0",
        port_name="sfp_port_0",
    )
    assert task_prompt(task) == (
        "insert sfp plug sfp_tip into sfp port nic_card_0/sfp_port_0"
    )
