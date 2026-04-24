STAGES = ("approach", "coarse_align", "fine_align", "insert")

ORACLE_DEFAULTS = {
    "oracle_speed_scale": 0.5,
    "oracle_command_period": 0.05,
    "oracle_cartesian_stiffness": [70.0, 70.0, 70.0, 35.0, 35.0, 35.0],
    "oracle_cartesian_damping": [60.0, 60.0, 60.0, 25.0, 25.0, 25.0],
    "oracle_approach_steps": 100,
    "oracle_approach_step_meters": 0.004,
    "oracle_approach_z_offset": 0.35,
    "oracle_coarse_align_steps": 100,
    "oracle_hover_z_offset": 0.01,
    "oracle_fine_align_steps": 70,
    "oracle_fine_align_hold_steps": 30,
    "oracle_alignment_integral_gain": 0.12,
    "oracle_alignment_integrator_limit": 0.05,
    "oracle_insert_step_meters": 0.00035,
    "oracle_insert_end_z_offset": -0.015,
    "oracle_final_settle_steps": 50,
}


def declare_oracle_parameters(node):
    for name, value in ORACLE_DEFAULTS.items():
        if not node.has_parameter(name):
            node.declare_parameter(name, value)
