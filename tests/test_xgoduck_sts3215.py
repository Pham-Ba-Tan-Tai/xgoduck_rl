"""All-STS3215 XgoDuck variants: servo swap, locked joints, command ranges."""

import math

import mujoco
import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.xgoduck_constants import (
    _bom_json,
    get_walk_spec,
    locked_joints,
)

LAYOUTS = {
    "leg8": {"left_hip_yaw", "right_hip_yaw", "head_roll"},
    "leg6a": {"left_hip_yaw", "right_hip_yaw", "head_roll", "left_hip_roll", "right_hip_roll"},
    "leg6b": {"left_hip_yaw", "right_hip_yaw", "head_roll", "left_ankle", "right_ankle"},
}


def _task(kind, model, layout):
    return f"Mjlab-{kind}-Flat-XgoDuck-STS-{model.upper()}-{layout.capitalize()}"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_locked_joints_and_servo_count(layout):
    assert set(locked_joints(f"sts3215_{layout}")) == LAYOUTS[layout]
    assert _bom_json(f"sts3215_{layout}")["servo_count"] == 14 - len(LAYOUTS[layout])


@pytest.mark.parametrize("layout", LAYOUTS)
def test_locked_joints_are_pinned_and_legs_are_longer(layout):
    cfg = load_env_cfg(_task("Velocity", "c001", layout))
    model = cfg.scene.entities["robot"].spec_fn().compile()
    stock = get_walk_spec().compile()
    for name in LAYOUTS[layout]:
        lo, hi = model.joint(name).range
        assert hi - lo < 0.01, f"{name} is not pinned"
    lo, hi = model.joint("left_knee").range
    assert hi - lo > 1.0, "a servo joint lost its range"
    thigh = float((model.body("leg").pos ** 2).sum() ** 0.5)
    stock_thigh = float((stock.body("leg").pos ** 2).sum() ** 0.5)
    assert math.isclose(thigh / stock_thigh, 1.30, abs_tol=0.01)
    assert model.body_mass.sum() > 1.0  # heavier than the 0.8 kg stock model


@pytest.mark.parametrize("layout", LAYOUTS)
def test_actuator_groups(layout):
    cfg = load_env_cfg(_task("Velocity", "c001", layout))
    servo, none = cfg.scene.entities["robot"].articulation.actuators
    assert servo.json_path.endswith("sts3215_c001_datasheet.json")
    assert none.json_path.endswith("no_servo.json")
    assert {e.strip("^$") for e in none.target_names_expr} == LAYOUTS[layout]


def test_walk_commands():
    slow = load_env_cfg(_task("Velocity", "c001", "leg8")).commands["twist"]
    fast = load_env_cfg(_task("Velocity", "c046", "leg8")).commands["twist"]
    no_roll = load_env_cfg(_task("Velocity", "c001", "leg6a")).commands["twist"]
    assert slow.ranges.lin_vel_x == (-0.2, 0.2)
    assert fast.ranges.lin_vel_x == (-0.4, 0.4)
    assert no_roll.ranges.lin_vel_y == (0.0, 0.0)
    assert slow.rel_turn_in_place_envs == 0.0


@pytest.mark.parametrize("kind", ("Velocity", "StandUp"))
def test_head_roll_is_never_commanded(kind):
    cfg = load_env_cfg(_task(kind, "c001", "leg8"))
    assert cfg.commands["head_pose"].ranges[3] == (0.0, 0.0)
    if "head_pose_range" in cfg.curriculum:
        for stage in cfg.curriculum["head_pose_range"].params["range_stages"]:
            assert stage["ranges"][3] == (0.0, 0.0)


def test_standup_heights_follow_the_longer_legs():
    dz = _bom_json("sts3215_leg8")["stand_dz_m"]
    assert 0.015 < dz < 0.03
    new = load_env_cfg(_task("StandUp", "c001", "leg8"))
    old = load_env_cfg("Mjlab-StandUp-Flat-XgoDuck")
    for name in ("height_stand", "standing_composite"):
        assert math.isclose(
            new.rewards[name].params["target_height"],
            old.rewards[name].params["target_height"] + dz,
            abs_tol=1e-6,
        )
