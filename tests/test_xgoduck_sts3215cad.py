"""STS3215 duck built from the CAD parts (cad/xgoduck_sts3215/build_sts_duck.py)."""

import math

import mujoco
import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.xgoduck_constants import _bom_json, locked_joints

LAYOUTS = {"leg8": 11, "leg6a": 9, "leg6b": 9}
HOME = {"hip_roll": 0.0873, "hip_pitch": math.radians(24), "knee": 0.0049, "ankle": math.radians(24),
        "neck_pitch": 0.3491, "head_pitch": 0.3491}


def _model(layout, kind="Velocity"):
    cfg = load_env_cfg(f"Mjlab-{kind}-Flat-XgoDuck-STS-CAD-C001-{layout.capitalize()}")
    return cfg, cfg.scene.entities["robot"].spec_fn().compile()


def _home(model):
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    for j in range(model.njnt):
        name = model.joint(j).name
        for key, value in HOME.items():
            if name.endswith(key):
                left_like = name.startswith("left") and key in ("hip_roll", "hip_pitch", "knee")
                right_ankle = name.startswith("right") and key == "ankle"
                data.qpos[model.jnt_qposadr[j]] = -value if (left_like or right_ankle) else value
    mujoco.mj_forward(model, data)
    return data


@pytest.mark.parametrize("layout", LAYOUTS)
def test_servo_count_and_mass(layout):
    bom = _bom_json(f"sts3215cad_{layout}")
    assert bom["servo_count"] == LAYOUTS[layout]
    assert 14 - len(locked_joints(f"sts3215cad_{layout}")) == LAYOUTS[layout]
    _, model = _model(layout)
    assert math.isclose(model.body_mass.sum(), bom["total_mass_kg"], rel_tol=1e-6)
    assert 1.1 < model.body_mass.sum() < 1.5


def test_leg_is_the_cad_leg_and_stands_vertical_at_home():
    _, model = _model("leg8")
    data = _home(model)
    hip = data.xanchor[model.joint("left_hip_pitch").id]
    knee = data.xanchor[model.joint("left_knee").id]
    ankle = data.xanchor[model.joint("left_ankle").id]
    assert math.isclose(math.dist(hip, knee), 0.056, abs_tol=1e-4)
    assert math.isclose(math.dist(knee, ankle), 0.056, abs_tol=1e-4)
    # the three leg axes are parallel to y: the leg plane is vertical at HOME
    for name in ("left_hip_pitch", "left_knee", "left_ankle"):
        assert abs(abs(data.xaxis[model.joint(name).id][1]) - 1.0) < 1e-4
    assert abs(hip[1] - ankle[1]) < 1e-4


def test_feet_are_flat_and_the_centre_of_mass_is_over_them():
    _, model = _model("leg8")
    data = _home(model)
    lows, xs = [], []
    for side in ("left", "right"):
        g = model.geom(f"{side}_foot_collision").id
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        sole = w[w[:, 2] < w[:, 2].min() + 1e-4]
        assert sole[:, 0].max() - sole[:, 0].min() > 0.09      # 95 mm long, all at one height = flat
        assert sole[:, 1].max() - sole[:, 1].min() > 0.065     # 68 mm wide
        lows.append(w[:, 2].min()); xs += [sole[:, 0].min(), sole[:, 0].max()]
        site = data.site_xpos[model.site(f"{side}_foot").id]
        assert abs(site[2] - w[:, 2].min()) < 1e-3
    assert abs(lows[0] - lows[1]) < 1e-4
    com_x = data.subtree_com[1][0]
    assert min(xs) + 0.035 < com_x < max(xs) - 0.035          # at least 35 mm from heel and toe edge


def test_joint_ranges_come_from_the_design():
    _, model = _model("leg8")
    lo, hi = model.joint("left_hip_roll").range
    assert math.isclose(lo, math.radians(-15), abs_tol=1e-6) and math.isclose(hi, math.radians(15), abs_tol=1e-6)
    lo, hi = model.joint("left_knee").range
    assert math.isclose(hi, math.radians(30), abs_tol=1e-6)
    lo, hi = model.joint("left_hip_yaw").range
    assert hi - lo < 0.01


def test_standup_task_loads_with_shifted_heights():
    dz = _bom_json("sts3215cad_leg8")["stand_dz_m"]
    assert 0.02 < dz < 0.035
    new, model = _model("leg8", "StandUp")
    old = load_env_cfg("Mjlab-StandUp-Flat-XgoDuck")
    assert math.isclose(new.rewards["height_stand"].params["target_height"],
                        old.rewards["height_stand"].params["target_height"] + dz, abs_tol=1e-6)
    assert model.nbody > 10
