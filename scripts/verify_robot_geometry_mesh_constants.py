"""Re-derive robot_geometry's mesh/URDF/yaml constants (LINK_RADIUS,
FINGER_RADIUS, GRIPPER_TIP_RADIUS, FLANGE_OFFSET, HAND_YAW_OFFSET,
FINGER_JOINT_Z_OFFSET, GRIPPER_TIP_OFFSET, JOINT_LIMITS) from the files
CALVIN loads (panda_longer_finger.urdf and its meshes), and print PASS/FAIL
per constant; exits nonzero on a mismatch. Mesh radius = max vertex
distance from the mesh's principal axis, rounded up to the cm. Static files
only, runs anywhere:

    python scripts/verify_robot_geometry_mesh_constants.py
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shortstop.calvin.robot_geometry import (  # noqa: E402
    FINGER_JOINT_Z_OFFSET, FINGER_RADIUS, FLANGE_OFFSET, GRIPPER_TIP_OFFSET,
    GRIPPER_TIP_RADIUS, HAND_YAW_OFFSET, JOINT_LIMITS, LINK_RADIUS,
)

FRANKA_PANDA_DIR = REPO_ROOT / "mdt_policy" / "calvin_env" / "data" / "franka_panda"
MESH_DIR = FRANKA_PANDA_DIR / "meshes" / "collision"
URDF_PATH = FRANKA_PANDA_DIR / "panda_longer_finger.urdf"
PANDA_YAML_PATH = REPO_ROOT / "mdt_policy" / "calvin_env" / "conf" / "robot" / "panda.yaml"

_FAILURES = []


def _check(label, real, expected, tol=1e-6):
    ok = abs(real - expected) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: real={real:.5f}  code={expected:.5f}")
    if not ok:
        _FAILURES.append(label)


def _load_obj_vertices(path):
    vertices = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=float)


def _max_perpendicular_radius(vertices):
    centered = vertices - vertices.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    proj = centered @ axis
    perp = centered - np.outer(proj, axis)
    return float(np.linalg.norm(perp, axis=1).max())


def _round_up_to_cm(value):
    return float(np.ceil(value * 100.0) / 100.0)


def _origin_xyz(root, joint_name):
    origin = root.find(f".//joint[@name='{joint_name}']/origin")
    return np.asarray([float(v) for v in origin.get("xyz").split()], dtype=float)


def _origin_rpy(root, joint_name):
    origin = root.find(f".//joint[@name='{joint_name}']/origin")
    return np.asarray([float(v) for v in origin.get("rpy").split()], dtype=float)


def _joint_limit_upper(root, joint_name):
    limit = root.find(f".//joint[@name='{joint_name}']/limit")
    return float(limit.get("upper"))


def _panda_yaml_joint_limits(path):
    """Manual line parse instead of a real YAML library -- keeps this
    script dependency-free (numpy + stdlib only), and panda.yaml's own
    lower_joint_limits/upper_joint_limits lines are simple flat
    `key: [a, b, c, ...]` lists, not nested structure."""
    text = path.read_text(encoding="utf-8")
    lower = [float(v) for v in re.search(r"lower_joint_limits:\s*\[([^\]]+)\]", text).group(1).split(",")]
    upper = [float(v) for v in re.search(r"upper_joint_limits:\s*\[([^\]]+)\]", text).group(1).split(",")]
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def main():
    print(f"[mesh radii] link0..link7 (from {MESH_DIR}) vs LINK_RADIUS:")
    for i in range(len(LINK_RADIUS)):
        vertices = _load_obj_vertices(MESH_DIR / f"link{i}.obj")
        measured = _round_up_to_cm(_max_perpendicular_radius(vertices))
        _check(f"LINK_RADIUS[{i}]", measured, LINK_RADIUS[i])

    print("[mesh radii] longer_finger_v2.obj (the REAL mesh panda_longer_finger.urdf loads) vs FINGER_RADIUS:")
    finger_vertices = _load_obj_vertices(MESH_DIR / "longer_finger_v2.obj")
    raw_finger_radius = _max_perpendicular_radius(finger_vertices)
    _check("FINGER_RADIUS", _round_up_to_cm(raw_finger_radius), FINGER_RADIUS)

    if not URDF_PATH.exists():
        print(f"[urdf] SKIPPED -- {URDF_PATH} not found")
    else:
        root = ET.parse(URDF_PATH).getroot()
        print(f"[urdf] {URDF_PATH.name}'s own <origin>/<limit> values vs robot_geometry.py's constants:")
        _check("FLANGE_OFFSET[2]", _origin_xyz(root, "panda_joint8")[2], FLANGE_OFFSET[2])
        _check("HAND_YAW_OFFSET", _origin_rpy(root, "panda_hand_joint")[2], HAND_YAW_OFFSET)
        _check("GRIPPER_TIP_OFFSET", _origin_xyz(root, "tcp_joint")[2], GRIPPER_TIP_OFFSET)
        _check("FINGER_JOINT_Z_OFFSET", _origin_xyz(root, "panda_finger_joint1")[2], FINGER_JOINT_Z_OFFSET)

        real_max_one_sided_spread = _joint_limit_upper(root, "panda_finger_joint1")
        expected_gripper_tip_radius = _round_up_to_cm(raw_finger_radius) + real_max_one_sided_spread
        print(
            f"  [check] GRIPPER_TIP_RADIUS derivation: round_up(finger_radius)={_round_up_to_cm(raw_finger_radius):.5f} "
            f"+ real_max_one_sided_spread={real_max_one_sided_spread:.5f} = {expected_gripper_tip_radius:.5f}"
        )
        _check("GRIPPER_TIP_RADIUS", expected_gripper_tip_radius, GRIPPER_TIP_RADIUS)

    if not PANDA_YAML_PATH.exists():
        print(f"[yaml] SKIPPED -- {PANDA_YAML_PATH} not found")
    else:
        lower, upper = _panda_yaml_joint_limits(PANDA_YAML_PATH)
        print(f"[yaml] {PANDA_YAML_PATH.name}'s own lower/upper_joint_limits vs JOINT_LIMITS:")
        for i in range(len(JOINT_LIMITS)):
            _check(f"JOINT_LIMITS[{i}][0] (lower)", lower[i], JOINT_LIMITS[i][0])
            _check(f"JOINT_LIMITS[{i}][1] (upper)", upper[i], JOINT_LIMITS[i][1])

    if _FAILURES:
        print(f"\n[result] {len(_FAILURES)} constant(s) disagree with the real source files: {_FAILURES}")
        sys.exit(1)
    print("\n[result] every constant checked here matches its real source file exactly.")


if __name__ == "__main__":
    main()
