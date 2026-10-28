"""Smoke test for the CALVIN + MDT + ShortStop environment.

Checks Python/CUDA/import health *before* attempting a real checkpoint
load or eval run, so a broken environment fails with a clear message
instead of a confusing traceback three layers deep in mdt_evaluate.py.
Does not require a checkpoint or dataset. See docs/CALVIN_SETUP.md.
"""
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so `import shortstop` works unpackaged

CHECKS_RUN = []
FAILURES = []


def check(name):
    def decorator(fn):
        CHECKS_RUN.append(name)
        try:
            detail = fn()
            print(f"[PASS] {name}" + (f" -- {detail}" if detail else ""))
        except Exception as e:
            FAILURES.append((name, str(e)))
            print(f"[FAIL] {name} -- {e}")
        return fn
    return decorator


@check("Python version (expect 3.8.x, per mdt_policy's official README)")
def _():
    v = sys.version_info
    if (v.major, v.minor) != (3, 8):
        raise RuntimeError(f"got Python {platform.python_version()}, expected 3.8.x")
    return platform.python_version()


@check("torch import + version (expect 2.0.1)")
def _():
    import torch
    if not torch.__version__.startswith("2.0.1"):
        raise RuntimeError(f"got torch {torch.__version__}, expected 2.0.1+cu117 -- see docs/CALVIN_SETUP.md")
    return torch.__version__


@check("torchvision import + version (expect 0.15.2, not auto-pinned by mdt_policy's requirements.txt)")
def _():
    import torchvision
    if not torchvision.__version__.startswith("0.15.2"):
        raise RuntimeError(f"got torchvision {torchvision.__version__}, expected 0.15.2")
    return torchvision.__version__


@check("CUDA available + GPU name")
def _():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False -- check LD_LIBRARY_PATH "
            "(docs/CALVIN_SETUP.md section 3, 'CUDA' fix) and `nvidia-smi` inside WSL2"
        )
    return torch.cuda.get_device_name(0)


@check("networkx version (expect 2.8.8, older versions use removed np.int)")
def _():
    import networkx
    if networkx.__version__ != "2.8.8":
        raise RuntimeError(f"got networkx {networkx.__version__}, expected 2.8.8")
    return networkx.__version__


@check("setuptools version (expect 57.5.0, pyhash build needs use_2to3 support)")
def _():
    import setuptools
    if setuptools.__version__ != "57.5.0":
        raise RuntimeError(f"got setuptools {setuptools.__version__}, expected 57.5.0")
    return setuptools.__version__


@check("pyhash import (vendored build from mdt_policy/pyhash-0.9.3)")
def _():
    import pyhash
    pyhash.fnv1_32()
    return "ok"


@check("calvin_env import (mdt_policy's vendored submodule, not a separate mees/calvin clone)")
def _():
    import calvin_env
    return getattr(calvin_env, "__file__", "ok")


@check("tacto import (calvin_env's own vendored submodule)")
def _():
    import tacto
    return getattr(tacto, "__file__", "ok")


@check("mdt (mdt_policy) import + MDTVAgent class")
def _():
    import calvin_env  # noqa: F401 -- before mdt_policy/ (which has a calvin_env/ dir) goes on sys.path
    sys.path.insert(0, str(REPO_ROOT / "mdt_policy"))
    from mdt.models.mdtv_agent import MDTVAgent
    return MDTVAgent.__module__


@check("shortstop import (mdt_policy_client, reach, shield, robot_geometry)")
def _():
    from shortstop.calvin import reach, shield, robot_geometry  # noqa: F401
    from shortstop.calvin.mdt_policy_client import ForwardOnlyPolicy, MockMDTPolicyClient  # noqa: F401
    return "ok"


def main():
    print(f"Repo root: {REPO_ROOT}\n")
    n_fail = len(FAILURES)
    print(f"\n{len(CHECKS_RUN) - n_fail}/{len(CHECKS_RUN)} checks passed.")
    if FAILURES:
        print("\nFailed checks:")
        for name, msg in FAILURES:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print("Environment looks correctly set up for CALVIN + MDT + ShortStop.")


if __name__ == "__main__":
    main()
