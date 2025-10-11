#!/usr/bin/env python3
"""
Inspect env.step() outputs for the same env used by dreamerv3/main.make_env.

Usage (run from the repository root, with required deps installed, e.g. dm_control):
  python tools/inspect_env.py --task dmc_cheetah_run --index 0

Defaults:
  task: dmc_cheetah_run
  index: 0

This script will:
 - instantiate the wrapped environment via dreamerv3.main.make_env
 - print obs_space and act_space keys
 - call env.step(...) twice (first with reset=True, then reset=False)
 - print the keys, types and shapes returned by each step
"""
import argparse
import pprint
import sys
from types import SimpleNamespace

import numpy as np

# Make sure the repo root is importable when running this script from the repo root.
sys.path.insert(0, ".")

import dreamerv3.main as main  # uses make_env from main.py


def build_config(task):
    """Minimal config-like object compatible with main.make_env."""
    cfg = SimpleNamespace()
    cfg.task = task
    # Provide some reasonable defaults for dmc-specific kwargs that make_env will pick up.
    cfg.env = {
        # These keys are looked up by main.make_env as config.env.get(suite, {})
        # If you use a different suite, override as needed via --task or edit here.
        "dmc": {"repeat": 1, "size": (64, 64), "proprio": True, "image": True, "camera": -1}
    }
    cfg.seed = 0
    cfg.logdir = "."
    return cfg


def sample_from_space(space):
    """Create a zero-valued sample that matches elements.Space dtype/shape.
    This uses only dtype and shape; if the space is discrete the zero is a valid index.
    """
    dtype = getattr(space, "dtype", np.float32)
    shape = getattr(space, "shape", ())
    # Ensure dtype is a numpy dtype or python type accepted by np.zeros
    try:
        return np.zeros(shape, dtype=dtype)
    except Exception:
        # Fall back to float32 zeros if dtype/shape are unexpected
        return np.zeros(shape, dtype=np.float32)


def make_action_sample(act_space, reset=False):
    acts = {}
    for k, space in act_space.items():
        if k == "reset":
            # driver uses boolean arrays; here we pass a scalar bool
            acts["reset"] = np.array(reset, dtype=bool)
            continue
        acts[k] = sample_from_space(space)
    return acts


def inspect_env(task, index):
    cfg = build_config(task)
    env = main.make_env(cfg, index)
    print("Environment object:", type(env))
    print("obs_space keys:", list(env.obs_space.keys()))
    print("act_space keys:", list(env.act_space.keys()))
    pp = pprint.PrettyPrinter(indent=2)

    # Step 1: reset=True
    act1 = make_action_sample(env.act_space, reset=True)
    print("\nSTEP 1 (reset=True) action keys/shapes:")
    for k, v in act1.items():
        print(f"  {k}: type={type(v)}, shape={np.shape(v)}")

    try:
        obs1 = env.step(act1)
    except Exception as e:
        print("\nenv.step raised an exception on STEP 1:")
        raise

    print("\nSTEP 1 obs keys:", list(obs1.keys()))
    print("STEP 1 obs summary (type, shape):")
    pp.pprint({k: (type(v), np.shape(v)) for k, v in obs1.items()})

    # Step 2: reset=False
    act2 = make_action_sample(env.act_space, reset=False)
    print("\nSTEP 2 (reset=False) action keys/shapes:")
    for k, v in act2.items():
        print(f"  {k}: type={type(v)}, shape={np.shape(v)}")

    try:
        obs2 = env.step(act2)
    except Exception as e:
        print("\nenv.step raised an exception on STEP 2:")
        raise

    print("\nSTEP 2 obs keys:", list(obs2.keys()))
    print("STEP 2 obs summary (type, shape):")
    pp.pprint({k: (type(v), np.shape(v)) for k, v in obs2.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect dreamerv3 env.step() outputs")
    parser.add_argument("--task", default="dmc_cheetah_run", help="Task string, e.g. dmc_cheetah_run")
    parser.add_argument("--index", type=int, default=0, help="Env index")
    args = parser.parse_args()
    inspect_env(args.task, args.index)