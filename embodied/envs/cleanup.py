
import atexit
import signal
from typing import List

_ENV_REGISTRY: List[object] = []

def register_env(env):
    if env is None:
        return
    _ENV_REGISTRY.append(env)

def _cleanup_envs():
    for e in list(_ENV_REGISTRY):
        try:
            if hasattr(e, "close"):
                e.close()
        except Exception:
            pass
    _ENV_REGISTRY.clear()

atexit.register(_cleanup_envs)

def _exit_handler(signum, frame):
    _cleanup_envs()
    signal.signal(signum, signal.SIG_DFL)
    raise SystemExit()

signal.signal(signal.SIGINT, _exit_handler)
signal.signal(signal.SIGTERM, _exit_handler)