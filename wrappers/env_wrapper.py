"""Env wrapper helpers that reuse `utils.wrapper` implementations.

Expose a simple `wrap_env_for_sb3` function to make an environment
suitable for SB3 (mostly a passthrough to existing utils wrappers).
"""
from utils.wrapper import ArcadeWrapper, MujocoWrapper, GridWrapper, FetchWrapper

def wrap_env_for_sb3(env, args=None):
    """Return the environment wrapped appropriately for SB3 usage.

    This is intentionally conservative: if the repo already provides a
    wrapper for a specific env type, reuse it; otherwise return `env`.
    """
    # These heuristics are intentionally simple — callers can apply more
    # specialised wrappers if needed.
    try:
        spec_id = env.spec.id if hasattr(env, 'spec') and env.spec is not None else ''
    except Exception:
        spec_id = ''

    if 'antmaze' in spec_id or 'maze' in spec_id:
        return MujocoWrapper(env, vel_threshold=getattr(args, 'vel_threshold', 0.0))

    return env
