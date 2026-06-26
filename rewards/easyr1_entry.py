"""EasyR1 reward entry point (loaded by file path, not as a package submodule)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rewards.compute_reward import REWARD_NAME, REWARD_TYPE, compute_score

__all__ = ["compute_score", "REWARD_NAME", "REWARD_TYPE"]
