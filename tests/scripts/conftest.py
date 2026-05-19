# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""conftest for scripts tests — adds external/saber/scripts to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `scripts/` importable so `from scripts.migrate_scoring_yaml import …` works.
_scripts_parent = Path(__file__).resolve().parents[2]  # external/saber
if str(_scripts_parent) not in sys.path:
    sys.path.insert(0, str(_scripts_parent))
