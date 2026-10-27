"""Fixtures. Helpers live in helpers.py."""

from __future__ import annotations

import helpers  # noqa: F401  -- sets MSWEA_SILENT_STARTUP before mini-swe-agent loads
import pytest


@pytest.fixture
def corpus(tmp_path):
    """Twelve 70-line python modules, distinct content."""
    for i in range(12):
        (tmp_path / f"mod{i:02d}.py").write_text("".join(
            f"def mod{i:02d}_op_{j}(value):  # operation {j} in module {i:02d}\n"
            f"    return compute(value, {j}) + offset_{i:02d}  # documented behaviour\n"
            for j in range(35)))
    return tmp_path


@pytest.fixture(autouse=True)
def _restore_runner_patches():
    """The runners patch module globals of mini-swe-agent's SWE-bench loop; undo that."""
    import minisweagent.run.benchmarks.swebench as sw

    saved = sw.ProgressTrackingAgent, sw.get_sb_environment
    yield
    sw.ProgressTrackingAgent, sw.get_sb_environment = saved
