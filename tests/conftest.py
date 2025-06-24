import os
import pytest

def pytest_collection_modifyitems(config, items):
    """Skip heavy tests in CI or act environments."""
    if os.environ.get("CI") or os.environ.get("ACT"):
        skip_heavy = pytest.mark.skip(reason="Skipped heavy test in CI/act due to memory limits.")
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
