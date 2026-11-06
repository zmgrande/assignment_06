"""pytest configuration shared by every test in tests/.

Two jobs:

1. Make the modules in code/ importable from the tests (`import packaging`), the
   same way they are importable from the apps that sit beside them.
2. Run every test from the repository root, whatever folder pytest was started
   in. The apps write their JSON output to `data/...` — a path relative to the
   working directory — and the tests look for it there.
"""

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

import sys  # noqa: E402  (the path change belongs with the explanation above)

sys.path.insert(0, os.path.join(REPO_ROOT, "code"))


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
