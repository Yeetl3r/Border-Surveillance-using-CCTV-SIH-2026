#!/usr/bin/env python
"""Launcher: `python run.py [args]`.

Deliberately NOT named sentinel.py -- a module in the project root with the
same name as the package shadows it on import, which is the exact failure mode
we flagged in the reference prototype's stray `decord.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sentinel.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
