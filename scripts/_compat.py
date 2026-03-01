#!/usr/bin/env python
"""Compatibility helpers for forwarding legacy script entrypoints."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Sequence


def forward_to(script_name: str, fixed_args: Sequence[str]) -> None:
    """Run another script in this directory while preserving CLI passthrough."""
    target = Path(__file__).with_name(script_name)
    cmd = [sys.executable, str(target), *fixed_args, *sys.argv[1:]]
    subprocess.run(cmd, check=True)
