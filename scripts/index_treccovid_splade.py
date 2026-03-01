#!/usr/bin/env python
"""Compatibility wrapper. Use scripts/index_preset.py for new usage."""

from _compat import forward_to


if __name__ == "__main__":
    forward_to("index_preset.py", ["--preset", "trec-covid", "--retrieval", "splade"])
