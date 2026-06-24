"""Shared script bootstrap: make `import tiletalk` work and load config."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiletalk import load_config, abspath, results_dir  # noqa: E402


def common_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=None, help="path to YAML config")
    return p
