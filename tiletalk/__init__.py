"""TileTalk: grounding biological text queries in pathology (H&E) images."""
import os

__version__ = "0.1.0"

# Repo root = parent of this package directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str = None) -> dict:
    """Load a YAML config (defaults to configs/xenium_breast.yaml)."""
    import yaml
    if path is None:
        path = os.path.join(REPO_ROOT, "configs", "xenium_breast.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def abspath(rel: str) -> str:
    """Resolve a path relative to the repo root (no-op if already absolute)."""
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def results_dir(cfg: dict, *parts) -> str:
    """Per-dataset results root (defaults to 'results'); joins optional subparts."""
    root = abspath(cfg.get("dataset", {}).get("results_dir", "results"))
    return os.path.join(root, *parts) if parts else root
