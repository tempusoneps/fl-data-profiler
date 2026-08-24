from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def get_default_config_path() -> Path:
    """Return the absolute path to the bundled default configuration file."""
    return Path(__file__).parent / "config.default.json"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary on top of base dictionary."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the project configuration, merging defaults with user overrides.

    Resolution priority:
    1. Explicit `config_path` if provided.
    2. `config.json` in the current working directory if present.
    3. `config.default.json` bundled in the package.

    Returns
    -------
    dict[str, Any]
        Complete nested configuration dictionary.
    """
    default_path = get_default_config_path()
    if default_path.exists():
        try:
            cfg = json.loads(default_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    else:
        cfg = {}

    target_user_path: Path | None = None
    if config_path:
        p = Path(config_path)
        if p.exists():
            target_user_path = p
        else:
            raise FileNotFoundError(f"Specified configuration file not found: {p}")
    else:
        cwd_custom = Path("config.json")
        if cwd_custom.exists():
            target_user_path = cwd_custom

    if target_user_path is not None:
        try:
            user_cfg = json.loads(target_user_path.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                cfg = _deep_merge(cfg, user_cfg)
        except Exception as exc:
            raise ValueError(f"Error reading configuration from {target_user_path}: {exc}") from exc

    return cfg


def get_global_config(config_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get global project configuration."""
    cfg = config_dict if config_dict is not None else load_config()
    return dict(cfg.get("global", {}))


def get_prune_config(config_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get feature pruning configuration."""
    cfg = config_dict if config_dict is not None else load_config()
    return dict(cfg.get("prune", {}))


def get_module_config(
    module_name: str, config_dict: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Get module-specific configuration for a profiling module."""
    cfg = config_dict if config_dict is not None else load_config()
    modules_cfg = cfg.get("modules", {})
    return dict(modules_cfg.get(module_name, {}))
