"""Feature/label data profiling CLI."""

from fldataprofiler.config import (
    get_default_config_path,
    get_global_config,
    get_module_config,
    get_prune_config,
    load_config,
)

__all__ = [
    "__version__",
    "load_config",
    "get_default_config_path",
    "get_global_config",
    "get_prune_config",
    "get_module_config",
]

__version__ = "0.1.0"
