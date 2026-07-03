from __future__ import annotations

from fldataprofier.modules.base import ProfilingModule
from fldataprofier.modules.scipy import ScipyRelationshipsModule
from fldataprofier.modules.sklearn import SklearnRelationshipsModule
from fldataprofier.modules.statsmodels import StatsmodelsRelationshipsModule
from fldataprofier.modules.statistics import StatisticsModule


_MODULES: dict[str, ProfilingModule] = {
    "scipy": ScipyRelationshipsModule(),
    "sklearn": SklearnRelationshipsModule(),
    "statistics": StatisticsModule(),
    "statsmodels": StatsmodelsRelationshipsModule(),
}


def list_modules() -> list[str]:
    return sorted(_MODULES)


def get_module(name: str) -> ProfilingModule:
    try:
        return _MODULES[name]
    except KeyError as exc:
        available = ", ".join(list_modules())
        raise ValueError(f"Unknown module {name!r}. Available modules: {available}") from exc
