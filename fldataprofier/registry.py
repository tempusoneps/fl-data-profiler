from __future__ import annotations

from fldataprofier.modules.base import ProfilingModule
from fldataprofier.modules.boruta import BorutaRelationshipsModule
from fldataprofier.modules.eda import EdaModule
from fldataprofier.modules.kmean import KMeanRelationshipsModule
from fldataprofier.modules.scipy import ScipyRelationshipsModule
from fldataprofier.modules.shap import ShapRelationshipsModule
from fldataprofier.modules.sklearn import SklearnRelationshipsModule
from fldataprofier.modules.statsmodels import StatsmodelsRelationshipsModule
from fldataprofier.modules.statistics import StatisticsModule
from fldataprofier.modules.xgboost import XGBoostRelationshipsModule


_MODULES: dict[str, ProfilingModule] = {
    "boruta": BorutaRelationshipsModule(),
    "eda": EdaModule(),
    "kmean": KMeanRelationshipsModule(),
    "scipy": ScipyRelationshipsModule(),
    "shap": ShapRelationshipsModule(),
    "sklearn": SklearnRelationshipsModule(),
    "statistics": StatisticsModule(),
    "statsmodels": StatsmodelsRelationshipsModule(),
    "xgboost": XGBoostRelationshipsModule(),
}


def list_modules() -> list[str]:
    return sorted(_MODULES)


def get_module(name: str) -> ProfilingModule:
    try:
        return _MODULES[name]
    except KeyError as exc:
        available = ", ".join(list_modules())
        raise ValueError(f"Unknown module {name!r}. Available modules: {available}") from exc
