from __future__ import annotations

from fldataprofier.modules.base import ProfilingModule
from fldataprofier.modules.alphalens_analysis import AlphalensAnalysisModule
from fldataprofier.modules.automl_autogluon import AutoGluonRelationshipsModule
from fldataprofier.modules.boruta import BorutaRelationshipsModule
from fldataprofier.modules.eda import EdaModule
from fldataprofier.modules.feature_interactions import FeatureInteractionsModule
from fldataprofier.modules.automl_flaml import FLAMLRelationshipsModule
from fldataprofier.modules.information_coefficient import InformationCoefficientModule
from fldataprofier.modules.kmean import KMeanRelationshipsModule
from fldataprofier.modules.lightgbm import LightGBMModule
from fldataprofier.modules.mrmr import MRMRModule
from fldataprofier.modules.mutual_information import MutualInformationModule
from fldataprofier.modules.permutation_importance_ts import PermutationImportanceTSModule
from fldataprofier.modules.automl_pycaret import PyCaretRelationshipsModule
from fldataprofier.modules.regime_scoring import RegimeScoringModule
from fldataprofier.modules.regularized_linear import RegularizedLinearModule
from fldataprofier.modules.scipy import ScipyRelationshipsModule
from fldataprofier.modules.shap import ShapRelationshipsModule
from fldataprofier.modules.signal_analysis import SignalAnalysisModule
from fldataprofier.modules.sklearn import SklearnRelationshipsModule
from fldataprofier.modules.stability_selection import StabilitySelectionModule
from fldataprofier.modules.statsmodels import StatsmodelsRelationshipsModule
from fldataprofier.modules.statistics import StatisticsModule
from fldataprofier.modules.timeseries_importance import TimeSeriesImportanceModule
from fldataprofier.modules.xgboost import XGBoostRelationshipsModule
from fldataprofier.modules.visual_regions import VisualRegionsModule


_MODULES: dict[str, ProfilingModule] = {
    "alphalens": AlphalensAnalysisModule(),
    "alphalens_analysis": AlphalensAnalysisModule(),
    "autogluon": AutoGluonRelationshipsModule(),
    "boruta": BorutaRelationshipsModule(),
    "eda": EdaModule(),
    "feature_interactions": FeatureInteractionsModule(),
    "flaml": FLAMLRelationshipsModule(),
    "information_coefficient": InformationCoefficientModule(),
    "kmean": KMeanRelationshipsModule(),
    "lightgbm": LightGBMModule(),
    "mrmr": MRMRModule(),
    "mutual_information": MutualInformationModule(),
    "permutation_importance_ts": PermutationImportanceTSModule(),
    "pycaret": PyCaretRelationshipsModule(),
    "regime_scoring": RegimeScoringModule(),
    "regularized_linear": RegularizedLinearModule(),
    "scipy": ScipyRelationshipsModule(),
    "shap": ShapRelationshipsModule(),
    "signal_analysis": SignalAnalysisModule(),
    "sklearn": SklearnRelationshipsModule(),
    "stability_selection": StabilitySelectionModule(),
    "statistics": StatisticsModule(),
    "statsmodels": StatsmodelsRelationshipsModule(),
    "timeseries_importance": TimeSeriesImportanceModule(),
    "visual_regions": VisualRegionsModule(),
    "xgboost": XGBoostRelationshipsModule(),
    "xgboost-numeric": XGBoostRelationshipsModule(),
}


def list_modules() -> list[str]:
    return sorted(_MODULES)


def get_module(name: str) -> ProfilingModule:
    try:
        return _MODULES[name]
    except KeyError as exc:
        available = ", ".join(list_modules())
        raise ValueError(f"Unknown module {name!r}. Available modules: {available}") from exc
