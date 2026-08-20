from __future__ import annotations

from fldataprofiler.modules.alphalens_analysis import AlphalensAnalysisModule
from fldataprofiler.modules.automl_autogluon import AutoGluonRelationshipsModule
from fldataprofiler.modules.automl_flaml import FLAMLRelationshipsModule
from fldataprofiler.modules.automl_pycaret import PyCaretRelationshipsModule
from fldataprofiler.modules.base import ProfilingModule
from fldataprofiler.modules.boruta import BorutaRelationshipsModule
from fldataprofiler.modules.eda import EdaModule
from fldataprofiler.modules.feature_interactions import FeatureInteractionsModule
from fldataprofiler.modules.information_coefficient import InformationCoefficientModule
from fldataprofiler.modules.kmean import KMeanRelationshipsModule
from fldataprofiler.modules.lightgbm import LightGBMModule
from fldataprofiler.modules.mrmr import MRMRModule
from fldataprofiler.modules.mutual_information import MutualInformationModule
from fldataprofiler.modules.permutation_importance_ts import (
    PermutationImportanceTSModule,
)
from fldataprofiler.modules.probability import ProbabilityModule
from fldataprofiler.modules.probability_2d import Probability2DModule
from fldataprofiler.modules.regime_scoring import RegimeScoringModule
from fldataprofiler.modules.regularized_linear import RegularizedLinearModule
from fldataprofiler.modules.scipy import ScipyRelationshipsModule
from fldataprofiler.modules.shap import ShapRelationshipsModule
from fldataprofiler.modules.signal_analysis import SignalAnalysisModule
from fldataprofiler.modules.sklearn import SklearnRelationshipsModule
from fldataprofiler.modules.stability_selection import StabilitySelectionModule
from fldataprofiler.modules.statistics import StatisticsModule
from fldataprofiler.modules.statsmodels import StatsmodelsRelationshipsModule
from fldataprofiler.modules.timeseries_importance import TimeSeriesImportanceModule
from fldataprofiler.modules.visual_regions import VisualRegionsModule
from fldataprofiler.modules.xgboost import XGBoostRelationshipsModule

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
    "probability": ProbabilityModule(),
    "probability_2d": Probability2DModule(),
    "probability2d": Probability2DModule(),
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
