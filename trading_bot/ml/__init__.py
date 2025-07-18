"""Machine Learning integration for predictive modeling and feature engineering."""

try:
    from .features import (FeatureEngineer, FundamentalFeatures,
                           TechnicalFeatures)
    from .models import EnsemblePredictor, MLPredictor
    from .training import CrossValidator, ModelTrainer

    ML_AVAILABLE = True
    __all__ = [
        "FeatureEngineer",
        "TechnicalFeatures",
        "FundamentalFeatures",
        "MLPredictor",
        "EnsemblePredictor",
        "ModelTrainer",
        "CrossValidator",
    ]
except ImportError:
    ML_AVAILABLE = False
    __all__ = []
