"""Machine learning models for trading predictions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

# Try importing ML libraries, fallback gracefully
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ..core.logging import TradingLogger


class BaseMLModel(ABC):
    """Base class for ML models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.logger = TradingLogger(f"ml_model_{name}")

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""

    def save_model(self, filepath: str) -> bool:
        """Save model to file."""
        try:
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
                "name": self.name,
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            self.logger.logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load model from file."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.is_trained = model_data["is_trained"]
            self.name = model_data["name"]
            return True
        except Exception as e:
            self.logger.logger.error(f"Error loading model: {e}")
            return False


class LinearModel(BaseMLModel):
    """Linear regression model."""

    def __init__(self, model_type: str = "linear"):
        super().__init__(f"linear_{model_type}")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")

        self.scaler = StandardScaler()

        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "ridge":
            self.model = Ridge(alpha=1.0)
        elif model_type == "lasso":
            self.model = Lasso(alpha=0.1)
        else:
            raise ValueError(f"Unknown linear model type: {model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train linear model."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            metrics = {
                "train_mse": mean_squared_error(y_train, y_pred_train),
                "test_mse": mean_squared_error(y_test, y_pred_test),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
            }

            self.is_trained = True
            self.logger.logger.info(
                f"Model trained - Test R2: {metrics['test_r2']:.4f}"
            )

            return metrics

        except Exception as e:
            self.logger.logger.error(f"Training failed: {e}")
            return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RandomForestModel(BaseMLModel):
    """Random Forest model."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("random_forest")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")

        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train Random Forest model."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            metrics = {
                "train_mse": mean_squared_error(y_train, y_pred_train),
                "test_mse": mean_squared_error(y_test, y_pred_test),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
            }

            self.is_trained = True
            self.logger.logger.info(
                f"Random Forest trained - Test R2: {metrics['test_r2']:.4f}"
            )

            return metrics

        except Exception as e:
            self.logger.logger.error(f"Training failed: {e}")
            return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        if self.is_trained and hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None


class XGBoostModel(BaseMLModel):
    """XGBoost model."""

    def __init__(
        self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1
    ):
        super().__init__("xgboost")

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")

        self.scaler = StandardScaler()
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train XGBoost model."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            # Evaluate
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            metrics = {
                "train_mse": mean_squared_error(y_train, y_pred_train),
                "test_mse": mean_squared_error(y_test, y_pred_test),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
            }

            self.is_trained = True
            self.logger.logger.info(
                f"XGBoost trained - Test R2: {metrics['test_r2']:.4f}"
            )

            return metrics

        except Exception as e:
            self.logger.logger.error(f"Training failed: {e}")
            return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class EnsemblePredictor:
    """Ensemble of multiple ML models."""

    def __init__(
        self, models: List[BaseMLModel], weights: Optional[List[float]] = None
    ):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.logger = TradingLogger("ensemble_predictor")

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

    def train_all(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Train all models in the ensemble."""
        results = {}

        for model in self.models:
            try:
                metrics = model.train(X, y)
                results[model.name] = metrics
                self.logger.logger.info(f"Trained {model.name}")
            except Exception as e:
                self.logger.logger.error(f"Failed to train {model.name}: {e}")
                results[model.name] = {}

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []

        for model, weight in zip(self.models, self.weights):
            if model.is_trained:
                try:
                    pred = model.predict(X)
                    predictions.append(pred * weight)
                except Exception as e:
                    self.logger.logger.error(f"Prediction failed for {model.name}: {e}")

        if not predictions:
            raise ValueError("No trained models available for prediction")

        # Weighted average
        ensemble_prediction = np.sum(predictions, axis=0)
        return ensemble_prediction

    def get_model_status(self) -> Dict[str, bool]:
        """Get training status of all models."""
        return {model.name: model.is_trained for model in self.models}


class MLPredictor:
    """Main ML predictor class."""

    def __init__(self):
        self.logger = TradingLogger("ml_predictor")
        self.models: Dict[str, BaseMLModel] = {}
        self.ensemble: Optional[EnsemblePredictor] = None

        # Initialize available models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize available ML models."""
        try:
            if SKLEARN_AVAILABLE:
                self.models["linear"] = LinearModel("linear")
                self.models["ridge"] = LinearModel("ridge")
                self.models["random_forest"] = RandomForestModel(
                    n_estimators=50, max_depth=8
                )

            if XGBOOST_AVAILABLE:
                self.models["xgboost"] = XGBoostModel(n_estimators=50, max_depth=6)

            self.logger.logger.info(f"Initialized {len(self.models)} ML models")

            # Create ensemble
            if self.models:
                model_list = list(self.models.values())
                self.ensemble = EnsemblePredictor(model_list)

        except Exception as e:
            self.logger.logger.error(f"Error initializing models: {e}")

    def train_models(
        self, X: np.ndarray, y: np.ndarray, models: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Train specified models or all available models."""

        if X.shape[0] < 50:
            self.logger.logger.warning(
                f"Insufficient data for training: {X.shape[0]} samples"
            )
            return {}

        models_to_train = models or list(self.models.keys())
        results = {}

        for model_name in models_to_train:
            if model_name in self.models:
                try:
                    metrics = self.models[model_name].train(X, y)
                    results[model_name] = metrics
                    self.logger.logger.info(f"Trained {model_name}")
                except Exception as e:
                    self.logger.logger.error(f"Training failed for {model_name}: {e}")
                    results[model_name] = {}

        return results

    def predict(
        self, X: np.ndarray, model_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Make predictions using specified model or ensemble."""
        predictions = {}

        if model_name:
            # Single model prediction
            if model_name in self.models and self.models[model_name].is_trained:
                try:
                    pred = self.models[model_name].predict(X)
                    predictions[model_name] = pred
                except Exception as e:
                    self.logger.logger.error(f"Prediction failed for {model_name}: {e}")
        else:
            # All model predictions
            for name, model in self.models.items():
                if model.is_trained:
                    try:
                        pred = model.predict(X)
                        predictions[name] = pred
                    except Exception as e:
                        self.logger.logger.error(f"Prediction failed for {name}: {e}")

            # Ensemble prediction
            if self.ensemble:
                try:
                    ensemble_pred = self.ensemble.predict(X)
                    predictions["ensemble"] = ensemble_pred
                except Exception as e:
                    self.logger.logger.error(f"Ensemble prediction failed: {e}")

        return predictions

    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary of all models."""
        performance = {}

        for name, model in self.models.items():
            performance[name] = {
                "is_trained": model.is_trained,
                "model_type": type(model).__name__,
            }

            # Add feature importance if available
            if hasattr(model, "get_feature_importance") and model.is_trained:
                importance = model.get_feature_importance()
                if importance is not None:
                    performance[name]["feature_importance"] = importance.tolist()

        return performance

    def save_models(self, directory: str) -> Dict[str, bool]:
        """Save all trained models."""
        results = {}

        for name, model in self.models.items():
            if model.is_trained:
                filepath = f"{directory}/{name}_model.joblib"
                success = model.save_model(filepath)
                results[name] = success

        return results

    def load_models(self, directory: str) -> Dict[str, bool]:
        """Load models from directory."""
        results = {}

        for name in self.models.keys():
            filepath = f"{directory}/{name}_model.joblib"
            try:
                success = self.models[name].load_model(filepath)
                results[name] = success
            except Exception as e:
                self.logger.logger.error(f"Failed to load {name}: {e}")
                results[name] = False

        return results
