"""Model training and cross-validation utilities."""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core.logging import TradingLogger
from .models import BaseMLModel, MLPredictor


class CrossValidator:
    """Time series cross-validation for trading models."""

    def __init__(self, n_splits: int = 5, test_size: int = 50):
        self.n_splits = n_splits
        self.test_size = test_size
        self.logger = TradingLogger("cross_validator")

        if not SKLEARN_AVAILABLE:
            self.logger.logger.warning(
                "scikit-learn not available - limited functionality"
            )

    def validate_model(
        self, model: BaseMLModel, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Perform time series cross-validation."""

        if not SKLEARN_AVAILABLE:
            return self._simple_validation(model, X, y)

        try:
            # Time series split
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)

            scores = {"mse_scores": [], "r2_scores": [], "mae_scores": []}

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train model on fold
                model_copy = self._copy_model(model)
                model_copy.train(X_train, y_train)

                # Predict and evaluate
                y_pred = model_copy.predict(X_test)

                scores["mse_scores"].append(mean_squared_error(y_test, y_pred))
                scores["r2_scores"].append(r2_score(y_test, y_pred))
                scores["mae_scores"].append(mean_absolute_error(y_test, y_pred))

            # Calculate statistics
            results = {}
            for metric, values in scores.items():
                results[f"{metric[:-7]}_mean"] = np.mean(values)
                results[f"{metric[:-7]}_std"] = np.std(values)

            self.logger.logger.info(
                f"CV completed - R2 mean: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}"
            )

            return results

        except Exception as e:
            self.logger.logger.error(f"Cross-validation failed: {e}")
            return {}

    def _simple_validation(
        self, model: BaseMLModel, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Simple train/test split validation when sklearn not available."""
        try:
            # Simple 80/20 split
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Train and evaluate
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)

            # Simple metrics calculation
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))

            # R2 calculation
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return {
                "mse_mean": mse,
                "mae_mean": mae,
                "r2_mean": r2,
                "mse_std": 0.0,
                "mae_std": 0.0,
                "r2_std": 0.0,
            }

        except Exception as e:
            self.logger.logger.error(f"Simple validation failed: {e}")
            return {}

    def _copy_model(self, model: BaseMLModel):
        """Create a copy of the model for cross-validation."""
        # This is a simplified copy - in practice would need proper model cloning
        return type(model).__new__(type(model))


class ModelTrainer:
    """Advanced model training with hyperparameter optimization."""

    def __init__(self):
        self.logger = TradingLogger("model_trainer")
        self.training_history: Dict[str, List[Dict]] = defaultdict(list)

    def train_with_validation(
        self,
        predictor: MLPredictor,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train models with validation."""

        if len(X) < 100:
            self.logger.logger.warning(
                f"Insufficient data for training: {len(X)} samples"
            )
            return {}

        # Split data
        split_idx = int((1 - validation_split) * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        results = {}

        # Train all models
        training_results = predictor.train_models(X_train, y_train)

        # Validate all models
        for model_name, model in predictor.models.items():
            if model.is_trained:
                try:
                    y_pred = model.predict(X_val)

                    # Calculate validation metrics
                    val_metrics = self._calculate_metrics(y_val, y_pred)

                    results[model_name] = {
                        "training_metrics": training_results.get(model_name, {}),
                        "validation_metrics": val_metrics,
                        "model_type": type(model).__name__,
                    }

                    # Store in history
                    self.training_history[model_name].append(
                        {
                            "timestamp": datetime.now(),
                            "train_samples": len(X_train),
                            "val_samples": len(X_val),
                            "validation_r2": val_metrics.get("r2", 0),
                        }
                    )

                except Exception as e:
                    self.logger.logger.error(f"Validation failed for {model_name}: {e}")

        # Rank models by validation performance
        model_ranking = self._rank_models(results)
        results["model_ranking"] = model_ranking

        return results

    def hyperparameter_search(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Simple grid search for hyperparameters."""

        if not SKLEARN_AVAILABLE:
            self.logger.logger.warning("Hyperparameter search requires scikit-learn")
            return {}

        best_score = -np.inf
        best_params = {}
        results = []

        # Simple grid search implementation
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        def _generate_combinations(values, current_combo=[]):
            if not values:
                yield current_combo
            else:
                for value in values[0]:
                    yield from _generate_combinations(
                        values[1:], current_combo + [value]
                    )

        for combination in _generate_combinations(param_values):
            params = dict(zip(param_names, combination))

            try:
                # Create model with these parameters
                if model_type == "random_forest":
                    from .models import RandomForestModel

                    model = RandomForestModel(**params)
                elif model_type == "xgboost":
                    from .models import XGBoostModel

                    model = XGBoostModel(**params)
                else:
                    continue

                # Cross-validate
                cv = CrossValidator(n_splits=3)
                cv_results = cv.validate_model(model, X, y)

                score = cv_results.get("r2_mean", -np.inf)

                result = {"params": params, "score": score, "cv_results": cv_results}
                results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = params

                self.logger.logger.info(f"Tested {params} - Score: {score:.4f}")

            except Exception as e:
                self.logger.logger.error(f"Error testing params {params}: {e}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results,
        }

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate validation metrics."""
        try:
            metrics = {}

            # MSE
            metrics["mse"] = np.mean((y_true - y_pred) ** 2)

            # MAE
            metrics["mae"] = np.mean(np.abs(y_true - y_pred))

            # R2
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # RMSE
            metrics["rmse"] = np.sqrt(metrics["mse"])

            # Direction accuracy (for returns prediction)
            direction_correct = np.sum((y_true > 0) == (y_pred > 0))
            metrics["direction_accuracy"] = direction_correct / len(y_true)

            return metrics

        except Exception as e:
            self.logger.logger.error(f"Error calculating metrics: {e}")
            return {}

    def _rank_models(self, results: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Rank models by validation performance."""
        rankings = []

        for model_name, result in results.items():
            if isinstance(result, dict) and "validation_metrics" in result:
                r2_score = result["validation_metrics"].get("r2", -np.inf)
                rankings.append((model_name, r2_score))

        # Sort by R2 score descending
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def get_training_history(
        self, model_name: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """Get training history for models."""
        if model_name:
            return {model_name: self.training_history.get(model_name, [])}
        return dict(self.training_history)

    def suggest_model_improvements(self, results: Dict[str, Any]) -> List[str]:
        """Suggest improvements based on training results."""
        suggestions = []

        for model_name, result in results.items():
            if isinstance(result, dict) and "validation_metrics" in result:
                val_metrics = result["validation_metrics"]
                train_metrics = result.get("training_metrics", {})

                # Check for overfitting
                train_r2 = train_metrics.get("train_r2", 0)
                val_r2 = val_metrics.get("r2", 0)

                if train_r2 - val_r2 > 0.2:
                    suggestions.append(
                        f"{model_name}: Possible overfitting - consider regularization"
                    )

                # Check for underfitting
                if val_r2 < 0.1:
                    suggestions.append(
                        f"{model_name}: Low performance - consider feature engineering"
                    )

                # Check direction accuracy
                direction_acc = val_metrics.get("direction_accuracy", 0.5)
                if direction_acc < 0.52:
                    suggestions.append(
                        f"{model_name}: Poor directional accuracy - review features"
                    )

        return suggestions

    def export_training_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Export comprehensive training report."""

        report = {
            "timestamp": datetime.now().isoformat(),
            "model_results": results,
            "model_ranking": results.get("model_ranking", []),
            "training_summary": {},
            "recommendations": self.suggest_model_improvements(results),
        }

        # Summary statistics
        if results:
            val_r2_scores = []
            for model_name, result in results.items():
                if isinstance(result, dict) and "validation_metrics" in result:
                    val_r2_scores.append(result["validation_metrics"].get("r2", 0))

            if val_r2_scores:
                report["training_summary"] = {
                    "best_r2": max(val_r2_scores),
                    "average_r2": np.mean(val_r2_scores),
                    "worst_r2": min(val_r2_scores),
                    "models_trained": len(val_r2_scores),
                }

        return report
