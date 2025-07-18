"""Feature engineering for machine learning models."""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logging import TradingLogger
from ..core.models import MarketData


class TechnicalFeatures:
    """Technical indicator feature engineering."""

    def __init__(self):
        self.logger = TradingLogger("technical_features")

    def calculate_returns(
        self, prices: List[float], periods: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, List[float]]:
        """Calculate returns over multiple periods."""
        features = {}

        for period in periods:
            returns = []
            for i in range(len(prices)):
                if i >= period:
                    ret = (prices[i] - prices[i - period]) / prices[i - period]
                    returns.append(ret)
                else:
                    returns.append(0.0)
            features[f"return_{period}d"] = returns

        return features

    def calculate_volatility_features(
        self, returns: List[float], windows: List[int] = [5, 10, 20]
    ) -> Dict[str, List[float]]:
        """Calculate rolling volatility features."""
        features = {}

        for window in windows:
            volatilities = []
            for i in range(len(returns)):
                if i >= window - 1:
                    window_returns = returns[i - window + 1 : i + 1]
                    vol = np.std(window_returns) * np.sqrt(252)  # Annualized
                    volatilities.append(vol)
                else:
                    volatilities.append(0.0)
            features[f"volatility_{window}d"] = volatilities

        return features

    def calculate_momentum_features(
        self, prices: List[float]
    ) -> Dict[str, List[float]]:
        """Calculate momentum-based features."""
        features = {}

        # RSI
        rsi = self._calculate_rsi(prices)
        features["rsi"] = rsi

        # Price ratios
        sma_20 = self._calculate_sma(prices, 20)
        features["price_to_sma20"] = [
            p / sma if sma > 0 else 1.0 for p, sma in zip(prices, sma_20)
        ]

        # Price position in recent range
        price_percentile = self._calculate_price_percentile(prices, 20)
        features["price_percentile_20d"] = price_percentile

        return features

    def calculate_volume_features(
        self, volumes: List[float], prices: List[float]
    ) -> Dict[str, List[float]]:
        """Calculate volume-based features."""
        features = {}

        # Volume moving averages
        vol_sma_10 = self._calculate_sma(volumes, 10)
        features["volume_ratio_10d"] = [
            v / sma if sma > 0 else 1.0 for v, sma in zip(volumes, vol_sma_10)
        ]

        # Price-volume correlation
        corr_5d = self._calculate_rolling_correlation(prices, volumes, 5)
        features["price_volume_corr_5d"] = corr_5d

        # Volume rate of change
        vol_roc = self._calculate_rate_of_change(volumes, 5)
        features["volume_roc_5d"] = vol_roc

        return features

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return [50.0] * len(prices)

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))

        rsi_values = [50.0]  # Default for first value

        if len(gains) >= period:
            # Initial average
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                rsi_values.append(rsi)

        # Pad to match input length
        while len(rsi_values) < len(prices):
            rsi_values.append(rsi_values[-1])

        return rsi_values

    def _calculate_sma(self, values: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average."""
        sma = []
        for i in range(len(values)):
            if i >= period - 1:
                avg = np.mean(values[i - period + 1 : i + 1])
                sma.append(avg)
            else:
                sma.append(values[i] if values else 0.0)
        return sma

    def _calculate_price_percentile(
        self, prices: List[float], period: int
    ) -> List[float]:
        """Calculate price percentile within rolling window."""
        percentiles = []
        for i in range(len(prices)):
            if i >= period - 1:
                window = prices[i - period + 1 : i + 1]
                current_price = prices[i]
                rank = sum(1 for p in window if p <= current_price)
                percentile = rank / len(window)
                percentiles.append(percentile)
            else:
                percentiles.append(0.5)  # Neutral
        return percentiles

    def _calculate_rolling_correlation(
        self, x: List[float], y: List[float], period: int
    ) -> List[float]:
        """Calculate rolling correlation between two series."""
        correlations = []
        for i in range(len(x)):
            if i >= period - 1:
                x_window = x[i - period + 1 : i + 1]
                y_window = y[i - period + 1 : i + 1]

                try:
                    corr = np.corrcoef(x_window, y_window)[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0.0)
                except:
                    correlations.append(0.0)
            else:
                correlations.append(0.0)
        return correlations

    def _calculate_rate_of_change(
        self, values: List[float], period: int
    ) -> List[float]:
        """Calculate rate of change over period."""
        roc = []
        for i in range(len(values)):
            if i >= period:
                change = (
                    (values[i] - values[i - period]) / values[i - period]
                    if values[i - period] != 0
                    else 0
                )
                roc.append(change)
            else:
                roc.append(0.0)
        return roc


class FundamentalFeatures:
    """Fundamental analysis feature engineering."""

    def __init__(self):
        self.logger = TradingLogger("fundamental_features")

    def calculate_market_features(
        self, market_data: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """Calculate market-wide fundamental features."""
        features = {}

        # VIX-like volatility index (simplified)
        if "vix" in market_data:
            features["vix_level"] = market_data["vix"]
            features["vix_change"] = self._calculate_change(market_data["vix"], 1)

        # Interest rate features
        if "interest_rate" in market_data:
            features["interest_rate"] = market_data["interest_rate"]
            features["rate_change"] = self._calculate_change(
                market_data["interest_rate"], 5
            )

        # Currency strength (if applicable)
        if "usd_index" in market_data:
            features["usd_strength"] = market_data["usd_index"]

        return features

    def calculate_sector_features(
        self, symbol: str, sector_data: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate sector-relative features."""
        features = {}

        # Sector performance relative to market
        if "sector_return" in sector_data and "market_return" in sector_data:
            features["sector_relative_performance"] = (
                sector_data["sector_return"] - sector_data["market_return"]
            )

        # Sector volatility
        if "sector_volatility" in sector_data:
            features["sector_volatility"] = sector_data["sector_volatility"]

        return features

    def _calculate_change(self, values: List[float], period: int) -> List[float]:
        """Calculate change over period."""
        changes = []
        for i in range(len(values)):
            if i >= period:
                change = values[i] - values[i - period]
                changes.append(change)
            else:
                changes.append(0.0)
        return changes


class FeatureEngineer:
    """Main feature engineering class that coordinates all feature types."""

    def __init__(self):
        self.logger = TradingLogger("feature_engineer")
        self.technical = TechnicalFeatures()
        self.fundamental = FundamentalFeatures()

        # Feature history storage
        self.feature_history: Dict[str, deque] = {}
        self.target_history: Dict[str, deque] = {}

    def engineer_features(
        self, symbol: str, bars: List[MarketData], lookback_periods: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer features for a symbol.

        Returns:
        - features: numpy array of shape (n_samples, n_features)
        - targets: numpy array of shape (n_samples,) - future returns
        """

        if len(bars) < lookback_periods:
            self.logger.logger.warning(
                f"Insufficient data for {symbol}: {len(bars)} < {lookback_periods}"
            )
            return np.array([]), np.array([])

        # Extract basic data
        prices = [float(bar.close) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]
        highs = [float(bar.high) for bar in bars]
        lows = [float(bar.low) for bar in bars]

        # Technical features
        feature_dict = {}

        # Returns
        returns_features = self.technical.calculate_returns(prices)
        feature_dict.update(returns_features)

        # Volatility
        daily_returns = returns_features["return_1d"]
        vol_features = self.technical.calculate_volatility_features(daily_returns)
        feature_dict.update(vol_features)

        # Momentum
        momentum_features = self.technical.calculate_momentum_features(prices)
        feature_dict.update(momentum_features)

        # Volume
        volume_features = self.technical.calculate_volume_features(volumes, prices)
        feature_dict.update(volume_features)

        # Additional technical features
        feature_dict.update(
            self._calculate_additional_features(prices, highs, lows, volumes)
        )

        # Convert to numpy array
        features_df = pd.DataFrame(feature_dict)

        # Handle missing values
        features_df = features_df.fillna(method="ffill").fillna(0)

        # Create targets (future returns)
        targets = self._create_targets(prices, horizon=5)  # 5-day forward returns

        # Ensure same length
        min_length = min(len(features_df), len(targets))
        features_array = features_df.iloc[:min_length].values
        targets_array = np.array(targets[:min_length])

        # Store in history
        self._update_feature_history(symbol, features_array, targets_array)

        return features_array, targets_array

    def _calculate_additional_features(
        self,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
    ) -> Dict[str, List[float]]:
        """Calculate additional technical features."""
        features = {}

        # High-Low spread
        hl_spread = [(h - l) / l if l > 0 else 0 for h, l in zip(highs, lows)]
        features["hl_spread"] = hl_spread

        # Price position within daily range
        price_position = [
            (p - l) / (h - l) if h > l else 0.5 for p, h, l in zip(prices, highs, lows)
        ]
        features["price_position"] = price_position

        # Volume-weighted price
        vwap = self._calculate_vwap(prices, volumes)
        features["price_to_vwap"] = [
            p / v if v > 0 else 1.0 for p, v in zip(prices, vwap)
        ]

        # Bollinger Band position
        bb_position = self._calculate_bollinger_position(prices)
        features["bollinger_position"] = bb_position

        return features

    def _calculate_vwap(
        self, prices: List[float], volumes: List[float], period: int = 20
    ) -> List[float]:
        """Calculate Volume Weighted Average Price."""
        vwap = []
        for i in range(len(prices)):
            if i >= period - 1:
                window_prices = prices[i - period + 1 : i + 1]
                window_volumes = volumes[i - period + 1 : i + 1]

                total_pv = sum(p * v for p, v in zip(window_prices, window_volumes))
                total_volume = sum(window_volumes)

                if total_volume > 0:
                    vwap.append(total_pv / total_volume)
                else:
                    vwap.append(prices[i])
            else:
                vwap.append(prices[i])

        return vwap

    def _calculate_bollinger_position(
        self, prices: List[float], period: int = 20, std_dev: float = 2.0
    ) -> List[float]:
        """Calculate position within Bollinger Bands."""
        positions = []

        for i in range(len(prices)):
            if i >= period - 1:
                window = prices[i - period + 1 : i + 1]
                mean = np.mean(window)
                std = np.std(window)

                upper_band = mean + (std_dev * std)
                lower_band = mean - (std_dev * std)

                if upper_band > lower_band:
                    position = (prices[i] - lower_band) / (upper_band - lower_band)
                    positions.append(max(0, min(1, position)))
                else:
                    positions.append(0.5)
            else:
                positions.append(0.5)

        return positions

    def _create_targets(self, prices: List[float], horizon: int = 5) -> List[float]:
        """Create target variables (future returns)."""
        targets = []

        for i in range(len(prices)):
            if i + horizon < len(prices):
                future_return = (prices[i + horizon] - prices[i]) / prices[i]
                targets.append(future_return)
            else:
                targets.append(0.0)  # No future data available

        return targets

    def _update_feature_history(
        self, symbol: str, features: np.ndarray, targets: np.ndarray
    ) -> None:
        """Update feature and target history for a symbol."""
        if symbol not in self.feature_history:
            self.feature_history[symbol] = deque(maxlen=1000)  # Keep last 1000 samples
            self.target_history[symbol] = deque(maxlen=1000)

        # Add new features and targets
        for feat_row, target in zip(features, targets):
            self.feature_history[symbol].append(feat_row)
            self.target_history[symbol].append(target)

    def get_feature_matrix(
        self, symbol: str, min_samples: int = 100
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get feature matrix and targets for training."""
        if symbol not in self.feature_history:
            return None, None

        if len(self.feature_history[symbol]) < min_samples:
            return None, None

        features = np.array(list(self.feature_history[symbol]))
        targets = np.array(list(self.target_history[symbol]))

        return features, targets

    def get_latest_features(self, symbol: str) -> Optional[np.ndarray]:
        """Get latest features for prediction."""
        if symbol not in self.feature_history or len(self.feature_history[symbol]) == 0:
            return None

        return np.array([list(self.feature_history[symbol])[-1]])

    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        # This would normally be tracked during feature engineering
        return [
            "return_1d",
            "return_5d",
            "return_10d",
            "return_20d",
            "volatility_5d",
            "volatility_10d",
            "volatility_20d",
            "rsi",
            "price_to_sma20",
            "price_percentile_20d",
            "volume_ratio_10d",
            "price_volume_corr_5d",
            "volume_roc_5d",
            "hl_spread",
            "price_position",
            "price_to_vwap",
            "bollinger_position",
        ]

    def calculate_feature_importance(
        self, symbol: str, model
    ) -> Optional[Dict[str, float]]:
        """Calculate feature importance if model supports it."""
        try:
            if hasattr(model, "feature_importances_"):
                feature_names = self.get_feature_names()
                importances = model.feature_importances_

                return dict(zip(feature_names, importances))
        except Exception as e:
            self.logger.logger.error(f"Error calculating feature importance: {e}")

        return None
