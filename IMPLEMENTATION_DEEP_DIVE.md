# Trading Bot Implementation Deep Dive

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Multiple Data Sources Implementation](#multiple-data-sources-implementation)
3. [Advanced Technical Indicators](#advanced-technical-indicators)
4. [AI/ML Integration Architecture](#aiml-integration-architecture)
5. [Portfolio Optimization Framework](#portfolio-optimization-framework)
6. [Advanced Backtesting Engine](#advanced-backtesting-engine)
7. [Real-time Strategy Optimization](#real-time-strategy-optimization)
8. [Production Deployment Strategy](#production-deployment-strategy)

---

## Architecture Overview

### Current vs Target Architecture

**Current Architecture:**
```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Market Data   │────│ Trading Bot  │────│   Execution     │
│   (Alpaca)      │    │   (Basic)    │    │  (Alpaca API)   │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                       ┌──────────────┐
                       │   SQLite     │
                       │  Database    │
                       └──────────────┘
```

**Target Architecture:**
```
┌─────────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────────┐
│    Multi-Source Data    │    │     AI/ML Engine         │    │    Multi-Broker         │
│  ┌─────────────────────┐│    │  ┌──────────────────────┐│    │  ┌─────────────────────┐│
│  │ Alpaca (Primary)    ││────│  │ Feature Engineering  ││────│  │ Alpaca (Primary)    ││
│  │ Alpha Vantage       ││    │  │ ML Models            ││    │  │ Interactive Brokers ││
│  │ Yahoo Finance       ││    │  │ Portfolio Optimizer  ││    │  │ TD Ameritrade       ││
│  │ Finnhub             ││    │  │ Risk Models          ││    │  │ Paper Trading       ││
│  └─────────────────────┘│    │  └──────────────────────┘│    │  └─────────────────────┘│
└─────────────────────────┘    └──────────────────────────┘    └─────────────────────────┘
              │                              │                              │
              └──────────────────────────────┼──────────────────────────────┘
                                            │
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              Event-Driven Trading Core                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ Strategy     │  │ Risk         │  │ Portfolio    │  │ Order        │                │
│  │ Manager      │  │ Manager      │  │ Manager      │  │ Manager      │                │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
              │
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              Data Storage & Analytics                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ TimescaleDB  │  │ Redis Cache  │  │ Model Store  │  │ Analytics    │                │
│  │ (Time Series)│  │ (Real-time)  │  │ (MLflow)     │  │ Dashboard    │                │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Multiple Data Sources Implementation

### 1. Data Provider Abstraction Layer

```python
# trading_bot/market_data/providers/base.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from dataclasses import dataclass

@dataclass
class DataProviderConfig:
    name: str
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 60
    max_retries: int = 3
    timeout: float = 10.0
    priority: int = 1  # Lower number = higher priority
    cost_per_call: float = 0.0  # For cost optimization

class BaseDataProvider(ABC):
    def __init__(self, config: DataProviderConfig):
        self.config = config
        self.session = None
        self.rate_limiter = AsyncRateLimiter(config.rate_limit_per_minute)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=300,
            expected_exception=Exception
        )
    
    @abstractmethod
    async def get_historical_bars(self, symbol: str, timeframe: str, 
                                 start: datetime, end: datetime) -> List[MarketData]:
        pass
    
    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> Quote:
        pass
    
    @abstractmethod
    async def get_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        pass
    
    async def is_available(self) -> bool:
        return not self.circuit_breaker.is_open()
    
    async def test_connection(self) -> bool:
        try:
            # Simple test call
            await self.get_real_time_quote("AAPL")
            return True
        except Exception:
            return False
```

### 2. Alpaca Provider (Primary)

```python
# trading_bot/market_data/providers/alpaca.py
class AlpacaProvider(BaseDataProvider):
    def __init__(self, config: DataProviderConfig):
        super().__init__(config)
        self.rest_client = None
        self.websocket_client = None
    
    async def initialize(self):
        from alpaca_trade_api import REST
        self.rest_client = REST(
            self.config.api_key,
            self.config.secret_key,
            base_url=self.config.base_url,
            api_version='v2'
        )
    
    async def get_historical_bars(self, symbol: str, timeframe: str, 
                                 start: datetime, end: datetime) -> List[MarketData]:
        async with self.rate_limiter:
            async with self.circuit_breaker:
                try:
                    bars = self.rest_client.get_bars(
                        symbol,
                        timeframe,
                        start=start.isoformat(),
                        end=end.isoformat(),
                        adjustment='raw'
                    )
                    return [self._convert_bar(bar) for bar in bars]
                except Exception as e:
                    logger.error(f"Alpaca API error: {e}")
                    raise DataProviderError(f"Alpaca failed: {e}")
    
    def _convert_bar(self, bar) -> MarketData:
        return MarketData(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            open=Decimal(str(bar.open)),
            high=Decimal(str(bar.high)),
            low=Decimal(str(bar.low)),
            close=Decimal(str(bar.close)),
            volume=bar.volume
        )
```

### 3. Alpha Vantage Provider (Fallback)

```python
# trading_bot/market_data/providers/alpha_vantage.py
class AlphaVantageProvider(BaseDataProvider):
    def __init__(self, config: DataProviderConfig):
        super().__init__(config)
        self.base_url = "https://www.alphavantage.co/query"
    
    async def get_historical_bars(self, symbol: str, timeframe: str, 
                                 start: datetime, end: datetime) -> List[MarketData]:
        async with self.rate_limiter:
            async with self.circuit_breaker:
                params = {
                    'function': 'TIME_SERIES_INTRADAY',
                    'symbol': symbol,
                    'interval': self._convert_timeframe(timeframe),
                    'apikey': self.config.api_key,
                    'outputsize': 'full'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_timeseries_data(data, symbol)
                        else:
                            raise DataProviderError(f"Alpha Vantage API error: {response.status}")
    
    def _convert_timeframe(self, timeframe: str) -> str:
        mapping = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1hour': '60min'
        }
        return mapping.get(timeframe, '5min')
    
    def _parse_timeseries_data(self, data: dict, symbol: str) -> List[MarketData]:
        # Implementation to parse Alpha Vantage response format
        time_series_key = [k for k in data.keys() if 'Time Series' in k][0]
        time_series = data[time_series_key]
        
        bars = []
        for timestamp_str, values in time_series.items():
            bars.append(MarketData(
                symbol=symbol,
                timestamp=datetime.fromisoformat(timestamp_str),
                open=Decimal(values['1. open']),
                high=Decimal(values['2. high']),
                low=Decimal(values['3. low']),
                close=Decimal(values['4. close']),
                volume=int(values['5. volume'])
            ))
        
        return sorted(bars, key=lambda x: x.timestamp)
```

### 4. Smart Data Manager

```python
# trading_bot/market_data/smart_manager.py
class SmartDataManager:
    def __init__(self, config: Config):
        self.config = config
        self.providers = {}
        self.cache = RedisCache()
        self.cost_tracker = CostTracker()
        
    async def initialize(self):
        # Initialize providers based on configuration
        provider_configs = [
            DataProviderConfig("alpaca", priority=1, rate_limit_per_minute=200),
            DataProviderConfig("alpha_vantage", priority=2, rate_limit_per_minute=25),
            DataProviderConfig("yahoo", priority=3, rate_limit_per_minute=1000),
            DataProviderConfig("finnhub", priority=4, rate_limit_per_minute=60)
        ]
        
        for provider_config in provider_configs:
            provider = self._create_provider(provider_config)
            if await provider.test_connection():
                self.providers[provider_config.name] = provider
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                 start: datetime, end: datetime, 
                                 fallback: bool = True) -> List[MarketData]:
        # Check cache first
        cache_key = f"hist:{symbol}:{timeframe}:{start.isoformat()}:{end.isoformat()}"
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # Try providers in priority order
        sorted_providers = sorted(
            self.providers.values(), 
            key=lambda p: p.config.priority
        )
        
        last_exception = None
        for provider in sorted_providers:
            try:
                if await provider.is_available():
                    data = await provider.get_historical_bars(symbol, timeframe, start, end)
                    
                    # Cache successful result
                    await self.cache.set(cache_key, data, ttl=300)
                    
                    # Track usage
                    await self.cost_tracker.record_usage(provider.config.name, 1)
                    
                    logger.info(f"Successfully retrieved data from {provider.config.name}")
                    return data
                    
            except Exception as e:
                logger.warning(f"Provider {provider.config.name} failed: {e}")
                last_exception = e
                if not fallback:
                    break
        
        # If all providers failed, raise the last exception
        raise DataProviderError(f"All data providers failed. Last error: {last_exception}")
    
    async def get_cost_report(self) -> Dict[str, Any]:
        return await self.cost_tracker.get_report()
```

---

## Advanced Technical Indicators

### 1. Technical Indicator Framework

```python
# trading_bot/indicators/base.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union, List, Optional

class TechnicalIndicator(ABC):
    def __init__(self, name: str, period: int):
        self.name = name
        self.period = period
        self.values = []
        self.is_ready = False
    
    @abstractmethod
    def calculate(self, data: pd.Series) -> Union[float, pd.Series]:
        pass
    
    def update(self, value: float) -> Optional[float]:
        self.values.append(value)
        if len(self.values) > self.period * 2:  # Keep some history
            self.values = self.values[-self.period * 2:]
        
        if len(self.values) >= self.period:
            self.is_ready = True
            return self.calculate(pd.Series(self.values))
        
        return None
    
    def reset(self):
        self.values = []
        self.is_ready = False
```

### 2. Comprehensive Indicator Library

```python
# trading_bot/indicators/momentum.py
class RSI(TechnicalIndicator):
    def __init__(self, period: int = 14):
        super().__init__("RSI", period)
    
    def calculate(self, data: pd.Series) -> float:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

class MACD(TechnicalIndicator):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD", slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.Series) -> dict:
        exp1 = data.ewm(span=self.fast_period).mean()
        exp2 = data.ewm(span=self.slow_period).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.signal_period).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1]
        }

class StochasticOscillator(TechnicalIndicator):
    def __init__(self, k_period: int = 14, d_period: int = 3):
        super().__init__("Stochastic", k_period)
        self.k_period = k_period
        self.d_period = d_period
        self.high_values = []
        self.low_values = []
    
    def update_ohlc(self, high: float, low: float, close: float) -> Optional[dict]:
        self.high_values.append(high)
        self.low_values.append(low)
        self.values.append(close)
        
        if len(self.values) > self.k_period * 2:
            self.high_values = self.high_values[-self.k_period * 2:]
            self.low_values = self.low_values[-self.k_period * 2:]
            self.values = self.values[-self.k_period * 2:]
        
        if len(self.values) >= self.k_period:
            return self.calculate_stoch()
        
        return None
    
    def calculate_stoch(self) -> dict:
        close_series = pd.Series(self.values)
        high_series = pd.Series(self.high_values)
        low_series = pd.Series(self.low_values)
        
        lowest_low = low_series.rolling(window=self.k_period).min()
        highest_high = high_series.rolling(window=self.k_period).max()
        
        k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        return {
            'k_percent': k_percent.iloc[-1],
            'd_percent': d_percent.iloc[-1]
        }
```

### 3. Advanced Volume Indicators

```python
# trading_bot/indicators/volume.py
class OnBalanceVolume(TechnicalIndicator):
    def __init__(self):
        super().__init__("OBV", 1)
        self.obv_value = 0
        self.prev_close = None
    
    def update_price_volume(self, close: float, volume: int) -> float:
        if self.prev_close is not None:
            if close > self.prev_close:
                self.obv_value += volume
            elif close < self.prev_close:
                self.obv_value -= volume
            # If close == prev_close, OBV stays the same
        
        self.prev_close = close
        return self.obv_value

class VWAP(TechnicalIndicator):
    def __init__(self, period: int = 20):
        super().__init__("VWAP", period)
        self.price_volume_sum = 0
        self.volume_sum = 0
        self.pv_values = []
        self.volume_values = []
    
    def update_price_volume(self, typical_price: float, volume: int) -> Optional[float]:
        pv = typical_price * volume
        self.pv_values.append(pv)
        self.volume_values.append(volume)
        
        if len(self.pv_values) > self.period:
            self.pv_values = self.pv_values[-self.period:]
            self.volume_values = self.volume_values[-self.period:]
        
        if len(self.pv_values) >= self.period:
            return sum(self.pv_values) / sum(self.volume_values)
        
        return None

class MoneyFlowIndex(TechnicalIndicator):
    def __init__(self, period: int = 14):
        super().__init__("MFI", period)
        self.typical_prices = []
        self.volumes = []
    
    def update_ohlcv(self, high: float, low: float, close: float, volume: int) -> Optional[float]:
        typical_price = (high + low + close) / 3
        self.typical_prices.append(typical_price)
        self.volumes.append(volume)
        
        if len(self.typical_prices) > self.period + 1:
            self.typical_prices = self.typical_prices[-(self.period + 1):]
            self.volumes = self.volumes[-(self.period + 1):]
        
        if len(self.typical_prices) >= self.period + 1:
            return self.calculate_mfi()
        
        return None
    
    def calculate_mfi(self) -> float:
        money_flows = []
        for i in range(1, len(self.typical_prices)):
            raw_money_flow = self.typical_prices[i] * self.volumes[i]
            if self.typical_prices[i] > self.typical_prices[i-1]:
                money_flows.append(('positive', raw_money_flow))
            elif self.typical_prices[i] < self.typical_prices[i-1]:
                money_flows.append(('negative', raw_money_flow))
            else:
                money_flows.append(('neutral', raw_money_flow))
        
        positive_flow = sum(flow for direction, flow in money_flows if direction == 'positive')
        negative_flow = sum(flow for direction, flow in money_flows if direction == 'negative')
        
        if negative_flow == 0:
            return 100
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
```

### 4. Indicator Manager and Caching

```python
# trading_bot/indicators/manager.py
class IndicatorManager:
    def __init__(self):
        self.indicators = {}
        self.indicator_cache = {}
        self.cache_ttl = 60  # 1 minute
    
    def register_indicator(self, symbol: str, indicator_name: str, 
                          indicator_class: type, **kwargs) -> str:
        indicator_id = f"{symbol}:{indicator_name}:{hash(str(kwargs))}"
        
        if indicator_id not in self.indicators:
            self.indicators[indicator_id] = indicator_class(**kwargs)
        
        return indicator_id
    
    async def calculate_indicators(self, symbol: str, 
                                 market_data: List[MarketData]) -> Dict[str, Any]:
        results = {}
        
        # Get all indicators for this symbol
        symbol_indicators = {
            id: indicator for id, indicator in self.indicators.items()
            if id.startswith(f"{symbol}:")
        }
        
        for indicator_id, indicator in symbol_indicators.items():
            try:
                # Check cache first
                cache_key = f"{indicator_id}:{len(market_data)}"
                cached_result = self.indicator_cache.get(cache_key)
                
                if cached_result and (time.time() - cached_result['timestamp']) < self.cache_ttl:
                    results[indicator.name] = cached_result['value']
                    continue
                
                # Calculate indicator
                if hasattr(indicator, 'update_ohlcv'):
                    # For indicators that need OHLCV data
                    for bar in market_data:
                        value = indicator.update_ohlcv(
                            float(bar.high), float(bar.low), 
                            float(bar.close), bar.volume
                        )
                    if value is not None:
                        results[indicator.name] = value
                else:
                    # For simple price-based indicators
                    close_prices = [float(bar.close) for bar in market_data]
                    value = indicator.calculate(pd.Series(close_prices))
                    if value is not None:
                        results[indicator.name] = value
                
                # Cache the result
                if indicator.name in results:
                    self.indicator_cache[cache_key] = {
                        'value': results[indicator.name],
                        'timestamp': time.time()
                    }
                
            except Exception as e:
                logger.error(f"Error calculating {indicator.name}: {e}")
        
        return results
    
    def get_indicator_definitions(self) -> Dict[str, type]:
        """Return all available indicator classes"""
        return {
            # Trend Indicators
            'SMA': SimpleMovingAverage,
            'EMA': ExponentialMovingAverage,
            'WMA': WeightedMovingAverage,
            'MACD': MACD,
            'ADX': AverageDirectionalIndex,
            'PSAR': ParabolicSAR,
            
            # Momentum Indicators
            'RSI': RSI,
            'STOCH': StochasticOscillator,
            'WILLIAMS_R': WilliamsR,
            'ROC': RateOfChange,
            'CMO': ChandeMomentumOscillator,
            'TSI': TrueStrengthIndex,
            
            # Volatility Indicators
            'BOLLINGER': BollingerBands,
            'ATR': AverageTrueRange,
            'KELTNER': KeltnerChannels,
            'DONCHIAN': DonchianChannels,
            
            # Volume Indicators
            'OBV': OnBalanceVolume,
            'VWAP': VWAP,
            'MFI': MoneyFlowIndex,
            'AD': AccumulationDistribution,
            'CHAIKIN': ChaikinOscillator,
            
            # Support/Resistance
            'PIVOT': PivotPoints,
            'FIBONACCI': FibonacciRetracements,
            'SUPPORT_RESISTANCE': SupportResistanceLevels
        }
```

---

## AI/ML Integration Architecture

### 1. Feature Engineering Pipeline

```python
# trading_bot/ml/features/technical.py
class TechnicalFeatureEngineer:
    def __init__(self, indicator_manager: IndicatorManager):
        self.indicator_manager = indicator_manager
        self.features = {}
    
    async def extract_features(self, symbol: str, 
                              market_data: List[MarketData],
                              timeframes: List[str] = ['1min', '5min', '15min', '1h']) -> pd.DataFrame:
        features = {}
        
        # Price-based features
        df = self._to_dataframe(market_data)
        features.update(self._extract_price_features(df))
        
        # Technical indicator features
        indicators = await self.indicator_manager.calculate_indicators(symbol, market_data)
        features.update(self._transform_indicators(indicators))
        
        # Multi-timeframe features
        for timeframe in timeframes:
            tf_data = self._resample_data(df, timeframe)
            tf_features = self._extract_price_features(tf_data, prefix=f"{timeframe}_")
            features.update(tf_features)
        
        # Statistical features
        features.update(self._extract_statistical_features(df))
        
        return pd.DataFrame([features])
    
    def _extract_price_features(self, df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
        """Extract price-based features"""
        features = {}
        
        # Returns
        df['returns'] = df['close'].pct_change()
        features[f'{prefix}return_1'] = df['returns'].iloc[-1]
        features[f'{prefix}return_5'] = df['returns'].rolling(5).sum().iloc[-1]
        features[f'{prefix}return_20'] = df['returns'].rolling(20).sum().iloc[-1]
        
        # Volatility
        features[f'{prefix}volatility_5'] = df['returns'].rolling(5).std().iloc[-1]
        features[f'{prefix}volatility_20'] = df['returns'].rolling(20).std().iloc[-1]
        
        # Price momentum
        features[f'{prefix}momentum_5'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1)
        features[f'{prefix}momentum_10'] = (df['close'].iloc[-1] / df['close'].iloc[-11] - 1)
        
        # Volume features
        features[f'{prefix}volume_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        features[f'{prefix}volume_trend'] = df['volume'].rolling(5).mean().iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        return {k: v for k, v in features.items() if not pd.isna(v)}
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical features"""
        features = {}
        
        # Price statistics
        features['price_percentile'] = (df['close'].iloc[-1] - df['close'].rolling(100).min().iloc[-1]) / (
            df['close'].rolling(100).max().iloc[-1] - df['close'].rolling(100).min().iloc[-1]
        )
        
        # Trend strength
        x = np.arange(len(df.tail(20)))
        y = df['close'].tail(20).values
        slope = np.polyfit(x, y, 1)[0]
        features['trend_strength'] = slope / df['close'].iloc[-1]
        
        # Mean reversion tendency
        features['distance_from_mean'] = (df['close'].iloc[-1] - df['close'].rolling(50).mean().iloc[-1]) / df['close'].rolling(50).std().iloc[-1]
        
        return {k: v for k, v in features.items() if not pd.isna(v)}
```

### 2. ML Model Framework

```python
# trading_bot/ml/models/ensemble.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from typing import Dict, List, Any
import joblib

class EnsembleModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize base models for ensemble"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
    
    async def train(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series):
        """Train ensemble model with cross-validation"""
        self.initialize_models()
        
        # Train individual models
        model_scores = {}
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                model_scores[name] = val_score
                logger.info(f"{name} validation score: {val_score:.4f}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                model_scores[name] = 0.0
        
        # Calculate weights based on validation performance
        total_score = sum(model_scores.values())
        if total_score > 0:
            self.weights = {name: score / total_score for name, score in model_scores.items()}
        else:
            # Equal weights if all models failed
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        self.is_trained = True
        logger.info(f"Ensemble weights: {self.weights}")
    
    async def predict(self, X: pd.DataFrame) -> Dict[str, float]:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    # Assuming binary classification for now
                    probabilities[name] = proba[0][1] if proba.shape[1] > 1 else proba[0][0]
                else:
                    pred = model.predict(X)
                    probabilities[name] = float(pred[0])
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
                probabilities[name] = 0.5  # Neutral prediction
        
        # Weighted ensemble prediction
        ensemble_prob = sum(prob * self.weights.get(name, 0) for name, prob in probabilities.items())
        
        return {
            'ensemble_probability': ensemble_prob,
            'ensemble_prediction': 1 if ensemble_prob > 0.5 else 0,
            'confidence': abs(ensemble_prob - 0.5) * 2,  # Confidence score 0-1
            'individual_predictions': probabilities
        }
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'config': self.config,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
```

### 3. Reinforcement Learning Framework

```python
# trading_bot/ml/models/reinforcement.py
import gym
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from typing import Dict, Any, Tuple
import pandas as pd

class TradingEnvironment(gym.Env):
    """Custom trading environment for RL"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000,
                 transaction_cost: float = 0.001):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Action space: [position_size] where position_size is between -1 and 1
        # -1 = full short, 0 = no position, 1 = full long
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: features from market data
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.data.columns),), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        
        return self._get_observation()
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate previous portfolio value
        prev_portfolio_value = self.portfolio_value
        
        # Execute action
        target_position = action[0]
        position_change = target_position - self.position
        
        # Calculate transaction cost
        cost = abs(position_change) * current_price * self.transaction_cost
        
        # Update position and balance
        if position_change != 0:
            self.balance -= cost
            self.position = target_position
            self.entry_price = current_price
        
        # Calculate current portfolio value
        position_value = self.position * current_price * self.initial_balance
        self.portfolio_value = self.balance + position_value
        
        # Update max portfolio value for drawdown calculation
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value)
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'drawdown': (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        if self.current_step >= len(self.data):
            return np.zeros(len(self.data.columns), dtype=np.float32)
        
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        
        # Add portfolio state to observation
        portfolio_state = np.array([
            self.position,
            self.portfolio_value / self.initial_balance - 1,  # Return
            (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value  # Drawdown
        ], dtype=np.float32)
        
        return np.concatenate([obs, portfolio_state])
    
    def _calculate_reward(self, prev_portfolio_value):
        # Return-based reward
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Risk-adjusted reward (Sharpe-like)
        risk_penalty = 0.0
        if hasattr(self, 'returns_history'):
            self.returns_history.append(portfolio_return)
            if len(self.returns_history) > 20:
                returns_std = np.std(self.returns_history[-20:])
                if returns_std > 0:
                    risk_penalty = returns_std * 0.1
        else:
            self.returns_history = [portfolio_return]
        
        # Drawdown penalty
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = drawdown * 0.5
        
        return portfolio_return - risk_penalty - drawdown_penalty

class RLTradingAgent:
    def __init__(self, algorithm: str = 'PPO', config: Dict[str, Any] = None):
        self.algorithm = algorithm
        self.config = config or {}
        self.model = None
        self.env = None
        
    def setup_environment(self, training_data: pd.DataFrame):
        """Setup trading environment with data"""
        self.env = TradingEnvironment(training_data)
        
        # Update observation space to include portfolio state
        obs_dim = len(training_data.columns) + 3  # +3 for portfolio state
        self.env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
    
    def train(self, training_data: pd.DataFrame, total_timesteps: int = 100000):
        """Train RL agent"""
        self.setup_environment(training_data)
        
        if self.algorithm == 'PPO':
            self.model = PPO('MlpPolicy', self.env, verbose=1, **self.config)
        elif self.algorithm == 'DDPG':
            self.model = DDPG('MlpPolicy', self.env, verbose=1, **self.config)
        elif self.algorithm == 'SAC':
            self.model = SAC('MlpPolicy', self.env, verbose=1, **self.config)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        self.model.learn(total_timesteps=total_timesteps)
        
        logger.info(f"RL agent training completed with {total_timesteps} timesteps")
    
    def predict(self, observation: np.ndarray) -> Tuple[float, float]:
        """Predict action given observation"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        action, _states = self.model.predict(observation, deterministic=True)
        
        # Return position size and confidence
        position_size = action[0]
        confidence = abs(position_size)  # Use absolute value as confidence
        
        return float(position_size), float(confidence)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        if self.algorithm == 'PPO':
            self.model = PPO.load(filepath)
        elif self.algorithm == 'DDPG':
            self.model = DDPG.load(filepath)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(filepath)
```

This comprehensive implementation guide provides the foundation for transforming our basic trading bot into a sophisticated, AI-powered trading platform. The modular architecture ensures compatibility with the Alpaca free tier while providing intelligent fallback mechanisms and cost optimization strategies. 