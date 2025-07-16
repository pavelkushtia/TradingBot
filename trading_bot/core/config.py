"""Configuration management for the trading bot."""

import os
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ExchangeConfig(BaseModel):
    """Exchange configuration."""
    
    name: str = Field(default="alpaca")
    api_key: str = Field(default="")
    secret_key: str = Field(default="")
    base_url: str = Field(default="https://paper-api.alpaca.markets")
    environment: str = Field(default="sandbox")


class TradingConfig(BaseModel):
    """Trading configuration."""
    
    portfolio_value: float = Field(default=100000.0, gt=0)
    max_position_size: float = Field(default=0.05, gt=0, le=1.0)
    stop_loss_percentage: float = Field(default=0.02, gt=0, le=1.0)
    take_profit_percentage: float = Field(default=0.04, gt=0, le=1.0)


class RiskConfig(BaseModel):
    """Risk management configuration."""
    
    max_daily_loss: float = Field(default=0.02, gt=0, le=1.0)
    max_open_positions: int = Field(default=10, gt=0)
    risk_free_rate: float = Field(default=0.02, ge=0)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    url: str = Field(default="sqlite+aiosqlite:///trading_bot.db")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO")
    format: str = Field(default="json")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    
    enable_prometheus: bool = Field(default=False)
    prometheus_port: int = Field(default=8000, gt=0, le=65535)


class StrategyConfig(BaseModel):
    """Strategy configuration."""
    
    default_strategy: str = Field(default="momentum_crossover")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('parameters', pre=True)
    def parse_parameters(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v


class MarketDataConfig(BaseModel):
    """Market data configuration."""
    
    provider: str = Field(default="alpaca")
    websocket_reconnect_delay: int = Field(default=5, gt=0)
    max_reconnect_attempts: int = Field(default=10, gt=0)


class Config(BaseModel):
    """Main configuration class."""
    
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        
        exchange_config = ExchangeConfig(
            name=os.getenv("EXCHANGE", "alpaca"),
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            environment=os.getenv("ENVIRONMENT", "sandbox")
        )
        
        trading_config = TradingConfig(
            portfolio_value=float(os.getenv("DEFAULT_PORTFOLIO_VALUE", 100000)),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", 0.05)),
            stop_loss_percentage=float(os.getenv("STOP_LOSS_PERCENTAGE", 0.02)),
            take_profit_percentage=float(os.getenv("TAKE_PROFIT_PERCENTAGE", 0.04))
        )
        
        risk_config = RiskConfig(
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", 0.02)),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", 10)),
            risk_free_rate=float(os.getenv("RISK_FREE_RATE", 0.02))
        )
        
        database_config = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///trading_bot.db")
        )
        
        logging_config = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "json")
        )
        
        monitoring_config = MonitoringConfig(
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", 8000))
        )
        
        strategy_config = StrategyConfig(
            default_strategy=os.getenv("DEFAULT_STRATEGY", "momentum_crossover"),
            parameters=os.getenv("STRATEGY_PARAMETERS", '{"short_window": 10, "long_window": 30}')
        )
        
        market_data_config = MarketDataConfig(
            provider=os.getenv("MARKET_DATA_PROVIDER", "alpaca"),
            websocket_reconnect_delay=int(os.getenv("WEBSOCKET_RECONNECT_DELAY", 5)),
            max_reconnect_attempts=int(os.getenv("MAX_RECONNECT_ATTEMPTS", 10))
        )
        
        return cls(
            exchange=exchange_config,
            trading=trading_config,
            risk=risk_config,
            database=database_config,
            logging=logging_config,
            monitoring=monitoring_config,
            strategy=strategy_config,
            market_data=market_data_config
        ) 