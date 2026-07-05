"""Strategy package — importing it registers every built-in strategy."""

from src.strategies.base import (  # noqa: F401
    STRATEGY_REGISTRY,
    STRATEGY_SPECS,
    ParamSpec,
    StrategySpec,
    register_strategy,
)
from src.strategies import bollinger, ma_crossover, rsi  # noqa: F401  (registration side effects)
