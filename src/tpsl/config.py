from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus


@dataclass(frozen=True)
class DatabaseConfig:
    direct_url: str = ""
    url_env: str = "MYSQL_URL"
    driver: str = "mysql+pymysql"
    host: str = "127.0.0.1"
    port: int = 3306
    username: str = "root"
    password: str = ""
    name: str = "take_profit_stop_loss"
    pool_size: int = 10
    max_overflow: int = 5
    tables: dict[str, str] = field(default_factory=dict)
    position_columns: dict[str, str] = field(default_factory=dict)
    bar_columns: dict[str, str] = field(default_factory=dict)

    @property
    def url(self) -> str:
        if self.direct_url.strip():
            return self.direct_url.strip()
        if self.password == "请填写MySQL密码":
            value = os.getenv(self.url_env, "").strip()
            if value:
                return value
            raise RuntimeError("请先在 config.toml 的 database.password 中填写 MySQL 密码")
        username = quote_plus(self.username)
        password = quote_plus(self.password)
        credentials = username if not password else f"{username}:{password}"
        return (
            f"{self.driver}://{credentials}@{self.host}:{self.port}"
            "?charset=utf8mb4"
        )


@dataclass(frozen=True)
class PositionsConfig:
    active_status: str = "HOLDING"
    default_max_loss_pct: float = 0.05


@dataclass(frozen=True)
class PerformanceConfig:
    workers: int = 10


@dataclass(frozen=True)
class DataSyncConfig:
    source: str = "auto"
    start_date: str = "2021-01-01"
    batch_size: int = 2000
    retry_count: int = 6
    retry_delay_seconds: float = 3.0
    socket_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class TrainingConfig:
    lookback_days: int = 1200
    validation_days: int = 120
    min_symbol_rows: int = 80
    max_training_symbols: int = 1500
    similarity_sample_limit: int = 200_000
    random_seed: int = 42
    high_quantiles: tuple[float, ...] = (0.30, 0.50, 0.70)
    low_quantiles: tuple[float, ...] = (0.10, 0.20, 0.50)
    n_estimators: int = 700
    max_depth: int = 7
    learning_rate: float = 0.04
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    use_gpu: bool = True


@dataclass(frozen=True)
class RecommendationConfig:
    take_profit_quantile: float = 0.70
    stop_loss_quantile: float = 0.20
    similarity_top_k: int = 100
    atr_stop_multiplier: float = 1.5
    stop_limit_slippage_pct: float = 0.003
    minimum_stop_gap_pct: float = 0.050
    minimum_take_profit_pct: float = 0.008
    minimum_profit_over_cost_pct: float = 0.003
    maximum_take_profit_pct: float = 0.15
    minimum_risk_reward: float = 1.2
    minimum_take_profit_confidence: float = 0.70
    price_tick: float = 0.01
    default_daily_price_limit_pct: float = 0.0


@dataclass(frozen=True)
class BacktestConfig:
    max_symbols: int = 100
    similarity_query_sample_limit: int = 20_000
    similarity_batch_size: int = 128
    stop_order_type: str = "limit"
    fixed_take_profit_pct: float = 0.03
    fixed_stop_loss_pct: float = 0.02
    atr_take_profit_multiplier: float = 2.0
    atr_stop_loss_multiplier: float = 1.5


@dataclass(frozen=True)
class StopTuningConfig:
    max_symbols: int = 100
    holding_days: tuple[int, ...] = (20, 40, 60)
    atr_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
    limit_slippage_pcts: tuple[float, ...] = (0.003, 0.006, 0.010)
    minimum_stop_gap_pcts: tuple[float, ...] = (
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.050,
        0.070,
        0.100,
    )
    include_market_orders: bool = True
    entry_frequency: str = "monthly"


@dataclass(frozen=True)
class RiskFitConfig:
    enabled: bool = True
    max_symbols: int = 300
    holding_days: tuple[int, ...] = (20, 40, 60)
    candidate_stop_gap_pcts: tuple[float, ...] = (
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.050,
        0.070,
        0.100,
    )
    atr_multiplier: float = 1.5
    limit_slippage_pct: float = 0.003
    min_stop_gap_pct: float = 0.005
    max_stop_gap_pct: float = 0.100
    validation_months: int = 2
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    proxy_lambda_loss: float = 3.0
    proxy_lambda_whip: float = 0.05
    proxy_lambda_unfilled: float = 0.03
    proxy_lambda_drawdown: float = 0.0
    fixed_baseline_min_gap_pct: float = 0.0
    fixed_baseline_return_tolerance_pct: float = 0.002
    walk_forward_train_months: int = 12
    walk_forward_step_months: int = 2
    walk_forward_bootstrap_samples: int = 200
    entry_frequency: str = "monthly"


@dataclass(frozen=True)
class VolatilityStopConfig:
    enabled: bool = False
    lookback_days: int = 120
    k: float = 1.5
    k_values: tuple[float, ...] = (
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
    )
    shrink_n0: float = 120.0
    gap_min: float = 0.005
    gap_max: float = 0.100
    pool: str = "industry"
    adjustment_factor: float = 1.0
    adjustment_min: float = 0.8
    adjustment_max: float = 1.2
    max_symbols: int = 300
    holding_days: tuple[int, ...] = (20, 40, 60)
    validation_months: int = 3
    walk_forward_train_months: int = 12
    walk_forward_step_months: int = 3
    bootstrap_samples: int = 200
    shape_target_gap_pct: float = 0.0
    return_tolerance_pct: float = 0.010
    max_loss_tie_tolerance_pct: float = 0.002
    fixed_baseline_gap_pcts: tuple[float, ...] = (
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.050,
        0.070,
        0.100,
    )
    stop_order_type: str = "limit"
    limit_slippage_pct: float = 0.003
    entry_frequency: str = "monthly"


@dataclass(frozen=True)
class ExitMachineConfig:
    enabled: bool = False
    max_symbols: int = 300
    holding_days: tuple[int, ...] = (20, 40, 60)
    n_init: int = 8
    progress_r: float = 1.0
    atr_initial_mult: float = 1.5
    entry_structure_buffer: float = 0.3
    chandelier_mult: float = 3.0
    swing_trend_buffer: float = 0.3
    breakdown_ma: int = 20
    breakdown_buffer: float = 0.2
    ma_fast: int = 10
    profit_tier_1_gain: float = 0.08
    profit_tier_1_floor_pct: float = 0.005
    profit_tier_2_gain: float = 0.15
    profit_tier_2_capture: float = 0.30
    profit_tier_3_gain: float = 0.30
    profit_tier_3_capture: float = 0.45
    swing_pivot_k: int = 2
    r_floor_atr_mult: float = 0.5
    lookahead_extend_days: int = 20
    fixed_baseline_gap: float = 0.02
    hard_constraint_tolerance_pct: float = 0.005
    sell_fly_days: tuple[int, ...] = (5, 10, 20)
    sell_fly_thresholds: tuple[float, ...] = (0.05, 0.10)
    primary_sell_fly_days: int = 10
    primary_sell_fly_threshold: float = 0.05
    winner_peak_gain: float = 0.15
    trend_capture_ratio: float = 0.5
    walk_forward_train_months: int = 12
    validation_months: int = 3
    walk_forward_step_months: int = 3
    bootstrap_samples: int = 200
    stop_order_type: str = "limit"
    limit_slippage_pct: float = 0.003
    entry_frequency: str = "monthly"


@dataclass(frozen=True)
class ChartExitConfig:
    """Research-only chart-aware holding and re-entry backtest settings."""

    enabled: bool = False
    max_symbols: int = 300
    holding_days: tuple[int, ...] = (20, 40, 60)
    lookahead_extend_days: int = 20
    initial_days: int = 8
    progress_r: float = 1.0
    initial_min_gap_pct: float = 0.025
    entry_structure_buffer: float = 0.3
    r_floor_atr_mult: float = 0.5
    ma_fast: int = 10
    ma_trend: int = 20
    ma_long: int = 60
    ma_slope_lookback: int = 5
    breakdown_confirm_closes: int = 2
    swing_pivot_k: int = 2
    profit_tier_1_gain: float = 0.08
    profit_tier_1_floor_pct: float = 0.005
    profit_tier_2_gain: float = 0.15
    profit_tier_2_capture: float = 0.50
    profit_tier_3_gain: float = 0.30
    profit_tier_3_capture: float = 0.60
    reentry_cooldown_days: int = 2
    reentry_breakout_lookback: int = 10
    max_entries_safety: int = 3
    max_reentry_risk_pct: float = 0.05
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    market_slippage_pct: float = 0.001
    stop_order_type: str = "limit"
    limit_slippage_pct: float = 0.003
    hard_constraint_tolerance_pct: float = 0.005
    winner_peak_gain: float = 0.15
    trend_capture_ratio: float = 0.5
    sell_fly_days: tuple[int, ...] = (5, 10, 20)
    sell_fly_thresholds: tuple[float, ...] = (0.05, 0.10)
    walk_forward_train_months: int = 12
    validation_months: int = 3
    walk_forward_step_months: int = 3
    bootstrap_samples: int = 200
    entry_frequency: str = "monthly"


@dataclass(frozen=True)
class AppConfig:
    database: DatabaseConfig
    positions: PositionsConfig
    performance: PerformanceConfig
    data_sync: DataSyncConfig
    training: TrainingConfig
    recommendation: RecommendationConfig
    backtest: BacktestConfig
    stop_tuning: StopTuningConfig
    risk_fit: RiskFitConfig
    volatility_stop: VolatilityStopConfig
    exit_machine: ExitMachineConfig
    chart_exit: ChartExitConfig
    artifacts_directory: Path
    config_path: Path


DEFAULT_TABLES = {
    "positions": "stock_positions",
    "bars": "stock_daily_bars",
    "recommendations": "tpsl_recommendations",
}

DEFAULT_POSITION_COLUMNS = {
    "symbol": "symbol",
    "quantity": "quantity",
    "avg_cost": "avg_cost",
    "entry_date": "entry_date",
    "max_loss_pct": "max_loss_pct",
    "current_stop": "current_stop",
    "status": "status",
}

DEFAULT_BAR_COLUMNS = {
    "symbol": "symbol",
    "trade_date": "trade_date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "industry": "industry_code",
    "adj_factor": "adj_factor",
    "is_suspended": "is_suspended",
}


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"配置段 [{name}] 必须是对象")
    return value


def _as_tuple(values: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if values is None:
        return default
    return tuple(float(value) for value in values)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    db_raw = _section(raw, "database")
    tables = {**DEFAULT_TABLES, **_section(db_raw, "tables")}
    position_columns = {
        **DEFAULT_POSITION_COLUMNS,
        **_section(db_raw, "position_columns"),
    }
    bar_columns = {**DEFAULT_BAR_COLUMNS, **_section(db_raw, "bar_columns")}

    database = DatabaseConfig(
        direct_url=str(db_raw.get("url", "")),
        url_env=str(db_raw.get("url_env", "MYSQL_URL")),
        driver=str(db_raw.get("driver", "mysql+pymysql")),
        host=str(db_raw.get("host", "127.0.0.1")),
        port=int(db_raw.get("port", 3306)),
        username=str(db_raw.get("username", "root")),
        password=str(db_raw.get("password", "")),
        name=str(db_raw.get("name", "take_profit_stop_loss")),
        pool_size=int(db_raw.get("pool_size", 10)),
        max_overflow=int(db_raw.get("max_overflow", 5)),
        tables=tables,
        position_columns=position_columns,
        bar_columns=bar_columns,
    )
    if not database.name or not database.name.replace("_", "").isalnum():
        raise ValueError("database.name 只能包含字母、数字和下划线")
    if not 1 <= database.port <= 65535:
        raise ValueError("database.port 必须在 1 到 65535 之间")

    positions_raw = _section(raw, "positions")
    positions = PositionsConfig(
        active_status=str(positions_raw.get("active_status", "HOLDING")),
        default_max_loss_pct=float(
            positions_raw.get("default_max_loss_pct", 0.05)
        ),
    )

    performance_raw = _section(raw, "performance")
    workers = int(performance_raw.get("workers", 10))
    if workers < 1:
        raise ValueError("performance.workers 必须大于等于 1")
    performance = PerformanceConfig(workers=workers)

    data_sync_raw = _section(raw, "data_sync")
    data_sync_defaults = DataSyncConfig()
    data_sync = DataSyncConfig(
        source=str(data_sync_raw.get("source", data_sync_defaults.source)),
        start_date=str(
            data_sync_raw.get("start_date", data_sync_defaults.start_date)
        ),
        batch_size=int(
            data_sync_raw.get("batch_size", data_sync_defaults.batch_size)
        ),
        retry_count=int(
            data_sync_raw.get("retry_count", data_sync_defaults.retry_count)
        ),
        retry_delay_seconds=float(
            data_sync_raw.get(
                "retry_delay_seconds",
                data_sync_defaults.retry_delay_seconds,
            )
        ),
        socket_timeout_seconds=float(
            data_sync_raw.get(
                "socket_timeout_seconds",
                data_sync_defaults.socket_timeout_seconds,
            )
        ),
    )
    if data_sync.source not in {"baostock", "akshare", "auto"}:
        raise ValueError("data_sync.source 只支持 baostock、akshare 或 auto")
    if data_sync.batch_size < 1:
        raise ValueError("data_sync.batch_size 必须大于等于 1")
    if data_sync.retry_count < 1:
        raise ValueError("data_sync.retry_count 必须大于等于 1")
    if data_sync.socket_timeout_seconds <= 0:
        raise ValueError("data_sync.socket_timeout_seconds 必须大于 0")

    training_raw = _section(raw, "training")
    defaults = TrainingConfig()
    training = TrainingConfig(
        lookback_days=int(training_raw.get("lookback_days", defaults.lookback_days)),
        validation_days=int(
            training_raw.get("validation_days", defaults.validation_days)
        ),
        min_symbol_rows=int(
            training_raw.get("min_symbol_rows", defaults.min_symbol_rows)
        ),
        max_training_symbols=int(
            training_raw.get(
                "max_training_symbols",
                defaults.max_training_symbols,
            )
        ),
        similarity_sample_limit=int(
            training_raw.get(
                "similarity_sample_limit", defaults.similarity_sample_limit
            )
        ),
        random_seed=int(training_raw.get("random_seed", defaults.random_seed)),
        high_quantiles=_as_tuple(
            training_raw.get("high_quantiles"), defaults.high_quantiles
        ),
        low_quantiles=_as_tuple(
            training_raw.get("low_quantiles"), defaults.low_quantiles
        ),
        n_estimators=int(
            training_raw.get("n_estimators", defaults.n_estimators)
        ),
        max_depth=int(training_raw.get("max_depth", defaults.max_depth)),
        learning_rate=float(
            training_raw.get("learning_rate", defaults.learning_rate)
        ),
        subsample=float(training_raw.get("subsample", defaults.subsample)),
        colsample_bytree=float(
            training_raw.get("colsample_bytree", defaults.colsample_bytree)
        ),
        use_gpu=bool(training_raw.get("use_gpu", defaults.use_gpu)),
    )

    recommendation_raw = _section(raw, "recommendation")
    rec_defaults = RecommendationConfig()
    recommendation = RecommendationConfig(
        take_profit_quantile=float(
            recommendation_raw.get(
                "take_profit_quantile", rec_defaults.take_profit_quantile
            )
        ),
        stop_loss_quantile=float(
            recommendation_raw.get(
                "stop_loss_quantile", rec_defaults.stop_loss_quantile
            )
        ),
        similarity_top_k=int(
            recommendation_raw.get(
                "similarity_top_k", rec_defaults.similarity_top_k
            )
        ),
        atr_stop_multiplier=float(
            recommendation_raw.get(
                "atr_stop_multiplier", rec_defaults.atr_stop_multiplier
            )
        ),
        stop_limit_slippage_pct=float(
            recommendation_raw.get(
                "stop_limit_slippage_pct", rec_defaults.stop_limit_slippage_pct
            )
        ),
        minimum_stop_gap_pct=float(
            recommendation_raw.get(
                "minimum_stop_gap_pct", rec_defaults.minimum_stop_gap_pct
            )
        ),
        minimum_take_profit_pct=float(
            recommendation_raw.get(
                "minimum_take_profit_pct", rec_defaults.minimum_take_profit_pct
            )
        ),
        minimum_profit_over_cost_pct=float(
            recommendation_raw.get(
                "minimum_profit_over_cost_pct",
                rec_defaults.minimum_profit_over_cost_pct,
            )
        ),
        maximum_take_profit_pct=float(
            recommendation_raw.get(
                "maximum_take_profit_pct", rec_defaults.maximum_take_profit_pct
            )
        ),
        minimum_risk_reward=float(
            recommendation_raw.get(
                "minimum_risk_reward", rec_defaults.minimum_risk_reward
            )
        ),
        minimum_take_profit_confidence=float(
            recommendation_raw.get(
                "minimum_take_profit_confidence",
                rec_defaults.minimum_take_profit_confidence,
            )
        ),
        price_tick=float(
            recommendation_raw.get("price_tick", rec_defaults.price_tick)
        ),
        default_daily_price_limit_pct=float(
            recommendation_raw.get(
                "default_daily_price_limit_pct",
                rec_defaults.default_daily_price_limit_pct,
            )
        ),
    )
    if training.max_training_symbols < 1:
        raise ValueError("training.max_training_symbols 必须大于等于 1")
    all_quantiles = training.high_quantiles + training.low_quantiles
    if any(not 0 < value < 1 for value in all_quantiles):
        raise ValueError("training 中所有分位数必须在 0 和 1 之间")
    if recommendation.take_profit_quantile not in training.high_quantiles:
        raise ValueError(
            "recommendation.take_profit_quantile 必须包含在 "
            "training.high_quantiles 中"
        )
    if recommendation.stop_loss_quantile not in training.low_quantiles:
        raise ValueError(
            "recommendation.stop_loss_quantile 必须包含在 "
            "training.low_quantiles 中"
        )
    if recommendation.price_tick <= 0:
        raise ValueError("recommendation.price_tick 必须大于 0")
    if recommendation.minimum_profit_over_cost_pct < 0:
        raise ValueError(
            "recommendation.minimum_profit_over_cost_pct 必须大于等于 0"
        )
    if not 0 <= recommendation.minimum_take_profit_confidence <= 1:
        raise ValueError(
            "recommendation.minimum_take_profit_confidence 必须在 0 和 1 之间"
        )
    if not 0 < positions.default_max_loss_pct < 1:
        raise ValueError("positions.default_max_loss_pct 必须在 0 和 1 之间")

    backtest_raw = _section(raw, "backtest")
    backtest_defaults = BacktestConfig()
    backtest = BacktestConfig(
        max_symbols=int(
            backtest_raw.get("max_symbols", backtest_defaults.max_symbols)
        ),
        similarity_query_sample_limit=int(
            backtest_raw.get(
                "similarity_query_sample_limit",
                backtest_defaults.similarity_query_sample_limit,
            )
        ),
        similarity_batch_size=int(
            backtest_raw.get(
                "similarity_batch_size",
                backtest_defaults.similarity_batch_size,
            )
        ),
        stop_order_type=str(
            backtest_raw.get(
                "stop_order_type",
                backtest_defaults.stop_order_type,
            )
        ),
        fixed_take_profit_pct=float(
            backtest_raw.get(
                "fixed_take_profit_pct",
                backtest_defaults.fixed_take_profit_pct,
            )
        ),
        fixed_stop_loss_pct=float(
            backtest_raw.get(
                "fixed_stop_loss_pct",
                backtest_defaults.fixed_stop_loss_pct,
            )
        ),
        atr_take_profit_multiplier=float(
            backtest_raw.get(
                "atr_take_profit_multiplier",
                backtest_defaults.atr_take_profit_multiplier,
            )
        ),
        atr_stop_loss_multiplier=float(
            backtest_raw.get(
                "atr_stop_loss_multiplier",
                backtest_defaults.atr_stop_loss_multiplier,
            )
        ),
    )
    if backtest.max_symbols < 1:
        raise ValueError("backtest.max_symbols 必须大于等于 1")
    if backtest.similarity_query_sample_limit < 1:
        raise ValueError("backtest.similarity_query_sample_limit 必须大于等于 1")
    if backtest.similarity_batch_size < 1:
        raise ValueError("backtest.similarity_batch_size 必须大于等于 1")
    if backtest.stop_order_type not in {"market", "limit"}:
        raise ValueError("backtest.stop_order_type 必须是 market 或 limit")

    stop_tuning_raw = _section(raw, "stop_tuning")
    stop_tuning_defaults = StopTuningConfig()
    stop_tuning = StopTuningConfig(
        max_symbols=int(
            stop_tuning_raw.get(
                "max_symbols",
                stop_tuning_defaults.max_symbols,
            )
        ),
        holding_days=tuple(
            int(value)
            for value in stop_tuning_raw.get(
                "holding_days",
                stop_tuning_defaults.holding_days,
            )
        ),
        atr_multipliers=tuple(
            float(value)
            for value in stop_tuning_raw.get(
                "atr_multipliers",
                stop_tuning_defaults.atr_multipliers,
            )
        ),
        limit_slippage_pcts=tuple(
            float(value)
            for value in stop_tuning_raw.get(
                "limit_slippage_pcts",
                stop_tuning_defaults.limit_slippage_pcts,
            )
        ),
        minimum_stop_gap_pcts=tuple(
            float(value)
            for value in stop_tuning_raw.get(
                "minimum_stop_gap_pcts",
                stop_tuning_defaults.minimum_stop_gap_pcts,
            )
        ),
        include_market_orders=bool(
            stop_tuning_raw.get(
                "include_market_orders",
                stop_tuning_defaults.include_market_orders,
            )
        ),
        entry_frequency=str(
            stop_tuning_raw.get(
                "entry_frequency",
                stop_tuning_defaults.entry_frequency,
            )
        ),
    )
    if stop_tuning.max_symbols < 1:
        raise ValueError("stop_tuning.max_symbols 必须大于等于 1")
    if not stop_tuning.holding_days or any(
        value < 1 for value in stop_tuning.holding_days
    ):
        raise ValueError("stop_tuning.holding_days 必须是正整数")
    if not stop_tuning.atr_multipliers or any(
        value <= 0 for value in stop_tuning.atr_multipliers
    ):
        raise ValueError("stop_tuning.atr_multipliers 必须大于 0")
    if not stop_tuning.limit_slippage_pcts or any(
        not 0 <= value < 1
        for value in stop_tuning.limit_slippage_pcts
    ):
        raise ValueError(
            "stop_tuning.limit_slippage_pcts 必须在 0 和 1 之间"
        )
    if not stop_tuning.minimum_stop_gap_pcts or any(
        not 0 < value < 1
        for value in stop_tuning.minimum_stop_gap_pcts
    ):
        raise ValueError(
            "stop_tuning.minimum_stop_gap_pcts 必须在 0 和 1 之间"
        )
    if stop_tuning.entry_frequency != "monthly":
        raise ValueError("stop_tuning.entry_frequency 当前只支持 monthly")

    risk_fit_raw = _section(raw, "risk_fit")
    risk_fit_defaults = RiskFitConfig()
    risk_fit = RiskFitConfig(
        enabled=bool(risk_fit_raw.get("enabled", risk_fit_defaults.enabled)),
        max_symbols=int(
            risk_fit_raw.get("max_symbols", risk_fit_defaults.max_symbols)
        ),
        holding_days=tuple(
            int(value)
            for value in risk_fit_raw.get(
                "holding_days",
                risk_fit_defaults.holding_days,
            )
        ),
        candidate_stop_gap_pcts=tuple(
            float(value)
            for value in risk_fit_raw.get(
                "candidate_stop_gap_pcts",
                risk_fit_defaults.candidate_stop_gap_pcts,
            )
        ),
        atr_multiplier=float(
            risk_fit_raw.get(
                "atr_multiplier",
                risk_fit_defaults.atr_multiplier,
            )
        ),
        limit_slippage_pct=float(
            risk_fit_raw.get(
                "limit_slippage_pct",
                risk_fit_defaults.limit_slippage_pct,
            )
        ),
        min_stop_gap_pct=float(
            risk_fit_raw.get(
                "min_stop_gap_pct",
                risk_fit_defaults.min_stop_gap_pct,
            )
        ),
        max_stop_gap_pct=float(
            risk_fit_raw.get(
                "max_stop_gap_pct",
                risk_fit_defaults.max_stop_gap_pct,
            )
        ),
        validation_months=int(
            risk_fit_raw.get(
                "validation_months",
                risk_fit_defaults.validation_months,
            )
        ),
        n_estimators=int(
            risk_fit_raw.get("n_estimators", risk_fit_defaults.n_estimators)
        ),
        max_depth=int(
            risk_fit_raw.get("max_depth", risk_fit_defaults.max_depth)
        ),
        learning_rate=float(
            risk_fit_raw.get(
                "learning_rate",
                risk_fit_defaults.learning_rate,
            )
        ),
        proxy_lambda_loss=float(
            risk_fit_raw.get(
                "proxy_lambda_loss",
                risk_fit_defaults.proxy_lambda_loss,
            )
        ),
        proxy_lambda_whip=float(
            risk_fit_raw.get(
                "proxy_lambda_whip",
                risk_fit_defaults.proxy_lambda_whip,
            )
        ),
        proxy_lambda_unfilled=float(
            risk_fit_raw.get(
                "proxy_lambda_unfilled",
                risk_fit_defaults.proxy_lambda_unfilled,
            )
        ),
        proxy_lambda_drawdown=float(
            risk_fit_raw.get(
                "proxy_lambda_drawdown",
                risk_fit_defaults.proxy_lambda_drawdown,
            )
        ),
        fixed_baseline_min_gap_pct=float(
            risk_fit_raw.get(
                "fixed_baseline_min_gap_pct",
                risk_fit_defaults.fixed_baseline_min_gap_pct,
            )
        ),
        fixed_baseline_return_tolerance_pct=float(
            risk_fit_raw.get(
                "fixed_baseline_return_tolerance_pct",
                risk_fit_defaults.fixed_baseline_return_tolerance_pct,
            )
        ),
        walk_forward_train_months=int(
            risk_fit_raw.get(
                "walk_forward_train_months",
                risk_fit_defaults.walk_forward_train_months,
            )
        ),
        walk_forward_step_months=int(
            risk_fit_raw.get(
                "walk_forward_step_months",
                risk_fit_defaults.walk_forward_step_months,
            )
        ),
        walk_forward_bootstrap_samples=int(
            risk_fit_raw.get(
                "walk_forward_bootstrap_samples",
                risk_fit_defaults.walk_forward_bootstrap_samples,
            )
        ),
        entry_frequency=str(
            risk_fit_raw.get(
                "entry_frequency",
                risk_fit_defaults.entry_frequency,
            )
        ),
    )
    if risk_fit.max_symbols < 1:
        raise ValueError("risk_fit.max_symbols 必须大于等于 1")
    if not risk_fit.holding_days or any(value < 1 for value in risk_fit.holding_days):
        raise ValueError("risk_fit.holding_days 必须是正整数")
    if not risk_fit.candidate_stop_gap_pcts or any(
        not 0 < value < 1 for value in risk_fit.candidate_stop_gap_pcts
    ):
        raise ValueError("risk_fit.candidate_stop_gap_pcts 必须在 0 和 1 之间")
    if not 0 < risk_fit.min_stop_gap_pct <= risk_fit.max_stop_gap_pct < 1:
        raise ValueError("risk_fit.min_stop_gap_pct/max_stop_gap_pct 范围不合法")
    if risk_fit.atr_multiplier <= 0:
        raise ValueError("risk_fit.atr_multiplier 必须大于 0")
    if not 0 <= risk_fit.limit_slippage_pct < 1:
        raise ValueError("risk_fit.limit_slippage_pct 必须在 0 和 1 之间")
    if risk_fit.validation_months < 1:
        raise ValueError("risk_fit.validation_months 必须大于等于 1")
    if risk_fit.n_estimators < 1 or risk_fit.max_depth < 1:
        raise ValueError("risk_fit.n_estimators/max_depth 必须大于等于 1")
    if risk_fit.learning_rate <= 0:
        raise ValueError("risk_fit.learning_rate 必须大于 0")
    if risk_fit.proxy_lambda_loss < 0:
        raise ValueError("risk_fit.proxy_lambda_loss 必须大于等于 0")
    if risk_fit.proxy_lambda_whip < 0 or risk_fit.proxy_lambda_unfilled < 0:
        raise ValueError("risk_fit proxy whipsaw/unfilled 权重必须大于等于 0")
    if risk_fit.fixed_baseline_min_gap_pct < 0:
        raise ValueError("risk_fit.fixed_baseline_min_gap_pct 必须大于等于 0")
    if risk_fit.fixed_baseline_return_tolerance_pct < 0:
        raise ValueError(
            "risk_fit.fixed_baseline_return_tolerance_pct 必须大于等于 0"
        )
    if risk_fit.walk_forward_train_months < 1:
        raise ValueError("risk_fit.walk_forward_train_months 必须大于等于 1")
    if risk_fit.walk_forward_step_months < 1:
        raise ValueError("risk_fit.walk_forward_step_months 必须大于等于 1")
    if risk_fit.walk_forward_bootstrap_samples < 0:
        raise ValueError("risk_fit.walk_forward_bootstrap_samples 必须大于等于 0")
    if risk_fit.entry_frequency != "monthly":
        raise ValueError("risk_fit.entry_frequency 当前只支持 monthly")

    volatility_raw = _section(raw, "volatility_stop")
    volatility_defaults = VolatilityStopConfig()
    volatility_stop = VolatilityStopConfig(
        enabled=bool(volatility_raw.get("enabled", volatility_defaults.enabled)),
        lookback_days=int(
            volatility_raw.get("lookback_days", volatility_defaults.lookback_days)
        ),
        k=float(volatility_raw.get("k", volatility_defaults.k)),
        k_values=tuple(
            float(value)
            for value in volatility_raw.get(
                "k_values",
                volatility_defaults.k_values,
            )
        ),
        shrink_n0=float(
            volatility_raw.get("shrink_n0", volatility_defaults.shrink_n0)
        ),
        gap_min=float(volatility_raw.get("gap_min", volatility_defaults.gap_min)),
        gap_max=float(volatility_raw.get("gap_max", volatility_defaults.gap_max)),
        pool=str(volatility_raw.get("pool", volatility_defaults.pool)),
        adjustment_factor=float(
            volatility_raw.get(
                "adjustment_factor",
                volatility_defaults.adjustment_factor,
            )
        ),
        adjustment_min=float(
            volatility_raw.get(
                "adjustment_min",
                volatility_defaults.adjustment_min,
            )
        ),
        adjustment_max=float(
            volatility_raw.get(
                "adjustment_max",
                volatility_defaults.adjustment_max,
            )
        ),
        max_symbols=int(
            volatility_raw.get("max_symbols", volatility_defaults.max_symbols)
        ),
        holding_days=tuple(
            int(value)
            for value in volatility_raw.get(
                "holding_days",
                volatility_defaults.holding_days,
            )
        ),
        validation_months=int(
            volatility_raw.get(
                "validation_months",
                volatility_defaults.validation_months,
            )
        ),
        walk_forward_train_months=int(
            volatility_raw.get(
                "walk_forward_train_months",
                volatility_defaults.walk_forward_train_months,
            )
        ),
        walk_forward_step_months=int(
            volatility_raw.get(
                "walk_forward_step_months",
                volatility_defaults.walk_forward_step_months,
            )
        ),
        bootstrap_samples=int(
            volatility_raw.get(
                "bootstrap_samples",
                volatility_defaults.bootstrap_samples,
            )
        ),
        shape_target_gap_pct=float(
            volatility_raw.get(
                "shape_target_gap_pct",
                volatility_defaults.shape_target_gap_pct,
            )
        ),
        return_tolerance_pct=float(
            volatility_raw.get(
                "return_tolerance_pct",
                volatility_defaults.return_tolerance_pct,
            )
        ),
        max_loss_tie_tolerance_pct=float(
            volatility_raw.get(
                "max_loss_tie_tolerance_pct",
                volatility_defaults.max_loss_tie_tolerance_pct,
            )
        ),
        fixed_baseline_gap_pcts=tuple(
            float(value)
            for value in volatility_raw.get(
                "fixed_baseline_gap_pcts",
                volatility_defaults.fixed_baseline_gap_pcts,
            )
        ),
        stop_order_type=str(
            volatility_raw.get(
                "stop_order_type",
                volatility_defaults.stop_order_type,
            )
        ),
        limit_slippage_pct=float(
            volatility_raw.get(
                "limit_slippage_pct",
                volatility_defaults.limit_slippage_pct,
            )
        ),
        entry_frequency=str(
            volatility_raw.get(
                "entry_frequency",
                volatility_defaults.entry_frequency,
            )
        ),
    )
    if volatility_stop.lookback_days < 1:
        raise ValueError("volatility_stop.lookback_days 必须大于等于 1")
    if volatility_stop.k <= 0 or not volatility_stop.k_values:
        raise ValueError("volatility_stop.k/k_values 必须为正数")
    if any(value <= 0 for value in volatility_stop.k_values):
        raise ValueError("volatility_stop.k_values 必须全为正数")
    if volatility_stop.shrink_n0 < 0:
        raise ValueError("volatility_stop.shrink_n0 必须大于等于 0")
    if not 0 < volatility_stop.gap_min <= volatility_stop.gap_max < 1:
        raise ValueError("volatility_stop.gap_min/gap_max 范围不合法")
    if volatility_stop.pool not in {"industry", "market"}:
        raise ValueError("volatility_stop.pool 只能是 industry 或 market")
    if not 0 < volatility_stop.adjustment_min <= volatility_stop.adjustment_factor <= volatility_stop.adjustment_max:
        raise ValueError("volatility_stop adjustment 范围不合法")
    if volatility_stop.max_symbols < 1:
        raise ValueError("volatility_stop.max_symbols 必须大于等于 1")
    if not volatility_stop.holding_days or any(
        value < 1 for value in volatility_stop.holding_days
    ):
        raise ValueError("volatility_stop.holding_days 必须是正整数")
    if volatility_stop.validation_months < 1:
        raise ValueError("volatility_stop.validation_months 必须大于等于 1")
    if volatility_stop.walk_forward_train_months < 1:
        raise ValueError("volatility_stop.walk_forward_train_months 必须大于等于 1")
    if volatility_stop.walk_forward_step_months < 1:
        raise ValueError("volatility_stop.walk_forward_step_months 必须大于等于 1")
    if volatility_stop.bootstrap_samples < 0:
        raise ValueError("volatility_stop.bootstrap_samples 必须大于等于 0")
    if volatility_stop.shape_target_gap_pct < 0:
        raise ValueError("volatility_stop.shape_target_gap_pct 必须大于等于 0")
    if volatility_stop.shape_target_gap_pct >= 1:
        raise ValueError("volatility_stop.shape_target_gap_pct 必须小于 1")
    if volatility_stop.return_tolerance_pct < 0:
        raise ValueError("volatility_stop.return_tolerance_pct 必须大于等于 0")
    if volatility_stop.max_loss_tie_tolerance_pct < 0:
        raise ValueError("volatility_stop.max_loss_tie_tolerance_pct 必须大于等于 0")
    if not volatility_stop.fixed_baseline_gap_pcts or any(
        not 0 < value < 1 for value in volatility_stop.fixed_baseline_gap_pcts
    ):
        raise ValueError("volatility_stop.fixed_baseline_gap_pcts 必须在 0 和 1 之间")
    if volatility_stop.stop_order_type not in {"market", "limit"}:
        raise ValueError("volatility_stop.stop_order_type 必须是 market 或 limit")
    if not 0 <= volatility_stop.limit_slippage_pct < 1:
        raise ValueError("volatility_stop.limit_slippage_pct 必须在 0 和 1 之间")
    if volatility_stop.entry_frequency != "monthly":
        raise ValueError("volatility_stop.entry_frequency 当前只支持 monthly")

    exit_raw = _section(raw, "exit_machine")
    exit_defaults = ExitMachineConfig()
    exit_machine = ExitMachineConfig(
        enabled=bool(exit_raw.get("enabled", exit_defaults.enabled)),
        max_symbols=int(exit_raw.get("max_symbols", exit_defaults.max_symbols)),
        holding_days=tuple(
            int(value)
            for value in exit_raw.get(
                "holding_days",
                exit_defaults.holding_days,
            )
        ),
        n_init=int(exit_raw.get("n_init", exit_defaults.n_init)),
        progress_r=float(exit_raw.get("progress_r", exit_defaults.progress_r)),
        atr_initial_mult=float(
            exit_raw.get("atr_initial_mult", exit_defaults.atr_initial_mult)
        ),
        entry_structure_buffer=float(
            exit_raw.get(
                "entry_structure_buffer",
                exit_defaults.entry_structure_buffer,
            )
        ),
        chandelier_mult=float(
            exit_raw.get("chandelier_mult", exit_defaults.chandelier_mult)
        ),
        swing_trend_buffer=float(
            exit_raw.get("swing_trend_buffer", exit_defaults.swing_trend_buffer)
        ),
        breakdown_ma=int(exit_raw.get("breakdown_ma", exit_defaults.breakdown_ma)),
        breakdown_buffer=float(
            exit_raw.get("breakdown_buffer", exit_defaults.breakdown_buffer)
        ),
        ma_fast=int(exit_raw.get("ma_fast", exit_defaults.ma_fast)),
        profit_tier_1_gain=float(
            exit_raw.get("profit_tier_1_gain", exit_defaults.profit_tier_1_gain)
        ),
        profit_tier_1_floor_pct=float(
            exit_raw.get(
                "profit_tier_1_floor_pct",
                exit_defaults.profit_tier_1_floor_pct,
            )
        ),
        profit_tier_2_gain=float(
            exit_raw.get("profit_tier_2_gain", exit_defaults.profit_tier_2_gain)
        ),
        profit_tier_2_capture=float(
            exit_raw.get(
                "profit_tier_2_capture",
                exit_defaults.profit_tier_2_capture,
            )
        ),
        profit_tier_3_gain=float(
            exit_raw.get("profit_tier_3_gain", exit_defaults.profit_tier_3_gain)
        ),
        profit_tier_3_capture=float(
            exit_raw.get(
                "profit_tier_3_capture",
                exit_defaults.profit_tier_3_capture,
            )
        ),
        swing_pivot_k=int(exit_raw.get("swing_pivot_k", exit_defaults.swing_pivot_k)),
        r_floor_atr_mult=float(
            exit_raw.get("r_floor_atr_mult", exit_defaults.r_floor_atr_mult)
        ),
        lookahead_extend_days=int(
            exit_raw.get(
                "lookahead_extend_days",
                exit_defaults.lookahead_extend_days,
            )
        ),
        fixed_baseline_gap=float(
            exit_raw.get("fixed_baseline_gap", exit_defaults.fixed_baseline_gap)
        ),
        hard_constraint_tolerance_pct=float(
            exit_raw.get(
                "hard_constraint_tolerance_pct",
                exit_defaults.hard_constraint_tolerance_pct,
            )
        ),
        sell_fly_days=tuple(
            int(value)
            for value in exit_raw.get(
                "sell_fly_days",
                exit_defaults.sell_fly_days,
            )
        ),
        sell_fly_thresholds=tuple(
            float(value)
            for value in exit_raw.get(
                "sell_fly_thresholds",
                exit_defaults.sell_fly_thresholds,
            )
        ),
        primary_sell_fly_days=int(
            exit_raw.get(
                "primary_sell_fly_days",
                exit_defaults.primary_sell_fly_days,
            )
        ),
        primary_sell_fly_threshold=float(
            exit_raw.get(
                "primary_sell_fly_threshold",
                exit_defaults.primary_sell_fly_threshold,
            )
        ),
        winner_peak_gain=float(
            exit_raw.get("winner_peak_gain", exit_defaults.winner_peak_gain)
        ),
        trend_capture_ratio=float(
            exit_raw.get(
                "trend_capture_ratio",
                exit_defaults.trend_capture_ratio,
            )
        ),
        walk_forward_train_months=int(
            exit_raw.get(
                "walk_forward_train_months",
                exit_defaults.walk_forward_train_months,
            )
        ),
        validation_months=int(
            exit_raw.get("validation_months", exit_defaults.validation_months)
        ),
        walk_forward_step_months=int(
            exit_raw.get(
                "walk_forward_step_months",
                exit_defaults.walk_forward_step_months,
            )
        ),
        bootstrap_samples=int(
            exit_raw.get("bootstrap_samples", exit_defaults.bootstrap_samples)
        ),
        stop_order_type=str(
            exit_raw.get("stop_order_type", exit_defaults.stop_order_type)
        ),
        limit_slippage_pct=float(
            exit_raw.get(
                "limit_slippage_pct",
                exit_defaults.limit_slippage_pct,
            )
        ),
        entry_frequency=str(
            exit_raw.get("entry_frequency", exit_defaults.entry_frequency)
        ),
    )
    if exit_machine.max_symbols < 1:
        raise ValueError("exit_machine.max_symbols 必须大于等于 1")
    if not exit_machine.holding_days or any(value < 1 for value in exit_machine.holding_days):
        raise ValueError("exit_machine.holding_days 必须是正整数")
    if exit_machine.n_init < 1:
        raise ValueError("exit_machine.n_init 必须大于等于 1")
    positive_values = (
        exit_machine.progress_r,
        exit_machine.atr_initial_mult,
        exit_machine.chandelier_mult,
        exit_machine.r_floor_atr_mult,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("exit_machine 关键倍数必须大于 0")
    if exit_machine.entry_structure_buffer < 0 or exit_machine.swing_trend_buffer < 0:
        raise ValueError("exit_machine swing buffer 必须大于等于 0")
    if exit_machine.breakdown_ma < 1 or exit_machine.ma_fast < 1:
        raise ValueError("exit_machine 均线窗口必须大于等于 1")
    if exit_machine.swing_pivot_k < 1:
        raise ValueError("exit_machine.swing_pivot_k 必须大于等于 1")
    if exit_machine.lookahead_extend_days < 0:
        raise ValueError("exit_machine.lookahead_extend_days 必须大于等于 0")
    if not 0 < exit_machine.fixed_baseline_gap < 1:
        raise ValueError("exit_machine.fixed_baseline_gap 必须在 0 和 1 之间")
    if exit_machine.hard_constraint_tolerance_pct < 0:
        raise ValueError("exit_machine.hard_constraint_tolerance_pct 必须大于等于 0")
    if not exit_machine.sell_fly_days or any(value < 1 for value in exit_machine.sell_fly_days):
        raise ValueError("exit_machine.sell_fly_days 必须是正整数")
    if not exit_machine.sell_fly_thresholds or any(
        not 0 < value < 1 for value in exit_machine.sell_fly_thresholds
    ):
        raise ValueError("exit_machine.sell_fly_thresholds 必须在 0 和 1 之间")
    if exit_machine.primary_sell_fly_days not in exit_machine.sell_fly_days:
        raise ValueError("exit_machine.primary_sell_fly_days 必须包含在 sell_fly_days 中")
    if exit_machine.primary_sell_fly_threshold not in exit_machine.sell_fly_thresholds:
        raise ValueError("exit_machine.primary_sell_fly_threshold 必须包含在 sell_fly_thresholds 中")
    if not 0 < exit_machine.winner_peak_gain < 1:
        raise ValueError("exit_machine.winner_peak_gain 必须在 0 和 1 之间")
    if not 0 < exit_machine.trend_capture_ratio <= 1:
        raise ValueError("exit_machine.trend_capture_ratio 必须在 0 和 1 之间")
    if exit_machine.walk_forward_train_months < 1:
        raise ValueError("exit_machine.walk_forward_train_months 必须大于等于 1")
    if exit_machine.validation_months < 1:
        raise ValueError("exit_machine.validation_months 必须大于等于 1")
    if exit_machine.walk_forward_step_months < 1:
        raise ValueError("exit_machine.walk_forward_step_months 必须大于等于 1")
    if exit_machine.bootstrap_samples < 0:
        raise ValueError("exit_machine.bootstrap_samples 必须大于等于 0")
    if exit_machine.stop_order_type not in {"market", "limit"}:
        raise ValueError("exit_machine.stop_order_type 必须是 market 或 limit")
    if not 0 <= exit_machine.limit_slippage_pct < 1:
        raise ValueError("exit_machine.limit_slippage_pct 必须在 0 和 1 之间")
    if exit_machine.entry_frequency != "monthly":
        raise ValueError("exit_machine.entry_frequency 当前只支持 monthly")

    chart_raw = _section(raw, "chart_exit")
    chart_defaults = ChartExitConfig()
    chart_exit = ChartExitConfig(
        enabled=bool(chart_raw.get("enabled", chart_defaults.enabled)),
        max_symbols=int(chart_raw.get("max_symbols", chart_defaults.max_symbols)),
        holding_days=tuple(
            int(value)
            for value in chart_raw.get("holding_days", chart_defaults.holding_days)
        ),
        lookahead_extend_days=int(
            chart_raw.get("lookahead_extend_days", chart_defaults.lookahead_extend_days)
        ),
        initial_days=int(chart_raw.get("initial_days", chart_defaults.initial_days)),
        progress_r=float(chart_raw.get("progress_r", chart_defaults.progress_r)),
        initial_min_gap_pct=float(
            chart_raw.get("initial_min_gap_pct", chart_defaults.initial_min_gap_pct)
        ),
        entry_structure_buffer=float(
            chart_raw.get("entry_structure_buffer", chart_defaults.entry_structure_buffer)
        ),
        r_floor_atr_mult=float(
            chart_raw.get("r_floor_atr_mult", chart_defaults.r_floor_atr_mult)
        ),
        ma_fast=int(chart_raw.get("ma_fast", chart_defaults.ma_fast)),
        ma_trend=int(chart_raw.get("ma_trend", chart_defaults.ma_trend)),
        ma_long=int(chart_raw.get("ma_long", chart_defaults.ma_long)),
        ma_slope_lookback=int(
            chart_raw.get("ma_slope_lookback", chart_defaults.ma_slope_lookback)
        ),
        breakdown_confirm_closes=int(
            chart_raw.get(
                "breakdown_confirm_closes",
                chart_defaults.breakdown_confirm_closes,
            )
        ),
        swing_pivot_k=int(
            chart_raw.get("swing_pivot_k", chart_defaults.swing_pivot_k)
        ),
        profit_tier_1_gain=float(
            chart_raw.get("profit_tier_1_gain", chart_defaults.profit_tier_1_gain)
        ),
        profit_tier_1_floor_pct=float(
            chart_raw.get(
                "profit_tier_1_floor_pct",
                chart_defaults.profit_tier_1_floor_pct,
            )
        ),
        profit_tier_2_gain=float(
            chart_raw.get("profit_tier_2_gain", chart_defaults.profit_tier_2_gain)
        ),
        profit_tier_2_capture=float(
            chart_raw.get("profit_tier_2_capture", chart_defaults.profit_tier_2_capture)
        ),
        profit_tier_3_gain=float(
            chart_raw.get("profit_tier_3_gain", chart_defaults.profit_tier_3_gain)
        ),
        profit_tier_3_capture=float(
            chart_raw.get("profit_tier_3_capture", chart_defaults.profit_tier_3_capture)
        ),
        reentry_cooldown_days=int(
            chart_raw.get("reentry_cooldown_days", chart_defaults.reentry_cooldown_days)
        ),
        reentry_breakout_lookback=int(
            chart_raw.get(
                "reentry_breakout_lookback",
                chart_defaults.reentry_breakout_lookback,
            )
        ),
        max_entries_safety=int(
            chart_raw.get("max_entries_safety", chart_defaults.max_entries_safety)
        ),
        max_reentry_risk_pct=float(
            chart_raw.get("max_reentry_risk_pct", chart_defaults.max_reentry_risk_pct)
        ),
        commission_rate=float(
            chart_raw.get("commission_rate", chart_defaults.commission_rate)
        ),
        stamp_tax_rate=float(
            chart_raw.get("stamp_tax_rate", chart_defaults.stamp_tax_rate)
        ),
        market_slippage_pct=float(
            chart_raw.get("market_slippage_pct", chart_defaults.market_slippage_pct)
        ),
        stop_order_type=str(
            chart_raw.get("stop_order_type", chart_defaults.stop_order_type)
        ),
        limit_slippage_pct=float(
            chart_raw.get("limit_slippage_pct", chart_defaults.limit_slippage_pct)
        ),
        hard_constraint_tolerance_pct=float(
            chart_raw.get(
                "hard_constraint_tolerance_pct",
                chart_defaults.hard_constraint_tolerance_pct,
            )
        ),
        winner_peak_gain=float(
            chart_raw.get("winner_peak_gain", chart_defaults.winner_peak_gain)
        ),
        trend_capture_ratio=float(
            chart_raw.get("trend_capture_ratio", chart_defaults.trend_capture_ratio)
        ),
        sell_fly_days=tuple(
            int(value)
            for value in chart_raw.get("sell_fly_days", chart_defaults.sell_fly_days)
        ),
        sell_fly_thresholds=tuple(
            float(value)
            for value in chart_raw.get(
                "sell_fly_thresholds",
                chart_defaults.sell_fly_thresholds,
            )
        ),
        walk_forward_train_months=int(
            chart_raw.get(
                "walk_forward_train_months",
                chart_defaults.walk_forward_train_months,
            )
        ),
        validation_months=int(
            chart_raw.get("validation_months", chart_defaults.validation_months)
        ),
        walk_forward_step_months=int(
            chart_raw.get(
                "walk_forward_step_months",
                chart_defaults.walk_forward_step_months,
            )
        ),
        bootstrap_samples=int(
            chart_raw.get("bootstrap_samples", chart_defaults.bootstrap_samples)
        ),
        entry_frequency=str(
            chart_raw.get("entry_frequency", chart_defaults.entry_frequency)
        ),
    )
    if chart_exit.max_symbols < 1:
        raise ValueError("chart_exit.max_symbols must be at least 1")
    if not chart_exit.holding_days or any(value < 1 for value in chart_exit.holding_days):
        raise ValueError("chart_exit.holding_days must contain positive integers")
    positive_integers = (
        chart_exit.initial_days,
        chart_exit.ma_fast,
        chart_exit.ma_trend,
        chart_exit.ma_long,
        chart_exit.ma_slope_lookback,
        chart_exit.breakdown_confirm_closes,
        chart_exit.swing_pivot_k,
        chart_exit.reentry_breakout_lookback,
        chart_exit.max_entries_safety,
        chart_exit.walk_forward_train_months,
        chart_exit.validation_months,
        chart_exit.walk_forward_step_months,
    )
    if any(value < 1 for value in positive_integers):
        raise ValueError("chart_exit integer windows must be positive")
    if chart_exit.reentry_cooldown_days < 0 or chart_exit.bootstrap_samples < 0:
        raise ValueError("chart_exit cooldown/bootstrap values must be non-negative")
    fraction_values = (
        chart_exit.initial_min_gap_pct,
        chart_exit.max_reentry_risk_pct,
        chart_exit.commission_rate,
        chart_exit.stamp_tax_rate,
        chart_exit.market_slippage_pct,
        chart_exit.limit_slippage_pct,
        chart_exit.hard_constraint_tolerance_pct,
        chart_exit.winner_peak_gain,
        chart_exit.trend_capture_ratio,
    )
    if any(not 0 <= value < 1 for value in fraction_values):
        raise ValueError("chart_exit percentage values must be between 0 and 1")
    if chart_exit.progress_r <= 0 or chart_exit.r_floor_atr_mult <= 0:
        raise ValueError("chart_exit progress/R settings must be positive")
    if chart_exit.entry_structure_buffer < 0:
        raise ValueError("chart_exit.entry_structure_buffer must be non-negative")
    if chart_exit.stop_order_type not in {"market", "limit"}:
        raise ValueError("chart_exit.stop_order_type must be market or limit")
    if not chart_exit.sell_fly_days or any(value < 1 for value in chart_exit.sell_fly_days):
        raise ValueError("chart_exit.sell_fly_days must contain positive integers")
    if chart_exit.lookahead_extend_days < max(chart_exit.sell_fly_days):
        raise ValueError("chart_exit.lookahead_extend_days must cover sell_fly_days")
    if not chart_exit.sell_fly_thresholds or any(
        not 0 < value < 1 for value in chart_exit.sell_fly_thresholds
    ):
        raise ValueError("chart_exit.sell_fly_thresholds must be between 0 and 1")
    if chart_exit.entry_frequency != "monthly":
        raise ValueError("chart_exit.entry_frequency currently supports monthly only")
    if not 0 < chart_exit.initial_min_gap_pct < 1:
        raise ValueError("chart_exit.initial_min_gap_pct must be between 0 and 1")
    if not (
        0 < chart_exit.profit_tier_1_gain
        < chart_exit.profit_tier_2_gain
        < chart_exit.profit_tier_3_gain
        < 1
    ):
        raise ValueError("chart_exit profit gain tiers must be strictly increasing")
    capture_values = (
        chart_exit.profit_tier_1_floor_pct,
        chart_exit.profit_tier_2_capture,
        chart_exit.profit_tier_3_capture,
    )
    if any(not 0 <= value <= 1 for value in capture_values):
        raise ValueError("chart_exit profit capture values must be between 0 and 1")
    if not chart_exit.ma_fast < chart_exit.ma_trend < chart_exit.ma_long:
        raise ValueError("chart_exit moving averages must satisfy fast < trend < long")

    artifacts_raw = _section(raw, "artifacts")
    artifacts_value = Path(str(artifacts_raw.get("directory", "artifacts")))
    artifacts_directory = (
        artifacts_value
        if artifacts_value.is_absolute()
        else (config_path.parent / artifacts_value)
    ).resolve()

    return AppConfig(
        database=database,
        positions=positions,
        performance=performance,
        data_sync=data_sync,
        training=training,
        recommendation=recommendation,
        backtest=backtest,
        stop_tuning=stop_tuning,
        risk_fit=risk_fit,
        volatility_stop=volatility_stop,
        exit_machine=exit_machine,
        chart_exit=chart_exit,
        artifacts_directory=artifacts_directory,
        config_path=config_path,
    )
