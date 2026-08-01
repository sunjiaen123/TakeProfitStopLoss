from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "gap_pct",
    "range_pct",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "volatility_5",
    "volatility_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "amount_ratio_20",
    "ma_distance_5",
    "ma_distance_10",
    "ma_distance_20",
    "atr_14_pct",
    "position_in_20",
]

CROSS_SECTION_FEATURES = [
    "industry_ret_1",
    "industry_up_ratio",
    "industry_volatility",
    "market_ret_1",
    "market_up_ratio",
    "market_volatility",
]

FEATURE_COLUMNS = BASE_FEATURES + CROSS_SECTION_FEATURES


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    frame = group.sort_values("trade_date").copy()
    close = frame["close"]
    previous_close = close.shift(1)
    returns = close.pct_change(fill_method=None)

    frame["ret_1"] = returns
    for window in (3, 5, 10, 20):
        frame[f"ret_{window}"] = close.pct_change(window, fill_method=None)

    frame["gap_pct"] = _safe_div(frame["open"], previous_close) - 1
    frame["range_pct"] = _safe_div(frame["high"] - frame["low"], previous_close)
    frame["body_pct"] = _safe_div(frame["close"] - frame["open"], previous_close)
    frame["upper_shadow_pct"] = _safe_div(
        frame["high"] - frame[["open", "close"]].max(axis=1),
        previous_close,
    )
    frame["lower_shadow_pct"] = _safe_div(
        frame[["open", "close"]].min(axis=1) - frame["low"],
        previous_close,
    )
    frame["volatility_5"] = returns.rolling(5).std()
    frame["volatility_20"] = returns.rolling(20).std()

    volume_mean_5 = frame["volume"].rolling(5).mean()
    volume_mean_20 = frame["volume"].rolling(20).mean()
    amount_mean_20 = frame["amount"].rolling(20).mean()
    frame["volume_ratio_5"] = _safe_div(frame["volume"], volume_mean_5)
    frame["volume_ratio_20"] = _safe_div(frame["volume"], volume_mean_20)
    frame["amount_ratio_20"] = _safe_div(frame["amount"], amount_mean_20)

    for window in (5, 10, 20):
        moving_average = close.rolling(window).mean()
        frame[f"ma_distance_{window}"] = _safe_div(close, moving_average) - 1

    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.rolling(14).mean()
    frame["atr_14_pct"] = _safe_div(frame["atr_14"], close)

    rolling_low = frame["low"].rolling(20).min()
    rolling_high = frame["high"].rolling(20).max()
    frame["position_in_20"] = _safe_div(
        close - rolling_low,
        rolling_high - rolling_low,
    )

    frame["next_high_return"] = _safe_div(frame["high"].shift(-1), close) - 1
    frame["next_low_return"] = _safe_div(frame["low"].shift(-1), close) - 1
    return frame


def _add_cross_section_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    industry_group = result.groupby(["trade_date", "industry"], sort=False)
    market_group = result.groupby("trade_date", sort=False)

    result["industry_ret_1"] = industry_group["ret_1"].transform("mean")
    result["industry_up_ratio"] = industry_group["ret_1"].transform(
        lambda values: (values > 0).mean()
    )
    result["industry_volatility"] = industry_group["ret_1"].transform("std")
    result["market_ret_1"] = market_group["ret_1"].transform("mean")
    result["market_up_ratio"] = market_group["ret_1"].transform(
        lambda values: (values > 0).mean()
    )
    result["market_volatility"] = market_group["ret_1"].transform("std")
    result["industry_volatility"] = result["industry_volatility"].fillna(0.0)
    result["market_volatility"] = result["market_volatility"].fillna(0.0)
    float_columns = FEATURE_COLUMNS + [
        "atr_14",
        "next_high_return",
        "next_low_return",
    ]
    result[float_columns] = result[float_columns].astype("float32")
    return result


def build_feature_frame(
    bars: pd.DataFrame,
    workers: int = 10,
    min_symbol_rows: int = 0,
) -> pd.DataFrame:
    required = {
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "industry",
    }
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"日线数据缺少字段：{sorted(missing)}")
    if bars.empty:
        return bars.copy()

    groups = [
        group.copy()
        for _, group in bars.groupby("symbol", sort=False)
        if len(group) >= min_symbol_rows
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        calculated = list(executor.map(_symbol_features, groups))
    if not calculated:
        return pd.DataFrame()

    frame = pd.concat(calculated, ignore_index=True)
    frame = _add_cross_section_features(frame)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
