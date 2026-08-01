from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import _select_evaluation_symbols
from .config import AppConfig
from .holding_backtest import _predict_stops, _prepare_position_features
from .stop_tuning import _entry_dates, _path_end_date, _simulate_path, summarize_paths


PURE_2PCT_STRATEGY = "A_pure_2pct"
PRODUCTION_STRATEGY = "B_production"
SELL_FLY_DAYS = (5, 10, 20)
SELL_FLY_THRESHOLDS = (0.05, 0.10)
PRIMARY_SELL_FLY_DAYS = 10
PRIMARY_SELL_FLY_THRESHOLD = 0.05
RETURN_TOLERANCE_PCT = 0.005
LOOKAHEAD_DAYS = 20


def _sell_fly_label(*, days: int, threshold: float) -> str:
    return f"{int(days)}d_{int(round(float(threshold) * 100))}pct"


def _entry_months(samples: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(samples["entry_date"]).dt.to_period("M").astype(str)


def _walk_forward_windows(samples: pd.DataFrame, config: AppConfig) -> list[dict[str, Any]]:
    months = sorted(_entry_months(samples).unique())
    train_months = int(config.volatility_stop.walk_forward_train_months)
    validation_months = int(config.volatility_stop.validation_months)
    step_months = int(config.volatility_stop.walk_forward_step_months)
    if len(months) < train_months + validation_months:
        return []
    entry_months = _entry_months(samples)
    windows: list[dict[str, Any]] = []
    window_no = 1
    for validation_start in range(
        train_months,
        len(months) - validation_months + 1,
        step_months,
    ):
        train_values = months[validation_start - train_months : validation_start]
        validation_values = months[
            validation_start : validation_start + validation_months
        ]
        windows.append(
            {
                "window_id": f"wf_{window_no:02d}",
                "train_months": train_values,
                "validation_months": validation_values,
                "validation_path_ids": set(
                    samples.loc[entry_months.isin(validation_values), "path_id"]
                    .astype(int)
                    .tolist()
                ),
            }
        )
        window_no += 1
    return windows


def _build_ab_paths(
    predicted: pd.DataFrame,
    symbols: list[str],
    entry_dates: list[date],
    holding_days_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    grouped = {
        symbol: group.sort_values("trade_date").reset_index(drop=True)
        for symbol, group in predicted.groupby("symbol", sort=False)
        if symbol in symbols
    }
    required_signal_columns = [
        "raw_close",
        "atr_14_pct",
        "predicted_high_return",
        "predicted_low_return",
        "median_high_return",
        "low_return_q20",
        "mean_distance",
        "sample_count",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "execution_date",
    ]
    required_full_columns = [
        "next_close",
        "execution_date",
    ]
    for entry_date in entry_dates:
        for symbol in symbols:
            symbol_frame = grouped.get(symbol)
            if symbol_frame is None:
                continue
            matches = symbol_frame.index[
                symbol_frame["trade_date"].dt.date == entry_date
            ].tolist()
            if not matches:
                continue
            start_index = matches[0]
            for holding_days in holding_days_values:
                row_count = int(holding_days) + LOOKAHEAD_DAYS
                full_rows = symbol_frame.iloc[
                    start_index : start_index + row_count
                ].copy()
                if len(full_rows) != row_count:
                    continue
                signal_rows = full_rows.iloc[:holding_days].copy()
                if signal_rows[required_signal_columns].isna().any().any():
                    continue
                if full_rows[required_full_columns].isna().any().any():
                    continue
                paths.append(
                    {
                        "entry_date": entry_date,
                        "symbol": symbol,
                        "holding_days": int(holding_days),
                        "entry_price": float(signal_rows.iloc[0]["raw_close"]),
                        "rows": signal_rows.reset_index(drop=True),
                        "full_rows": full_rows.reset_index(drop=True),
                    }
                )
    return paths


def _path_records(paths: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path_id": path_id,
                "entry_date": path["entry_date"],
                "symbol": path["symbol"],
                "holding_days": int(path["holding_days"]),
            }
            for path_id, path in enumerate(paths)
        ]
    )


def _exit_day(result: dict[str, Any], path: dict[str, Any]) -> int:
    holding_days = int(path["holding_days"])
    if not result.get("stopped"):
        return holding_days
    exit_date = result.get("exit_date")
    if exit_date is None or pd.isna(exit_date):
        return holding_days
    execution_dates = pd.to_datetime(path["rows"]["execution_date"]).dt.date
    matches = np.flatnonzero(execution_dates.to_numpy() == exit_date)
    if len(matches) == 0:
        return holding_days
    return int(matches[0]) + 1


def _post_exit_high_close(path: dict[str, Any], *, exit_day: int, days: int) -> float | None:
    rows = path["full_rows"]
    start = int(exit_day)
    end = start + int(days)
    if len(rows) < end:
        return None
    values = pd.to_numeric(rows.iloc[start:end]["next_close"], errors="coerce")
    values = values.dropna()
    if len(values) != int(days):
        return None
    return float(values.max())


def _simulate_ab_paths(paths: list[dict[str, Any]], config: AppConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    base_gap = float(config.recommendation.minimum_stop_gap_pct)
    for path_id, path in enumerate(paths):
        strategies = (
            (PURE_2PCT_STRATEGY, 0.0),
            (PRODUCTION_STRATEGY, float(config.recommendation.atr_stop_multiplier)),
        )
        for strategy, atr_multiplier in strategies:
            result = _simulate_path(
                path=path,
                config=config,
                stop_order_type=config.backtest.stop_order_type,
                atr_multiplier=atr_multiplier,
                limit_slippage_pct=float(config.recommendation.stop_limit_slippage_pct),
                minimum_stop_gap_pct=base_gap,
            )
            exit_day = _exit_day(result, path)
            record = {
                "path_id": int(path_id),
                "strategy": strategy,
                "variant": "pure_2pct" if strategy == PURE_2PCT_STRATEGY else "production",
                "configured_atr_multiplier": float(atr_multiplier),
                "exit_day": int(exit_day),
                **result,
            }
            for days in SELL_FLY_DAYS:
                record[f"post_exit_high_close_{days}d"] = _post_exit_high_close(
                    path,
                    exit_day=exit_day,
                    days=int(days),
                )
            records.append(record)
    return pd.DataFrame(records)


def summarize_ab_results(frame: pd.DataFrame) -> dict[str, Any]:
    summary = summarize_paths(frame)
    stopped = frame["stopped"].astype(bool)
    summary["average_holding_days"] = float(frame["exit_day"].mean())
    for days in SELL_FLY_DAYS:
        close_column = f"post_exit_high_close_{int(days)}d"
        observable = frame[close_column].notna()
        exited_observable = observable & stopped
        for threshold in SELL_FLY_THRESHOLDS:
            label = _sell_fly_label(days=int(days), threshold=float(threshold))
            fly = (
                observable
                & stopped
                & frame["exit_price"].notna()
                & (
                    frame[close_column].astype(float)
                    >= frame["exit_price"].astype(float) * (1.0 + float(threshold))
                )
            )
            summary[f"sell_fly_all_{label}"] = (
                float(fly.sum() / observable.sum()) if observable.any() else 0.0
            )
            summary[f"sell_fly_exited_{label}"] = (
                float(fly.sum() / exited_observable.sum())
                if exited_observable.any()
                else 0.0
            )
    return summary


def _summarize_by_strategy(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        strategy: summarize_ab_results(group)
        for strategy, group in frame.groupby("strategy", sort=True)
    }


def _delta_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    primary_label = _sell_fly_label(
        days=PRIMARY_SELL_FLY_DAYS,
        threshold=PRIMARY_SELL_FLY_THRESHOLD,
    )
    metrics = [
        "average_strategy_max_loss",
        "p05_strategy_return",
        "average_strategy_return",
        "average_excess_return",
        "whipsaw_rate",
        f"sell_fly_all_{primary_label}",
        "average_holding_days",
        "stopped_rate",
        "stop_limit_unfilled_rate",
    ]
    deltas: dict[str, float] = {}
    for metric in metrics:
        if metric in a and metric in b:
            deltas[metric] = float(a[metric]) - float(b[metric])
    return deltas


def _ab_diagnostics(summaries: dict[str, Any]) -> dict[str, Any]:
    a = summaries[PURE_2PCT_STRATEGY]
    b = summaries[PRODUCTION_STRATEGY]
    deltas = _delta_metrics(a, b)
    primary_label = _sell_fly_label(
        days=PRIMARY_SELL_FLY_DAYS,
        threshold=PRIMARY_SELL_FLY_THRESHOLD,
    )
    a_criteria = {
        "max_loss_not_worse": deltas.get("average_strategy_max_loss", 0.0) >= 0.0,
        "p05_not_worse": deltas.get("p05_strategy_return", 0.0) >= 0.0,
        "return_not_worse_0_5pp": deltas.get("average_strategy_return", 0.0)
        >= -RETURN_TOLERANCE_PCT,
    }
    b_criteria = {
        "max_loss_not_worse": deltas.get("average_strategy_max_loss", 0.0)
        <= 0.0,
        "p05_not_worse": deltas.get("p05_strategy_return", 0.0) <= 0.0,
        "return_better_0_5pp": deltas.get("average_strategy_return", 0.0)
        <= -RETURN_TOLERANCE_PCT,
        "sell_fly_lower": deltas.get(f"sell_fly_all_{primary_label}", 0.0) > 0.0,
    }
    if all(a_criteria.values()):
        recommendation = "align_to_A_pure_2pct"
    elif all(b_criteria.values()):
        recommendation = "keep_B_production"
    else:
        recommendation = "inconclusive_review_deltas"
    return {
        "recommendation": recommendation,
        "a_criteria": a_criteria,
        "b_criteria": b_criteria,
        "deltas_a_minus_b": deltas,
    }


def _bootstrap_deltas(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        return {"status": "skipped", "reason": "bootstrap disabled"}
    required = {PURE_2PCT_STRATEGY, PRODUCTION_STRATEGY}
    if not required.issubset(set(frame["strategy"].astype(str))):
        return {"status": "skipped", "reason": "required strategies missing"}
    path_ids = frame["path_id"].astype(int).drop_duplicates().to_numpy()
    if len(path_ids) < 2:
        return {"status": "skipped", "reason": "not enough paths"}
    rng = np.random.default_rng(seed)
    index_by_path = {
        int(path_id): frame.index[frame["path_id"].astype(int) == int(path_id)].to_numpy()
        for path_id in path_ids
    }
    collected: dict[str, list[float]] = {}
    recommendation_counts: dict[str, int] = {}
    for _ in range(iterations):
        sampled = rng.choice(path_ids, size=len(path_ids), replace=True)
        indices = np.concatenate([index_by_path[int(path_id)] for path_id in sampled])
        diagnostics = _ab_diagnostics(_summarize_by_strategy(frame.loc[indices]))
        recommendation = str(diagnostics["recommendation"])
        recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
        for metric, value in diagnostics["deltas_a_minus_b"].items():
            collected.setdefault(metric, []).append(float(value))
    return {
        "status": "ok",
        "iterations": int(iterations),
        "path_count": int(len(path_ids)),
        "recommendation_rate": {
            key: float(value / iterations)
            for key, value in sorted(recommendation_counts.items())
        },
        "deltas": {
            metric: {
                "mean": float(np.mean(values)),
                "standard_error": float(np.std(values, ddof=1)),
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for metric, values in collected.items()
        },
    }


def _directional_pass(metric: str, mean: float, se: float) -> bool:
    primary_label = _sell_fly_label(
        days=PRIMARY_SELL_FLY_DAYS,
        threshold=PRIMARY_SELL_FLY_THRESHOLD,
    )
    lower_is_better = {
        "whipsaw_rate",
        f"sell_fly_all_{primary_label}",
        "stopped_rate",
        "stop_limit_unfilled_rate",
    }
    if metric in lower_is_better:
        return bool(mean < -se)
    return bool(mean > se)


def _aggregate_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    if not ok:
        return {"status": "skipped", "reason": "no successful windows"}
    deltas = pd.DataFrame(
        [record["diagnostics"]["deltas_a_minus_b"] for record in ok]
    ).fillna(0.0)
    delta_summary: dict[str, Any] = {}
    for metric in deltas.columns:
        values = deltas[metric].astype(float)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        positive_direction_rate = float(
            np.mean(
                [
                    _directional_pass(metric, float(value), 0.0)
                    for value in values
                ]
            )
        )
        delta_summary[metric] = {
            "mean": mean,
            "standard_error_across_windows": se,
            "window_direction_rate": positive_direction_rate,
            "passes_directional_gate": bool(
                positive_direction_rate >= 0.5
                and _directional_pass(metric, mean, se)
            ),
        }
    a_criteria = {
        "max_loss_not_worse_than_one_se": delta_summary[
            "average_strategy_max_loss"
        ]["mean"]
        >= -delta_summary["average_strategy_max_loss"][
            "standard_error_across_windows"
        ],
        "p05_not_worse_than_one_se": delta_summary["p05_strategy_return"]["mean"]
        >= -delta_summary["p05_strategy_return"]["standard_error_across_windows"],
        "return_not_worse_0_5pp": delta_summary["average_strategy_return"]["mean"]
        >= -RETURN_TOLERANCE_PCT,
    }
    recommendations = [record["diagnostics"]["recommendation"] for record in ok]
    recommendation_counts = {
        value: recommendations.count(value) for value in sorted(set(recommendations))
    }
    recommendation_rates = {
        key: float(value / len(ok)) for key, value in recommendation_counts.items()
    }
    return {
        "status": "ok",
        "window_count": int(len(ok)),
        "recommendation_rates": recommendation_rates,
        "a_criteria": a_criteria,
        "recommendation": (
            "align_to_A_pure_2pct"
            if all(a_criteria.values())
            else "inconclusive_review_deltas"
        ),
        "delta_summary": delta_summary,
    }


def _run_walk_forward(
    samples: pd.DataFrame,
    results: pd.DataFrame,
    config: AppConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    windows = _walk_forward_windows(samples, config)
    if not windows:
        return (
            {
                "status": "skipped",
                "reason": "not enough entry months for configured walk-forward",
            },
            pd.DataFrame(),
        )
    records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for window in windows:
        frame = results.loc[
            results["path_id"].astype(int).isin(window["validation_path_ids"])
        ].copy()
        if frame.empty:
            records.append(
                {
                    "window_id": window["window_id"],
                    "status": "failed",
                    "error": "empty validation frame",
                }
            )
            continue
        summaries = _summarize_by_strategy(frame)
        diagnostics = _ab_diagnostics(summaries)
        records.append(
            {
                "window_id": window["window_id"],
                "status": "ok",
                "train_months": window["train_months"],
                "validation_months": window["validation_months"],
                "validation_path_count": int(len(window["validation_path_ids"])),
                "summaries": summaries,
                "diagnostics": diagnostics,
                "bootstrap": _bootstrap_deltas(
                    frame,
                    iterations=int(config.volatility_stop.bootstrap_samples),
                    seed=int(config.training.random_seed),
                ),
            }
        )
        frame["window_id"] = window["window_id"]
        frames.append(frame)
    aggregate = _aggregate_diagnostics(records)
    return (
        {
            "status": "ok" if frames else "failed",
            "records": records,
            "aggregate": aggregate,
        },
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
    )


def run_ab_stop_backtest(
    engine: Any,
    config: AppConfig,
    *,
    start_date: date,
    end_date: date,
    max_symbols: int | None = None,
    output_directory: str | Path = "output/ab-stop",
) -> dict[str, Any]:
    if start_date > end_date:
        raise ValueError("A/B 止损回测开始日期不能晚于结束日期")
    max_symbols = max_symbols or int(config.volatility_stop.max_symbols)
    entry_dates = _entry_dates(engine, start_date, end_date)
    if not entry_dates:
        raise RuntimeError("A/B 止损回测范围没有交易日")
    holding_days = tuple(int(value) for value in config.volatility_stop.holding_days)
    max_holding = max(holding_days)
    complete_entry_dates: list[date] = []
    skipped_incomplete_entry_dates: list[date] = []
    for entry_date in entry_dates:
        try:
            _path_end_date(engine, entry_date, max_holding + LOOKAHEAD_DAYS)
            complete_entry_dates.append(entry_date)
        except RuntimeError:
            skipped_incomplete_entry_dates.append(entry_date)
    if not complete_entry_dates:
        raise RuntimeError("A/B 止损回测没有满足 holding_days + lookahead 的完整建仓月份")
    path_end_date = _path_end_date(
        engine,
        complete_entry_dates[-1],
        max_holding + LOOKAHEAD_DAYS,
    )
    entry_dates = complete_entry_dates
    symbols = _select_evaluation_symbols(engine, config, start_date, max_symbols)
    if not symbols:
        raise RuntimeError("没有可用于 A/B 止损回测的股票")

    features = _prepare_position_features(
        engine,
        config,
        entry_dates[0],
        path_end_date,
        symbols,
    )
    features = features.loc[
        (features["trade_date"].dt.date >= entry_dates[0])
        & (features["trade_date"].dt.date < path_end_date)
    ].copy()
    predicted = _predict_stops(engine, config, features)
    if predicted.empty:
        raise RuntimeError("A/B 止损回测没有可用预测特征")
    paths = _build_ab_paths(
        predicted,
        symbols,
        entry_dates,
        holding_days,
    )
    if not paths:
        raise RuntimeError("没有构建出完整的 A/B 止损回测路径")
    print(
        f"A/B 止损回测：{len(symbols)} 支股票，{len(entry_dates)} 个建仓月，"
        f"{len(paths)} 条路径",
        flush=True,
    )

    results = _simulate_ab_paths(paths, config)
    samples = _path_records(paths)
    summaries = _summarize_by_strategy(results)
    diagnostics = _ab_diagnostics(summaries)
    walk_forward, walk_forward_paths = _run_walk_forward(samples, results, config)

    run_id = datetime.now(timezone.utc).strftime("ab_%Y%m%dT%H%M%SZ")
    output_root = Path(output_directory).resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    paths_path = output_root / "paths.csv"
    results_path = output_root / "ab_paths.csv"
    walk_forward_path = output_root / "walk_forward_summary.json"
    walk_forward_paths_path = output_root / "walk_forward_paths.csv"
    summary_path = output_root / "summary.json"
    samples.to_csv(paths_path, index=False, encoding="utf-8-sig")
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    if not walk_forward_paths.empty:
        walk_forward_paths.to_csv(walk_forward_paths_path, index=False, encoding="utf-8-sig")
    walk_forward_path.write_text(
        json.dumps(walk_forward, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    summary = {
        "run_id": run_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "path_end_date": path_end_date.isoformat(),
        "symbol_count": int(len(symbols)),
        "entry_dates": [value.isoformat() for value in entry_dates],
        "skipped_incomplete_entry_dates": [
            value.isoformat() for value in skipped_incomplete_entry_dates
        ],
        "path_count": int(len(paths)),
        "sample_count": int(len(samples)),
        "summaries": summaries,
        "diagnostics": diagnostics,
        "walk_forward": walk_forward,
        "promotion_action": "report_only",
        "production_changed": False,
        "ab_variants": {
            PURE_2PCT_STRATEGY: {
                "atr_stop_multiplier": 0.0,
                "minimum_stop_gap_pct": float(config.recommendation.minimum_stop_gap_pct),
                "description": "同一 make_risk_decision；atr_stop_multiplier=0，使 soft_stop=close*(1-minimum_stop_gap_pct)",
            },
            PRODUCTION_STRATEGY: {
                "atr_stop_multiplier": float(config.recommendation.atr_stop_multiplier),
                "minimum_stop_gap_pct": float(config.recommendation.minimum_stop_gap_pct),
                "description": "当前生产 make_risk_decision 参数",
            },
        },
        "recommendation_config": asdict(config.recommendation),
        "output_directory": str(output_root),
        "paths_path": str(paths_path),
        "ab_paths_path": str(results_path),
        "walk_forward_summary_path": str(walk_forward_path),
        "walk_forward_paths_path": (
            str(walk_forward_paths_path) if not walk_forward_paths.empty else None
        ),
        "method": (
            "A/B report-only：A=纯2%冠军，B=现生产。两者同批路径、同 make_risk_decision、"
            "同成交模型；仅 atr_stop_multiplier 不同。A-B delta 为正表示 A 在该指标更高。"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary
