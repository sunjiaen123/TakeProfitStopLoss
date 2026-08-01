from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .ab_stop import run_ab_stop_backtest
from .backtest import run_backtest
from .chart_exit import run_chart_exit_backtest
from .config import load_config
from .db import create_engine, initialize_database
from .data_sync import sync_market_data
from .exit_machine import run_exit_machine_backtest
from .holding_backtest import run_holding_backtest
from .pipeline import (
    export_preview,
    recommend_pipeline,
    resolve_recommend_as_of_date,
    train_pipeline,
)
from .risk_fit import fit_risk_model
from .stop_tuning import run_stop_tuning
from .volatility_stop import fit_volatility_stop


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式必须是 YYYY-MM-DD") from exc


def _parse_recommend_date(value: str) -> date | str:
    if value.strip().lower() == "latest":
        return "latest"
    return _parse_date(value)


def _parse_sync_end_date(value: str) -> date:
    if value.strip().lower() == "today":
        return date.today()
    return _parse_date(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tpsl",
        description="持仓股票次日止盈止损建议",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="TOML 配置文件，默认 config.toml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="创建数据库和全部业务表")

    sync = subparsers.add_parser("sync-data", help="同步股票基础信息和历史日线")
    sync.add_argument(
        "--end-date",
        required=True,
        type=_parse_sync_end_date,
        help="YYYY-MM-DD，或 today 自动使用本机当天日期",
    )
    sync.add_argument("--start-date", type=_parse_date)
    sync.add_argument(
        "--scope",
        choices=("positions", "all"),
        default="positions",
        help="positions 仅同步持仓，all 同步全部 A 股",
    )

    train = subparsers.add_parser("train", help="训练分位数模型并构建相似形态索引")
    train.add_argument("--end-date", required=True, type=_parse_date)

    backtest = subparsers.add_parser("backtest", help="执行按月滚动回测")
    backtest.add_argument("--start-date", required=True, type=_parse_date)
    backtest.add_argument("--end-date", required=True, type=_parse_date)
    backtest.add_argument("--max-symbols", type=int)
    backtest.add_argument(
        "--stop-order-type",
        choices=("market", "limit"),
    )
    backtest.add_argument(
        "--output-dir",
        default="output/backtests",
    )
    backtest.add_argument(
        "--no-db",
        action="store_true",
        help="不写入 MySQL，只生成本地报告",
    )
    backtest.add_argument(
        "--no-reuse-folds",
        action="store_true",
        help="不复用已经存在的月度模型",
    )

    holding_backtest = subparsers.add_parser(
        "holding-backtest",
        help="从当前持仓真实建仓日执行多日止损回测",
    )
    holding_backtest.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
    )
    holding_backtest.add_argument(
        "--stop-order-type",
        choices=("market", "limit"),
    )
    holding_backtest.add_argument(
        "--output-dir",
        default="output/holding-backtests",
    )
    holding_backtest.add_argument(
        "--no-db",
        action="store_true",
        help="不写入 MySQL，只生成本地报告",
    )

    stop_tuning = subparsers.add_parser(
        "tune-stop",
        help="扫描多周期历史持仓的止损参数",
    )
    stop_tuning.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="模拟建仓开始日期",
    )
    stop_tuning.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="模拟建仓结束日期",
    )
    stop_tuning.add_argument("--max-symbols", type=int)
    stop_tuning.add_argument(
        "--output-dir",
        default="output/stop-tuning",
    )
    stop_tuning.add_argument(
        "--no-db",
        action="store_true",
        help="不写入 MySQL，只生成本地报告",
    )

    fit_risk = subparsers.add_parser(
        "fit-risk",
        help="拟合动态止损距离模型",
    )
    fit_risk.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="历史建仓路径开始日期",
    )
    fit_risk.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="历史建仓路径结束日期",
    )
    fit_risk.add_argument("--max-symbols", type=int)
    fit_risk.add_argument(
        "--output-dir",
        default="output/risk-fit",
    )
    fit_risk.add_argument(
        "--promote",
        action="store_true",
        help="将本次拟合模型晋级为生产动态风控模型；默认只生成报告",
    )
    fit_risk.add_argument(
        "--force-promote",
        action="store_true",
        help="晋级检查失败时仍强制切换生产动态风控模型",
    )

    fit_volatility = subparsers.add_parser(
        "fit-volatility-stop",
        help="标定 per-股波动分层止损宽度",
    )
    fit_volatility.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="历史建仓路径开始日期",
    )
    fit_volatility.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="历史建仓路径结束日期",
    )
    fit_volatility.add_argument("--max-symbols", type=int)
    fit_volatility.add_argument(
        "--output-dir",
        default="output/volatility-stop",
    )
    fit_volatility.add_argument(
        "--promote",
        action="store_true",
        help="将本次 k 标定结果晋级为生产波动止损参数；默认只生成报告",
    )
    fit_volatility.add_argument(
        "--force-promote",
        action="store_true",
        help="晋级检查失败时仍强制切换生产波动止损参数",
    )

    exit_machine = subparsers.add_parser(
        "backtest-exit-machine",
        help="回测状态机退出策略（report-only，不改生产）",
    )
    exit_machine.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="历史建仓路径开始日期",
    )
    exit_machine.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="历史建仓路径结束日期",
    )
    exit_machine.add_argument("--max-symbols", type=int)
    exit_machine.add_argument(
        "--output-dir",
        default="output/exit-machine",
    )

    ab_stop = subparsers.add_parser(
        "backtest-ab-stop",
        help="A/B 回测纯 2%% 止损 vs 现生产止损（report-only，不改生产）",
    )
    ab_stop.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="历史建仓路径开始日期",
    )
    ab_stop.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="历史建仓路径结束日期",
    )
    ab_stop.add_argument("--max-symbols", type=int)
    ab_stop.add_argument(
        "--output-dir",
        default="output/ab-stop",
    )

    chart_exit = subparsers.add_parser(
        "backtest-chart-exit",
        help="回测K线/均线、分段利润底和条件式再上车（report-only）",
    )
    chart_exit.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="历史月度建仓路径开始日期",
    )
    chart_exit.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="历史月度建仓路径结束日期",
    )
    chart_exit.add_argument("--max-symbols", type=int)
    chart_exit.add_argument(
        "--output-dir",
        default="output/chart-exit",
    )

    recommend = subparsers.add_parser("recommend", help="生成指定交易日收盘后的建议")
    recommend.add_argument(
        "--as-of-date",
        required=True,
        type=_parse_recommend_date,
        help="YYYY-MM-DD，或 latest 自动使用所有持仓都有行情的最新日期",
    )
    recommend.add_argument("--output", help="额外输出 CSV 预览文件")
    recommend.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算和显示，不写入 MySQL",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "init-db":
            result = initialize_database(config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        engine = create_engine(config)
        if args.command == "sync-data":
            result = sync_market_data(
                engine,
                config,
                end_date=args.end_date,
                scope=args.scope,
                start_date=args.start_date,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "train":
            metadata = train_pipeline(engine, config, args.end_date)
            print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "backtest":
            summary = run_backtest(
                engine,
                config,
                start_date=args.start_date,
                end_date=args.end_date,
                max_symbols=args.max_symbols,
                stop_order_type=args.stop_order_type,
                output_directory=args.output_dir,
                write_database=not args.no_db,
                reuse_folds=not args.no_reuse_folds,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "holding-backtest":
            result = run_holding_backtest(
                engine,
                config,
                end_date=args.end_date,
                stop_order_type=args.stop_order_type,
                output_directory=args.output_dir,
                write_database=not args.no_db,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "tune-stop":
            result = run_stop_tuning(
                engine,
                config,
                entry_start_date=args.start_date,
                entry_end_date=args.end_date,
                max_symbols=args.max_symbols,
                output_directory=args.output_dir,
                write_database=not args.no_db,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "fit-risk":
            if args.force_promote and not args.promote:
                raise ValueError("--force-promote 必须和 --promote 一起使用")
            result = fit_risk_model(
                engine,
                config,
                start_date=args.start_date,
                end_date=args.end_date,
                max_symbols=args.max_symbols,
                output_directory=args.output_dir,
                save_model=args.promote,
                force_promote=args.force_promote,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "fit-volatility-stop":
            if args.force_promote and not args.promote:
                raise ValueError("--force-promote 必须和 --promote 一起使用")
            result = fit_volatility_stop(
                engine,
                config,
                start_date=args.start_date,
                end_date=args.end_date,
                max_symbols=args.max_symbols,
                output_directory=args.output_dir,
                save_model=args.promote,
                force_promote=args.force_promote,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "backtest-exit-machine":
            result = run_exit_machine_backtest(
                engine,
                config,
                start_date=args.start_date,
                end_date=args.end_date,
                max_symbols=args.max_symbols,
                output_directory=args.output_dir,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "backtest-ab-stop":
            result = run_ab_stop_backtest(
                engine,
                config,
                start_date=args.start_date,
                end_date=args.end_date,
                max_symbols=args.max_symbols,
                output_directory=args.output_dir,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "backtest-chart-exit":
            result = run_chart_exit_backtest(
                engine,
                config,
                start_date=args.start_date,
                end_date=args.end_date,
                max_symbols=args.max_symbols,
                output_directory=args.output_dir,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return

        as_of_date = resolve_recommend_as_of_date(
            engine,
            config,
            args.as_of_date,
        )
        if args.as_of_date == "latest":
            print(f"使用最新完整持仓行情日：{as_of_date}")
        frame = recommend_pipeline(
            engine,
            config,
            as_of_date,
            write_database=not args.dry_run,
        )
        export_preview(frame, args.output)
        if frame.empty:
            print("没有有效持仓，无建议需要生成。")
            return
        display_columns = [
            "symbol",
            "close_price",
            "take_profit_enabled",
            "take_profit_price",
            "stop_trigger_price",
            "stop_limit_price",
            "dynamic_stop_gap_pct",
            "risk_reward_ratio",
            "confidence",
        ]
        display_columns = [
            column for column in display_columns if column in frame.columns
        ]
        print(frame[display_columns].to_string(index=False))
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
