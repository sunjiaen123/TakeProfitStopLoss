from __future__ import annotations

import atexit
import math
import socket
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from multiprocessing import current_process
from typing import Any, Callable, Iterable

import pandas as pd
from sqlalchemy import bindparam, text

from .config import AppConfig
from .db import load_positions


BAR_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "turn,tradestatus,isST"
)
ETF_BAR_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,tradestatus"
)

SECURITY_TYPE_STOCK = "stock"
SECURITY_TYPE_ETF = "etf"

BAOSTOCK_FATAL_LOGIN_CODES = {"10001011"}
SUPPORTED_DATA_SOURCES = {"baostock", "akshare", "auto"}

_WORKER_LOGGED_IN = False
_WORKER_LOGOUT_REGISTERED = False


def to_baostock_code(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if "." not in normalized:
        raise ValueError(f"股票代码必须包含交易所后缀：{symbol}")
    code, exchange = normalized.split(".", 1)
    if exchange not in {"SH", "SZ", "BJ"} or len(code) != 6 or not code.isdigit():
        raise ValueError(f"不支持的股票代码格式：{symbol}")
    return f"{exchange.lower()}.{code}"


def from_baostock_code(code: str) -> str:
    exchange, number = code.strip().split(".", 1)
    return f"{number}.{exchange.upper()}"


def to_akshare_code(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if "." not in normalized:
        raise ValueError(f"股票代码必须包含交易所后缀：{symbol}")
    code, exchange = normalized.split(".", 1)
    if exchange not in {"SH", "SZ", "BJ"} or len(code) != 6 or not code.isdigit():
        raise ValueError(f"不支持的股票代码格式：{symbol}")
    return code


def _exchange_for_akshare_code(code: str) -> str:
    value = str(code).strip().zfill(6)
    if value.startswith("6"):
        return "SH"
    if value.startswith(("0", "3")):
        return "SZ"
    if value.startswith(("43", "83", "87", "88", "92")):
        return "BJ"
    raise ValueError(f"无法识别 AkShare 股票代码所属交易所：{code}")


def from_akshare_code(code: str) -> str:
    value = str(code).strip().zfill(6)
    return f"{value}.{_exchange_for_akshare_code(value)}"


def from_akshare_etf_code(code: str) -> str:
    value = str(code).strip().zfill(6)
    if value.startswith(("51", "52", "56", "58")):
        exchange = "SH"
    elif value.startswith("159"):
        exchange = "SZ"
    else:
        raise ValueError(f"无法识别 AkShare ETF 代码所属交易所：{code}")
    return f"{value}.{exchange}"


def _looks_like_etf_symbol(symbol: str) -> bool:
    normalized = str(symbol).strip().upper()
    if "." not in normalized:
        return False
    code, exchange = normalized.split(".", 1)
    return (
        exchange == "SH"
        and code.startswith(("51", "52", "56", "58"))
    ) or (exchange == "SZ" and code.startswith("159"))


def _supported_baostock_types(include_etfs: bool) -> set[str]:
    supported = {"1"}
    if include_etfs:
        # BaoStock query_stock_basic: 1=股票，5=ETF。
        supported.add("5")
    return supported


def _collect_result(result: Any) -> pd.DataFrame:
    if result is None:
        raise RuntimeError("BaoStock 返回空结果")
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock {result.error_code}: {result.error_msg}")
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def _board_for(symbol: str) -> str:
    code, exchange = symbol.split(".", 1)
    if exchange == "BJ":
        return "BSE"
    if exchange == "SH" and code.startswith("688"):
        return "STAR"
    if exchange == "SZ" and code.startswith(("300", "301")):
        return "CHINEXT"
    return "MAIN"


def _price_limit(symbol: str, stock_name: str) -> float:
    if "ST" in stock_name.upper():
        return 0.05
    board = _board_for(symbol)
    if board in {"STAR", "CHINEXT"}:
        return 0.20
    if board == "BSE":
        return 0.30
    return 0.10


def _close_baostock_socket() -> None:
    try:
        import baostock.common.context as context

        socket = getattr(context, "default_socket", None)
        if socket is not None:
            socket.close()
        setattr(context, "default_socket", None)
    except Exception:
        pass


def _is_fatal_login_error(error_code: str) -> bool:
    return str(error_code).strip() in BAOSTOCK_FATAL_LOGIN_CODES


def _format_login_error(error_code: str, error_msg: str) -> str:
    base = f"{error_code}: {error_msg}"
    if _is_fatal_login_error(error_code):
        return (
            f"{base}。BaoStock 服务端已拒绝本机登录，继续重试无效；"
            "请暂停 sync-data，稍后更换网络或联系 BaoStock 管理员。"
            "在恢复前可用 recommend --as-of-date latest 基于本地最新行情生成建议。"
        )
    return base


def _is_fatal_baostock_error_message(message: str) -> bool:
    return any(code in str(message) for code in BAOSTOCK_FATAL_LOGIN_CODES)


def _login_with_retry(
    retry_count: int,
    retry_delay: float,
    stagger: bool = False,
    register_logout: bool = True,
    socket_timeout: float = 30.0,
) -> str:
    global _WORKER_LOGGED_IN, _WORKER_LOGOUT_REGISTERED

    if _WORKER_LOGGED_IN:
        return ""
    try:
        import baostock as bs
    except ImportError:
        return "缺少 baostock，请先安装 requirements.txt"

    if stagger:
        identity = current_process()._identity
        worker_number = identity[0] if identity else 1
        time.sleep(max(0, worker_number - 1) * min(1.0, retry_delay / 2))

    last_error = "未知登录错误"
    for attempt in range(1, retry_count + 1):
        _close_baostock_socket()
        try:
            previous_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(socket_timeout)
            try:
                login = bs.login()
            finally:
                socket.setdefaulttimeout(previous_timeout)
            if login.error_code == "0":
                _WORKER_LOGGED_IN = True
                if register_logout and not _WORKER_LOGOUT_REGISTERED:
                    atexit.register(bs.logout)
                    _WORKER_LOGOUT_REGISTERED = True
                return ""
            last_error = _format_login_error(login.error_code, login.error_msg)
            if _is_fatal_login_error(login.error_code):
                return last_error
        except Exception as exc:
            last_error = str(exc)
        if attempt < retry_count:
            time.sleep(retry_delay * attempt)
    return last_error


def fetch_baostock_universe(
    as_of_date: date,
    retry_count: int,
    retry_delay: float,
    socket_timeout: float,
    include_etfs: bool = False,
) -> pd.DataFrame:
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError("缺少 baostock，请先安装 requirements.txt") from exc

    login_error = _login_with_retry(
        retry_count,
        retry_delay,
        register_logout=False,
        socket_timeout=socket_timeout,
    )
    if login_error:
        raise RuntimeError(f"BaoStock 登录失败：{login_error}")
    try:
        basic = _collect_result(bs.query_stock_basic())
        industry = _collect_result(
            bs.query_stock_industry(date=as_of_date.isoformat())
        )
    finally:
        bs.logout()
        global _WORKER_LOGGED_IN
        _WORKER_LOGGED_IN = False
        _close_baostock_socket()

    if basic.empty:
        raise RuntimeError("BaoStock 没有返回股票基础信息")
    basic["type"] = basic["type"].astype(str)
    basic = basic.loc[
        basic["type"].isin(_supported_baostock_types(include_etfs))
    ].copy()
    if "status" in basic:
        basic = basic.loc[basic["status"].astype(str).eq("1")].copy()
    basic["symbol"] = basic["code"].map(from_baostock_code)
    basic["security_type"] = basic["type"].map(
        {"1": SECURITY_TYPE_STOCK, "5": SECURITY_TYPE_ETF}
    )

    if industry.empty:
        industry_map = pd.DataFrame(columns=["code", "industry"])
    else:
        industry_map = industry[["code", "industry"]].drop_duplicates("code")
    merged = basic.merge(industry_map, on="code", how="left")
    merged["industry"] = merged["industry"].fillna("UNKNOWN")
    merged.loc[
        merged["security_type"].eq(SECURITY_TYPE_ETF), "industry"
    ] = "ETF"
    return merged.reset_index(drop=True)


def _akshare_spot_records(
    spot: pd.DataFrame,
    security_type: str,
) -> list[dict[str, Any]]:
    if "代码" not in spot.columns:
        raise RuntimeError(
            f"AkShare 证券列表缺少代码列，实际列：{list(spot.columns)}"
        )
    name_column = "名称" if "名称" in spot.columns else None
    frame = spot.copy()
    frame["ak_code"] = frame["代码"].astype(str).str.extract(r"(\d{6})")[0]
    frame = frame.dropna(subset=["ak_code"]).copy()
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        code = str(row["ak_code"]).zfill(6)
        try:
            symbol = (
                from_akshare_etf_code(code)
                if security_type == SECURITY_TYPE_ETF
                else from_akshare_code(code)
            )
        except ValueError:
            continue
        records.append(
            {
                "code": code,
                "code_name": str(row.get(name_column, "")) if name_column else "",
                "ipoDate": None,
                "outDate": "",
                "industry": (
                    "ETF" if security_type == SECURITY_TYPE_ETF else "UNKNOWN"
                ),
                "symbol": symbol,
                "security_type": security_type,
            }
        )
    return records


def fetch_akshare_universe(
    as_of_date: date,
    include_etfs: bool = False,
) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "缺少 akshare，请先执行 pip install akshare，或将 data_sync.source 改回 baostock"
        ) from exc

    spot = ak.stock_zh_a_spot_em()
    if spot is None or spot.empty:
        raise RuntimeError("AkShare 没有返回 A 股列表")
    records = _akshare_spot_records(spot, SECURITY_TYPE_STOCK)
    if include_etfs:
        etf_spot = ak.fund_etf_spot_em()
        if etf_spot is None or etf_spot.empty:
            raise RuntimeError("AkShare 没有返回 ETF 列表")
        records.extend(_akshare_spot_records(etf_spot, SECURITY_TYPE_ETF))
    if not records:
        raise RuntimeError("AkShare 证券列表没有可识别的沪深京代码")
    return pd.DataFrame(records).drop_duplicates("symbol").reset_index(drop=True)


def fetch_universe(
    source: str,
    as_of_date: date,
    retry_count: int,
    retry_delay: float,
    socket_timeout: float,
    include_etfs: bool = False,
) -> tuple[pd.DataFrame, str]:
    source = source.lower()
    if source == "akshare":
        return fetch_akshare_universe(as_of_date, include_etfs), "akshare"
    if source == "baostock":
        return (
            fetch_baostock_universe(
                as_of_date,
                retry_count,
                retry_delay,
                socket_timeout,
                include_etfs,
            ),
            "baostock",
        )
    if source != "auto":
        raise ValueError("data_sync.source 只支持 baostock、akshare 或 auto")

    if include_etfs:
        # BaoStock SDK 在大列表分页时可能吞掉 timeout/UnicodeDecodeError，
        # 只打印错误并把不完整结果当作正常结束，外层无法可靠触发 fallback。
        # ETF 持仓因此优先使用 AkShare 的专用 ETF 列表和历史接口。
        try:
            return fetch_akshare_universe(as_of_date, True), "akshare"
        except Exception as exc:
            print(f"AkShare 不可用，自动回退 BaoStock：{exc}", flush=True)
            try:
                return (
                    fetch_baostock_universe(
                        as_of_date,
                        retry_count,
                        retry_delay,
                        socket_timeout,
                        True,
                    ),
                    "baostock",
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    "AkShare 与 BaoStock 均不可用。"
                    f"AkShare：{exc}；BaoStock：{fallback_exc}"
                ) from fallback_exc

    try:
        return (
            fetch_baostock_universe(
                as_of_date,
                retry_count,
                retry_delay,
                socket_timeout,
                include_etfs,
            ),
            "baostock",
        )
    except Exception as exc:
        # auto 的职责是数据源故障转移。BaoStock 的登录、基础信息、
        # 行业查询都可能以 RuntimeError、TimeoutError 或 socket error
        # 失败，不能只在黑名单错误时才切换备用源。
        print(f"BaoStock 不可用，自动切换 AkShare：{exc}", flush=True)
        try:
            return fetch_akshare_universe(as_of_date, include_etfs), "akshare"
        except Exception as fallback_exc:
            raise RuntimeError(
                "BaoStock 与 AkShare 均不可用。"
                f"BaoStock：{exc}；AkShare：{fallback_exc}"
            ) from fallback_exc


def _worker_login(
    retry_count: int,
    retry_delay: float,
    socket_timeout: float,
) -> None:
    # 初始化阶段失败不能抛出，否则 ProcessPool 会整体损坏。
    # 具体任务开始时会再次登录并把最终错误作为该股票的失败结果返回。
    socket.setdefaulttimeout(socket_timeout)
    _login_with_retry(
        retry_count,
        retry_delay,
        stagger=True,
        socket_timeout=socket_timeout,
    )


def _download_symbol_baostock(
    task: tuple[str, str, str, str, str, int, float, float],
) -> dict[str, Any]:
    import baostock as bs

    (
        symbol,
        start_date,
        end_date,
        industry_code,
        security_type,
        retry_count,
        retry_delay,
        socket_timeout,
    ) = task
    login_error = _login_with_retry(
        retry_count,
        retry_delay,
        socket_timeout=socket_timeout,
    )
    if login_error:
        return {
            "symbol": symbol,
            "rows": [],
            "error": f"BaoStock worker 登录失败：{login_error}",
        }
    code = to_baostock_code(symbol)
    last_error: Exception | None = None
    for attempt in range(1, retry_count + 1):
        try:
            bar_fields = (
                ETF_BAR_FIELDS
                if security_type == SECURITY_TYPE_ETF
                else BAR_FIELDS
            )
            raw = _collect_result(
                bs.query_history_k_data_plus(
                    code,
                    bar_fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3",
                )
            )
            try:
                adjusted = _collect_result(
                    bs.query_history_k_data_plus(
                        code,
                        "date,close",
                        start_date=start_date,
                        end_date=end_date,
                        frequency="d",
                        adjustflag="2",
                    )
                )
            except Exception:
                # 部分基金不返回复权序列；此时原始净值行情可直接使用。
                adjusted = pd.DataFrame(columns=["date", "close"])
            if raw.empty:
                return {"symbol": symbol, "rows": [], "error": ""}

            if adjusted.empty:
                raw["adjusted_close"] = raw["close"]
            else:
                raw = raw.merge(
                    adjusted.rename(columns={"close": "adjusted_close"}),
                    on="date",
                    how="left",
                )
            if "turn" not in raw.columns:
                raw["turn"] = float("nan")
            if "tradestatus" not in raw.columns:
                raw["tradestatus"] = "1"
            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "turn",
                "adjusted_close",
            ]
            raw[numeric_columns] = raw[numeric_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
            raw = raw.dropna(subset=["open", "high", "low", "close"])
            raw = raw.loc[
                raw[["open", "high", "low", "close"]].gt(0).all(axis=1)
            ].copy()
            raw["adj_factor"] = raw["adjusted_close"] / raw["close"]
            raw["adj_factor"] = raw["adj_factor"].fillna(1.0)
            raw.loc[raw["adj_factor"] <= 0, "adj_factor"] = 1.0

            records: list[dict[str, Any]] = []
            for row in raw.to_dict("records"):
                records.append(
                    {
                        "symbol": symbol,
                        "trade_date": row["date"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "pre_close": row["preclose"],
                        "volume": row["volume"],
                        "amount": row["amount"],
                        "turnover_rate": row["turn"],
                        "adj_factor": row["adj_factor"],
                        "industry_code": industry_code or "UNKNOWN",
                        "is_suspended": 0
                        if str(row.get("tradestatus", "1")) == "1"
                        else 1,
                        "source": "BAOSTOCK",
                    }
                )
            return {"symbol": symbol, "rows": records, "error": ""}
        except Exception as exc:
            last_error = exc
            global _WORKER_LOGGED_IN
            _WORKER_LOGGED_IN = False
            _close_baostock_socket()
            if attempt < retry_count:
                time.sleep(retry_delay * attempt)
                login_error = _login_with_retry(
                    retry_count,
                    retry_delay,
                    socket_timeout=socket_timeout,
                )
                if login_error:
                    last_error = RuntimeError(
                        f"BaoStock worker 重新登录失败：{login_error}"
                    )
    return {
        "symbol": symbol,
        "rows": [],
        "error": str(last_error) if last_error else "未知错误",
    }


def _normalize_akshare_hist(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turn",
        "涨跌额": "change_amount",
    }
    missing = [column for column in ("日期", "开盘", "最高", "最低", "收盘") if column not in frame.columns]
    if missing:
        raise RuntimeError(f"AkShare 日线缺少必要列：{missing}，实际列：{list(frame.columns)}")
    output = frame.rename(columns=rename_map).copy()
    numeric_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turn",
            "change_amount",
        )
        if column in output.columns
    ]
    output[numeric_columns] = output[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output = output.dropna(subset=["date", "open", "high", "low", "close"])
    output["date"] = output["date"].dt.date.astype(str)
    output = output.loc[
        output[["open", "high", "low", "close"]].gt(0).all(axis=1)
    ].copy()
    return output.sort_values("date").reset_index(drop=True)


def _normalize_akshare_sina_etf_hist(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(
            f"AkShare 新浪 ETF 日线缺少必要列：{missing}，"
            f"实际列：{list(frame.columns)}"
        )
    output = frame.copy()
    numeric_columns = [
        column
        for column in ("open", "high", "low", "close", "volume", "amount")
        if column in output.columns
    ]
    output[numeric_columns] = output[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output = output.dropna(subset=["date", "open", "high", "low", "close"])
    output["date"] = output["date"].dt.date.astype(str)
    output = output.loc[
        output[["open", "high", "low", "close"]].gt(0).all(axis=1)
    ].copy()
    return output.sort_values("date").reset_index(drop=True)


def _download_symbol_akshare(
    task: tuple[str, str, str, str, str, int, float, float],
) -> dict[str, Any]:
    try:
        import akshare as ak
    except ImportError:
        return {
            "symbol": task[0],
            "rows": [],
            "error": "缺少 akshare，请先执行 pip install akshare",
        }

    (
        symbol,
        start_date,
        end_date,
        industry_code,
        security_type,
        retry_count,
        retry_delay,
        _socket_timeout,
    ) = task
    code = to_akshare_code(symbol)
    start_value = start_date.replace("-", "")
    end_value = end_date.replace("-", "")
    last_error: Exception | None = None
    for attempt in range(1, retry_count + 1):
        try:
            history_function = (
                ak.fund_etf_hist_em
                if security_type == SECURITY_TYPE_ETF
                else ak.stock_zh_a_hist
            )
            used_sina_fallback = False
            try:
                raw = history_function(
                    symbol=code,
                    period="daily",
                    start_date=start_value,
                    end_date=end_value,
                    adjust="",
                )
                if raw is None or raw.empty:
                    if security_type != SECURITY_TYPE_ETF:
                        return {"symbol": symbol, "rows": [], "error": ""}
                    raise RuntimeError("东方财富 ETF 日线返回空结果")
                raw = _normalize_akshare_hist(raw)
            except Exception as primary_exc:
                if security_type != SECURITY_TYPE_ETF:
                    raise
                sina_symbol = to_baostock_code(symbol).replace(".", "")
                try:
                    raw = ak.fund_etf_hist_sina(symbol=sina_symbol)
                    if raw is None or raw.empty:
                        raise RuntimeError("新浪 ETF 日线返回空结果")
                    raw = _normalize_akshare_sina_etf_hist(raw)
                    raw = raw.loc[
                        raw["date"].between(start_date, end_date)
                    ].copy()
                    used_sina_fallback = True
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "AkShare ETF 历史双接口均失败。"
                        f"东方财富：{primary_exc}；新浪：{fallback_exc}"
                    ) from fallback_exc

            adjusted_close = pd.DataFrame(columns=["date", "adjusted_close"])
            if not used_sina_fallback:
                try:
                    adjusted = history_function(
                        symbol=code,
                        period="daily",
                        start_date=start_value,
                        end_date=end_value,
                        adjust="qfq",
                    )
                    if adjusted is not None and not adjusted.empty:
                        adjusted = _normalize_akshare_hist(adjusted)
                        adjusted_close = adjusted[["date", "close"]].rename(
                            columns={"close": "adjusted_close"}
                        )
                except Exception:
                    adjusted_close = pd.DataFrame(
                        columns=["date", "adjusted_close"]
                    )

            if adjusted_close.empty:
                raw["adjusted_close"] = pd.to_numeric(
                    raw["close"], errors="coerce"
                )
            else:
                raw = raw.merge(adjusted_close, on="date", how="left")
            if "change_amount" in raw.columns:
                raw["pre_close"] = raw["close"] - raw["change_amount"]
            else:
                raw["pre_close"] = raw["close"].shift(1)
            raw["adj_factor"] = raw["adjusted_close"] / raw["close"]
            raw["adj_factor"] = raw["adj_factor"].fillna(1.0)
            raw.loc[raw["adj_factor"] <= 0, "adj_factor"] = 1.0
            records: list[dict[str, Any]] = []
            for row in raw.to_dict("records"):
                records.append(
                    {
                        "symbol": symbol,
                        "trade_date": row["date"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "pre_close": row.get("pre_close"),
                        "volume": row.get("volume"),
                        "amount": row.get("amount"),
                        "turnover_rate": row.get("turn"),
                        "adj_factor": row["adj_factor"],
                        "industry_code": industry_code or "UNKNOWN",
                        "is_suspended": 0,
                        "source": (
                            "AKSHARE_SINA"
                            if used_sina_fallback
                            else "AKSHARE"
                        ),
                    }
                )
            return {"symbol": symbol, "rows": records, "error": ""}
        except Exception as exc:
            last_error = exc
            if attempt < retry_count:
                time.sleep(retry_delay * attempt)
    return {
        "symbol": symbol,
        "rows": [],
        "error": str(last_error) if last_error else "未知错误",
    }


def _chunks(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _database_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sanitize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: _database_value(value) for key, value in record.items()}
        for record in records
    ]


def _completed_symbols(
    engine: Any,
    symbols: list[str],
    end_date: str,
) -> set[str]:
    if not symbols:
        return set()
    query = (
        text(
            """
            SELECT symbol
            FROM stock_daily_bars
            WHERE symbol IN :symbols
            GROUP BY symbol
            HAVING MAX(trade_date) >= :end_date
            """
        )
        .bindparams(bindparam("symbols", expanding=True))
    )
    with engine.connect() as connection:
        return {
            str(row[0])
            for row in connection.execute(
                query,
                {"symbols": symbols, "end_date": end_date},
            )
        }


def _replace_stock_master(
    engine: Any,
    universe: pd.DataFrame,
    symbols: list[str],
) -> int:
    selected = universe.loc[universe["symbol"].isin(symbols)].copy()
    records: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        symbol = str(row["symbol"])
        stock_name = str(row.get("code_name", ""))
        out_date = str(row.get("outDate", "")).strip()
        security_type = str(
            row.get("security_type", SECURITY_TYPE_STOCK)
        ).lower()
        records.append(
            {
                "symbol": symbol,
                "stock_name": stock_name,
                "exchange": symbol.split(".")[1],
                "board": (
                    "ETF"
                    if security_type == SECURITY_TYPE_ETF
                    else _board_for(symbol)
                ),
                "industry_code": str(row.get("industry", "UNKNOWN")),
                "industry_name": str(row.get("industry", "UNKNOWN")),
                "is_st": int("ST" in stock_name.upper()),
                "price_limit_pct": _price_limit(symbol, stock_name),
                "list_date": str(row.get("ipoDate", "")).strip() or None,
                "delist_date": out_date or None,
                "status": "DELISTED" if out_date else "LISTED",
            }
        )
    if not records:
        return 0

    delete_sql = (
        text("DELETE FROM stock_master WHERE symbol IN :symbols")
        .bindparams(bindparam("symbols", expanding=True))
    )
    insert_sql = text(
        """
        INSERT INTO stock_master (
            symbol, stock_name, exchange, board, industry_code, industry_name,
            is_st, price_limit_pct, list_date, delist_date, status
        ) VALUES (
            :symbol, :stock_name, :exchange, :board, :industry_code,
            :industry_name, :is_st, :price_limit_pct, :list_date,
            :delist_date, :status
        )
        """
    )
    with engine.begin() as connection:
        connection.execute(delete_sql, {"symbols": symbols})
        connection.execute(insert_sql, records)
    return len(records)


def _replace_bars(
    engine: Any,
    symbol: str,
    start_date: str,
    end_date: str,
    records: list[dict[str, Any]],
    batch_size: int,
) -> int:
    records = _sanitize_records(records)
    delete_sql = text(
        """
        DELETE FROM stock_daily_bars
        WHERE symbol = :symbol
          AND trade_date BETWEEN :start_date AND :end_date
        """
    )
    insert_sql = text(
        """
        INSERT INTO stock_daily_bars (
            symbol, trade_date, open, high, low, close, pre_close, volume,
            amount, turnover_rate, adj_factor, industry_code, is_suspended,
            source
        ) VALUES (
            :symbol, :trade_date, :open, :high, :low, :close, :pre_close,
            :volume, :amount, :turnover_rate, :adj_factor, :industry_code,
            :is_suspended, :source
        )
        """
    )
    with engine.begin() as connection:
        connection.execute(
            delete_sql,
            {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        for batch in _chunks(records, batch_size):
            connection.execute(insert_sql, batch)
    return len(records)


def sync_market_data(
    engine: Any,
    config: AppConfig,
    end_date: date,
    scope: str,
    start_date: date | None = None,
) -> dict[str, Any]:
    effective_start = (
        start_date.isoformat()
        if start_date is not None
        else config.data_sync.start_date
    )
    effective_end = end_date.isoformat()
    if scope == "positions":
        positions = load_positions(engine, config)
        symbols = sorted(positions["symbol"].astype(str).unique().tolist())
    elif scope == "all":
        positions = pd.DataFrame()
        symbols = []
    else:
        raise ValueError("scope 必须是 positions 或 all")

    include_etfs = scope == "positions" and any(
        _looks_like_etf_symbol(symbol) for symbol in symbols
    )
    universe, source_used = fetch_universe(
        config.data_sync.source,
        end_date,
        config.data_sync.retry_count,
        config.data_sync.retry_delay_seconds,
        config.data_sync.socket_timeout_seconds,
        include_etfs=include_etfs,
    )

    if scope == "all":
        symbols = sorted(
            universe.loc[
                universe["security_type"].eq(SECURITY_TYPE_STOCK), "symbol"
            ]
            .astype(str)
            .unique()
            .tolist()
        )
    if not symbols:
        return {
            "source": source_used,
            "scope": scope,
            "symbols": 0,
            "successful_symbols": 0,
            "failed_symbols": 0,
            "bar_rows": 0,
        }

    universe_symbols = set(universe["symbol"])
    missing = sorted(set(symbols).difference(universe_symbols))
    if missing:
        raise RuntimeError(f"数据源中找不到股票：{missing}")

    _replace_stock_master(engine, universe, symbols)
    industry_map = dict(zip(universe["symbol"], universe["industry"]))
    security_type_map = dict(
        zip(universe["symbol"], universe["security_type"])
    )
    completed_symbols = _completed_symbols(engine, symbols, effective_end)
    pending_symbols = [
        symbol for symbol in symbols if symbol not in completed_symbols
    ]
    tasks = [
        (
            symbol,
            effective_start,
            effective_end,
            str(industry_map.get(symbol, "UNKNOWN")),
            str(security_type_map.get(symbol, SECURITY_TYPE_STOCK)),
            config.data_sync.retry_count,
            config.data_sync.retry_delay_seconds,
            config.data_sync.socket_timeout_seconds,
        )
        for symbol in pending_symbols
    ]

    successful = 0
    failed: dict[str, str] = {}
    bar_rows = 0
    if not tasks:
        return {
            "source": source_used,
            "scope": scope,
            "start_date": effective_start,
            "end_date": effective_end,
            "workers": config.performance.workers,
            "symbols": len(symbols),
            "skipped_complete_symbols": len(completed_symbols),
            "attempted_symbols": 0,
            "successful_symbols": 0,
            "failed_symbols": 0,
            "bar_rows": 0,
            "errors": {},
        }
    download_function: Callable[
        [tuple[str, str, str, str, str, int, float, float]],
        dict[str, Any],
    ] = (
        _download_symbol_akshare
        if source_used == "akshare"
        else _download_symbol_baostock
    )
    executor_kwargs: dict[str, Any] = {"max_workers": config.performance.workers}
    if source_used == "baostock":
        executor_kwargs.update(
            {
                "initializer": _worker_login,
                "initargs": (
                    config.data_sync.retry_count,
                    config.data_sync.retry_delay_seconds,
                    config.data_sync.socket_timeout_seconds,
                ),
            }
        )
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = {
            executor.submit(download_function, task): task[0]
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failed[symbol] = str(exc)
                continue
            if result["error"]:
                failed[symbol] = result["error"]
                continue
            inserted = _replace_bars(
                engine,
                symbol,
                effective_start,
                effective_end,
                result["rows"],
                config.data_sync.batch_size,
            )
            successful += 1
            bar_rows += inserted
            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"行情同步进度 {completed}/{len(futures)}，"
                    f"成功 {successful}，失败 {len(failed)}"
                )

    return {
        "source": source_used,
        "scope": scope,
        "start_date": effective_start,
        "end_date": effective_end,
        "workers": config.performance.workers,
        "symbols": len(symbols),
        "skipped_complete_symbols": len(completed_symbols),
        "attempted_symbols": len(pending_symbols),
        "successful_symbols": successful,
        "failed_symbols": len(failed),
        "bar_rows": bar_rows,
        "errors": failed,
    }
