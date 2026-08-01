-- TakeProfitStopLoss MySQL 8.0 数据库结构。
-- 所有表使用 id 自增主键，业务查询字段只建立普通索引。

CREATE TABLE IF NOT EXISTS stock_master (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    symbol VARCHAR(24) NOT NULL COMMENT '股票统一代码，如 600519.SH',
    stock_name VARCHAR(128) NULL COMMENT '股票名称',
    exchange VARCHAR(16) NULL COMMENT '交易所代码',
    board VARCHAR(32) NULL COMMENT '所属板块，如主板、科创板、创业板、北交所',
    industry_code VARCHAR(64) NULL COMMENT '行业代码',
    industry_name VARCHAR(128) NULL COMMENT '行业名称',
    is_st TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否为 ST 股票，0 否，1 是',
    price_limit_pct DECIMAL(10, 6) NULL COMMENT '当日涨跌停比例',
    list_date DATE NULL COMMENT '上市日期',
    delist_date DATE NULL COMMENT '退市日期',
    status VARCHAR(24) NOT NULL DEFAULT 'LISTED' COMMENT '股票状态',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (id),
    KEY idx_stock_master_symbol (symbol),
    KEY idx_stock_master_industry (industry_code),
    KEY idx_stock_master_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='股票基础信息表';

CREATE TABLE IF NOT EXISTS stock_positions (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    symbol VARCHAR(24) NOT NULL COMMENT '股票统一代码',
    quantity DECIMAL(20, 4) NOT NULL COMMENT '当前持仓数量',
    available_quantity DECIMAL(20, 4) NULL COMMENT '可卖持仓数量',
    avg_cost DECIMAL(20, 6) NOT NULL COMMENT '持仓平均成本',
    entry_date DATE NULL COMMENT '首次建仓日期',
    max_loss_pct DECIMAL(10, 6) NULL COMMENT '该持仓最大允许亏损比例',
    current_stop DECIMAL(20, 6) NULL COMMENT '当前已设置的止损价格',
    status VARCHAR(24) NOT NULL DEFAULT 'HOLDING' COMMENT '持仓状态',
    source VARCHAR(32) NOT NULL DEFAULT 'MANUAL' COMMENT '持仓数据来源',
    note VARCHAR(500) NULL COMMENT '备注',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (id),
    KEY idx_positions_symbol (symbol),
    KEY idx_positions_status_symbol (status, symbol),
    CONSTRAINT chk_positions_quantity CHECK (quantity >= 0),
    CONSTRAINT chk_positions_avg_cost CHECK (avg_cost > 0),
    CONSTRAINT chk_positions_max_loss CHECK (
        max_loss_pct IS NULL OR (max_loss_pct > 0 AND max_loss_pct < 1)
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='当前持仓表';

CREATE TABLE IF NOT EXISTS stock_daily_bars (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    symbol VARCHAR(24) NOT NULL COMMENT '股票统一代码',
    trade_date DATE NOT NULL COMMENT '交易日期',
    open DECIMAL(20, 6) NOT NULL COMMENT '开盘价，原始可交易价格',
    high DECIMAL(20, 6) NOT NULL COMMENT '最高价，原始可交易价格',
    low DECIMAL(20, 6) NOT NULL COMMENT '最低价，原始可交易价格',
    close DECIMAL(20, 6) NOT NULL COMMENT '收盘价，原始可交易价格',
    pre_close DECIMAL(20, 6) NULL COMMENT '前一交易日收盘价',
    volume DECIMAL(24, 4) NULL COMMENT '成交量',
    amount DECIMAL(24, 4) NULL COMMENT '成交额',
    turnover_rate DECIMAL(14, 8) NULL COMMENT '换手率',
    adj_factor DECIMAL(24, 10) NOT NULL DEFAULT 1.0 COMMENT '复权因子',
    industry_code VARCHAR(64) NULL COMMENT '交易当日所属行业代码',
    is_suspended TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否停牌，0 否，1 是',
    source VARCHAR(32) NOT NULL DEFAULT 'UNKNOWN' COMMENT '行情数据来源',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (id),
    KEY idx_bars_symbol_date (symbol, trade_date),
    KEY idx_bars_date_industry (trade_date, industry_code),
    KEY idx_bars_trade_date (trade_date),
    CONSTRAINT chk_bars_prices CHECK (
        open > 0 AND high > 0 AND low > 0 AND close > 0
        AND high >= low
    ),
    CONSTRAINT chk_bars_adj_factor CHECK (adj_factor > 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='股票日线行情表';

CREATE TABLE IF NOT EXISTS tpsl_recommendations (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    as_of_date DATE NOT NULL COMMENT '建议所依据的交易日期',
    symbol VARCHAR(24) NOT NULL COMMENT '股票统一代码',
    close_price DECIMAL(20, 6) NOT NULL COMMENT '当日收盘价',
    avg_cost DECIMAL(20, 6) NOT NULL COMMENT '持仓平均成本',
    take_profit_price DECIMAL(20, 6) NULL COMMENT '建议止盈价格，不建议止盈时为空',
    take_profit_enabled TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否建议设置止盈，0 否，1 是',
    take_profit_reason VARCHAR(500) NOT NULL COMMENT '是否设置止盈的原因',
    stop_trigger_price DECIMAL(20, 6) NOT NULL COMMENT '建议止损触发价格',
    stop_limit_price DECIMAL(20, 6) NOT NULL COMMENT '止损触发后的建议限价',
    dynamic_stop_gap_pct DECIMAL(14, 8) NULL COMMENT '动态风控模型建议的最小止损距离',
    predicted_high_return DECIMAL(14, 8) NOT NULL COMMENT '模型预测次日最高收益率',
    predicted_low_return DECIMAL(14, 8) NOT NULL COMMENT '模型预测次日最低收益率',
    similar_high_return DECIMAL(14, 8) NULL COMMENT '相似形态次日最高收益率统计值',
    similar_low_return DECIMAL(14, 8) NULL COMMENT '相似形态次日最低收益率统计值',
    risk_reward_ratio DECIMAL(14, 6) NULL COMMENT '预期盈亏比',
    confidence DECIMAL(10, 6) NOT NULL COMMENT '内部置信度评分',
    sample_count INT NOT NULL COMMENT '相似形态样本数量',
    model_version VARCHAR(64) NOT NULL COMMENT '模型版本',
    risk_model_version VARCHAR(64) NULL COMMENT '动态风控模型版本',
    reason VARCHAR(1000) NOT NULL COMMENT '建议理由和风险说明',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (id),
    KEY idx_tpsl_date_symbol_model (as_of_date, symbol, model_version),
    KEY idx_tpsl_symbol_date (symbol, as_of_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='每日止盈止损建议表';

CREATE TABLE IF NOT EXISTS tpsl_model_runs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    model_version VARCHAR(64) NOT NULL COMMENT '模型版本',
    train_end_date DATE NOT NULL COMMENT '训练数据截止日期',
    status VARCHAR(24) NOT NULL COMMENT '训练状态',
    device VARCHAR(64) NULL COMMENT '实际训练设备',
    training_rows BIGINT NULL COMMENT '训练样本数量',
    validation_rows BIGINT NULL COMMENT '验证样本数量',
    metrics_json JSON NULL COMMENT '模型评估指标',
    error_message VARCHAR(2000) NULL COMMENT '训练失败错误信息',
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '训练开始时间',
    finished_at DATETIME NULL COMMENT '训练结束时间',
    PRIMARY KEY (id),
    KEY idx_model_runs_version (model_version),
    KEY idx_model_runs_date_status (train_end_date, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='模型训练运行记录表';

CREATE TABLE IF NOT EXISTS tpsl_backtest_runs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    run_id VARCHAR(64) NOT NULL COMMENT '回测运行编号',
    start_date DATE NOT NULL COMMENT '回测开始日期',
    end_date DATE NOT NULL COMMENT '回测结束日期',
    retrain_frequency VARCHAR(24) NOT NULL COMMENT '模型重训频率',
    execution_mode VARCHAR(24) NOT NULL COMMENT '成交模拟模式',
    stop_order_type VARCHAR(24) NOT NULL COMMENT '止损订单类型',
    symbol_count INT NOT NULL COMMENT '回测股票数量',
    fold_count INT NOT NULL DEFAULT 0 COMMENT '滚动训练批次数量',
    status VARCHAR(24) NOT NULL COMMENT '回测状态',
    config_json JSON NULL COMMENT '回测配置',
    metrics_json JSON NULL COMMENT '回测汇总指标',
    error_message VARCHAR(2000) NULL COMMENT '失败错误信息',
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开始时间',
    finished_at DATETIME NULL COMMENT '结束时间',
    PRIMARY KEY (id),
    KEY idx_backtest_runs_run_id (run_id),
    KEY idx_backtest_runs_dates (start_date, end_date),
    KEY idx_backtest_runs_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='滚动回测运行记录表';

CREATE TABLE IF NOT EXISTS tpsl_backtest_trades (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    run_id VARCHAR(64) NOT NULL COMMENT '回测运行编号',
    strategy VARCHAR(32) NOT NULL COMMENT '策略名称',
    fold_train_end_date DATE NOT NULL COMMENT '该批模型训练截止日期',
    signal_date DATE NOT NULL COMMENT '生成条件单的交易日期',
    execution_date DATE NOT NULL COMMENT '模拟成交日期',
    symbol VARCHAR(24) NOT NULL COMMENT '股票统一代码',
    close_price DECIMAL(20, 6) NOT NULL COMMENT '信号日收盘价',
    take_profit_price DECIMAL(20, 6) NULL COMMENT '止盈价格，不设置时为空',
    take_profit_enabled TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否设置止盈，0 否，1 是',
    stop_trigger_price DECIMAL(20, 6) NOT NULL COMMENT '止损触发价格',
    stop_limit_price DECIMAL(20, 6) NOT NULL COMMENT '止损限价',
    next_open DECIMAL(20, 6) NOT NULL COMMENT '次日开盘价',
    next_high DECIMAL(20, 6) NOT NULL COMMENT '次日最高价',
    next_low DECIMAL(20, 6) NOT NULL COMMENT '次日最低价',
    next_close DECIMAL(20, 6) NOT NULL COMMENT '次日收盘价',
    exit_price DECIMAL(20, 6) NOT NULL COMMENT '模拟退出或估值价格',
    outcome VARCHAR(40) NOT NULL COMMENT '模拟结果类型',
    return_pct DECIMAL(14, 8) NOT NULL COMMENT '单次收益率',
    confidence DECIMAL(10, 6) NULL COMMENT '模型内部置信度',
    risk_reward_ratio DECIMAL(14, 6) NULL COMMENT '建议盈亏比',
    stop_limit_unfilled TINYINT(1) NOT NULL DEFAULT 0 COMMENT '止损限价是否未成交',
    dual_hit TINYINT(1) NOT NULL DEFAULT 0 COMMENT '次日是否同时触及止盈止损',
    gap_stop TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否跳空触发止损',
    model_version VARCHAR(64) NULL COMMENT '模型版本',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (id),
    KEY idx_backtest_trades_run_strategy (run_id, strategy),
    KEY idx_backtest_trades_signal_date (signal_date),
    KEY idx_backtest_trades_symbol_date (symbol, signal_date),
    KEY idx_backtest_trades_outcome (outcome)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='滚动回测交易明细表';

CREATE TABLE IF NOT EXISTS tpsl_holding_backtest_runs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    run_id VARCHAR(64) NOT NULL COMMENT '持仓回测运行编号',
    start_date DATE NOT NULL COMMENT '最早建仓日期',
    end_date DATE NOT NULL COMMENT '回测截止日期',
    stop_order_type VARCHAR(24) NOT NULL COMMENT '止损订单类型',
    position_count INT NOT NULL COMMENT '回测持仓数量',
    status VARCHAR(24) NOT NULL COMMENT '运行状态',
    config_json JSON NULL COMMENT '回测配置',
    metrics_json JSON NULL COMMENT '汇总指标',
    error_message VARCHAR(2000) NULL COMMENT '失败错误信息',
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开始时间',
    finished_at DATETIME NULL COMMENT '结束时间',
    PRIMARY KEY (id),
    KEY idx_holding_runs_run_id (run_id),
    KEY idx_holding_runs_dates (start_date, end_date),
    KEY idx_holding_runs_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='真实持仓多日回测运行记录表';

CREATE TABLE IF NOT EXISTS tpsl_holding_backtest_positions (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    run_id VARCHAR(64) NOT NULL COMMENT '持仓回测运行编号',
    symbol VARCHAR(24) NOT NULL COMMENT '股票统一代码',
    entry_date DATE NOT NULL COMMENT '建仓日期',
    avg_cost DECIMAL(20, 6) NOT NULL COMMENT '建仓平均成本',
    quantity DECIMAL(20, 4) NOT NULL COMMENT '持仓数量',
    exit_date DATE NULL COMMENT '模拟退出日期',
    exit_price DECIMAL(20, 6) NULL COMMENT '模拟退出价格',
    exit_outcome VARCHAR(40) NOT NULL COMMENT '退出或持有结果',
    final_stop_price DECIMAL(20, 6) NULL COMMENT '最终有效止损触发价',
    strategy_return DECIMAL(14, 8) NOT NULL COMMENT '动态止损策略收益率',
    hold_return DECIMAL(14, 8) NOT NULL COMMENT '不止损持有收益率',
    excess_return DECIMAL(14, 8) NOT NULL COMMENT '动态止损相对持有超额收益',
    strategy_max_drawdown DECIMAL(14, 8) NOT NULL COMMENT '动态止损最大回撤',
    hold_max_drawdown DECIMAL(14, 8) NOT NULL COMMENT '不止损持有最大回撤',
    strategy_max_loss_from_cost DECIMAL(14, 8) NOT NULL COMMENT '动态止损相对成本最大亏损',
    hold_max_loss_from_cost DECIMAL(14, 8) NOT NULL COMMENT '持有相对成本最大亏损',
    stop_update_count INT NOT NULL COMMENT '止损上调次数',
    stop_limit_unfilled_count INT NOT NULL COMMENT '止损限价未成交次数',
    trading_days INT NOT NULL COMMENT '参与回测交易日数量',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (id),
    KEY idx_holding_positions_run_symbol (run_id, symbol),
    KEY idx_holding_positions_entry_date (entry_date),
    KEY idx_holding_positions_outcome (exit_outcome)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='真实持仓多日回测汇总表';

CREATE TABLE IF NOT EXISTS tpsl_holding_backtest_daily (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    run_id VARCHAR(64) NOT NULL COMMENT '持仓回测运行编号',
    symbol VARCHAR(24) NOT NULL COMMENT '股票统一代码',
    signal_date DATE NOT NULL COMMENT '生成止损建议的交易日期',
    execution_date DATE NULL COMMENT '止损建议对应的下一交易日',
    close_price DECIMAL(20, 6) NOT NULL COMMENT '信号日收盘价',
    proposed_stop_price DECIMAL(20, 6) NOT NULL COMMENT '当日模型建议止损触发价',
    active_stop_price DECIMAL(20, 6) NOT NULL COMMENT '只升不降后的有效止损触发价',
    stop_limit_price DECIMAL(20, 6) NOT NULL COMMENT '有效止损限价',
    next_open DECIMAL(20, 6) NULL COMMENT '下一交易日开盘价',
    next_high DECIMAL(20, 6) NULL COMMENT '下一交易日最高价',
    next_low DECIMAL(20, 6) NULL COMMENT '下一交易日最低价',
    next_close DECIMAL(20, 6) NULL COMMENT '下一交易日收盘价',
    event VARCHAR(40) NOT NULL COMMENT '当日回测事件',
    strategy_equity DECIMAL(20, 8) NOT NULL COMMENT '动态止损策略净值',
    hold_equity DECIMAL(20, 8) NOT NULL COMMENT '不止损持有净值',
    confidence DECIMAL(10, 6) NOT NULL COMMENT '模型内部置信度',
    model_version VARCHAR(64) NOT NULL COMMENT '月度模型版本',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (id),
    KEY idx_holding_daily_run_symbol (run_id, symbol),
    KEY idx_holding_daily_signal_date (signal_date),
    KEY idx_holding_daily_event (event)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='真实持仓多日回测每日路径表';

CREATE TABLE IF NOT EXISTS tpsl_stop_tuning_runs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    run_id VARCHAR(64) NOT NULL COMMENT '止损参数扫描运行编号',
    entry_start_date DATE NOT NULL COMMENT '模拟建仓开始日期',
    entry_end_date DATE NOT NULL COMMENT '模拟建仓结束日期',
    path_end_date DATE NOT NULL COMMENT '最长持仓路径截止日期',
    symbol_count INT NOT NULL COMMENT '扫描股票数量',
    entry_count INT NOT NULL COMMENT '模拟建仓数量',
    combination_count INT NOT NULL COMMENT '参数组合数量',
    status VARCHAR(24) NOT NULL COMMENT '运行状态',
    config_json JSON NULL COMMENT '参数扫描配置',
    metrics_json JSON NULL COMMENT '最优参数和汇总指标',
    error_message VARCHAR(2000) NULL COMMENT '失败错误信息',
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开始时间',
    finished_at DATETIME NULL COMMENT '结束时间',
    PRIMARY KEY (id),
    KEY idx_stop_tuning_runs_run_id (run_id),
    KEY idx_stop_tuning_runs_dates (entry_start_date, entry_end_date),
    KEY idx_stop_tuning_runs_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='止损参数扫描运行记录表';

CREATE TABLE IF NOT EXISTS tpsl_stop_tuning_results (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    run_id VARCHAR(64) NOT NULL COMMENT '止损参数扫描运行编号',
    holding_days INT NOT NULL COMMENT '持有交易日数量，0 表示跨周期综合',
    stop_order_type VARCHAR(24) NOT NULL COMMENT '止损订单类型',
    atr_multiplier DECIMAL(10, 4) NOT NULL COMMENT 'ATR 止损倍数',
    limit_slippage_pct DECIMAL(10, 6) NOT NULL COMMENT '止损限价滑点比例',
    minimum_stop_gap_pct DECIMAL(10, 6) NOT NULL COMMENT '相对收盘价的最小止损距离',
    path_count INT NOT NULL COMMENT '模拟持仓路径数量',
    stopped_rate DECIMAL(14, 8) NOT NULL COMMENT '止损退出比例',
    stop_limit_unfilled_rate DECIMAL(14, 8) NOT NULL COMMENT '止损限价未成交比例',
    average_strategy_return DECIMAL(14, 8) NOT NULL COMMENT '止损策略平均收益率',
    average_hold_return DECIMAL(14, 8) NOT NULL COMMENT '持有基准平均收益率',
    average_excess_return DECIMAL(14, 8) NOT NULL COMMENT '相对持有平均超额收益',
    p05_strategy_return DECIMAL(14, 8) NOT NULL COMMENT '止损策略收益率百分之五分位',
    median_strategy_return DECIMAL(14, 8) NOT NULL COMMENT '止损策略收益率中位数',
    average_strategy_max_drawdown DECIMAL(14, 8) NOT NULL COMMENT '止损策略平均最大回撤',
    average_hold_max_drawdown DECIMAL(14, 8) NOT NULL COMMENT '持有基准平均最大回撤',
    average_drawdown_reduction DECIMAL(14, 8) NOT NULL COMMENT '平均回撤改善',
    average_strategy_max_loss DECIMAL(14, 8) NOT NULL COMMENT '止损策略相对成本平均最大亏损',
    average_hold_max_loss DECIMAL(14, 8) NOT NULL COMMENT '持有基准相对成本平均最大亏损',
    average_max_loss_reduction DECIMAL(14, 8) NOT NULL COMMENT '平均最大亏损改善',
    loss_avoidance_rate DECIMAL(14, 8) NOT NULL COMMENT '止损后优于持有的比例',
    whipsaw_rate DECIMAL(14, 8) NOT NULL COMMENT '止损后持有明显反弹的比例',
    rank_score DECIMAL(14, 6) NULL COMMENT '综合排名分数，越低越好',
    rank_no INT NULL COMMENT '综合排名',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (id),
    KEY idx_stop_tuning_results_run_rank (run_id, rank_no),
    KEY idx_stop_tuning_results_params (
        stop_order_type, atr_multiplier, limit_slippage_pct,
        minimum_stop_gap_pct
    ),
    KEY idx_stop_tuning_results_holding_days (holding_days)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='止损参数扫描结果表';
