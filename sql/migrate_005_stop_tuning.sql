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
