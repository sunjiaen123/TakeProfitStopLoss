ALTER TABLE tpsl_recommendations
    MODIFY COLUMN strategy_profile VARCHAR(64) NULL
        COMMENT '当前启用的策略配置名称',
    MODIFY COLUMN chart_exit_enabled TINYINT(1) NOT NULL DEFAULT 0
        COMMENT '是否启用K线退出策略，0否，1是',
    MODIFY COLUMN chart_exit_action VARCHAR(32) NULL
        COMMENT 'K线退出策略建议动作',
    MODIFY COLUMN chart_exit_reason VARCHAR(64) NULL
        COMMENT 'K线退出策略动作原因',
    MODIFY COLUMN chart_exit_stop_trigger_price DECIMAL(20, 6) NULL
        COMMENT 'S3盘中保命止损触发价',
    MODIFY COLUMN chart_exit_stop_limit_price DECIMAL(20, 6) NULL
        COMMENT 'S3盘中保命止损限价',
    MODIFY COLUMN chart_exit_position_scale DECIMAL(10, 6) NULL
        COMMENT 'S3策略目标仓位系数',
    MODIFY COLUMN chart_exit_held_peak_close DECIMAL(20, 6) NULL
        COMMENT '持仓期间最高收盘价',
    MODIFY COLUMN chart_exit_trend_active TINYINT(1) NULL
        COMMENT '是否已进入趋势持有阶段，0否，1是',
    MODIFY COLUMN chart_exit_trade_days INT NULL
        COMMENT '从建仓日起计算的持仓交易日数',
    MODIFY COLUMN chart_exit_diagnostic VARCHAR(500) NULL
        COMMENT 'K线退出策略数据诊断信息',
    ADD COLUMN chart_exit_initial_stop_price DECIMAL(20, 6) NULL
        COMMENT '建仓阶段结构止损参考价'
        AFTER chart_exit_trade_days,
    ADD COLUMN chart_exit_progress_price DECIMAL(20, 6) NULL
        COMMENT '进入趋势阶段需要达到的进度价格'
        AFTER chart_exit_initial_stop_price,
    ADD COLUMN chart_exit_profit_floor_price DECIMAL(20, 6) NULL
        COMMENT '根据持仓最高收盘价计算的分段利润底'
        AFTER chart_exit_progress_price,
    ADD COLUMN chart_exit_entry_swing_low DECIMAL(20, 6) NULL
        COMMENT '建仓时已确认的摆动低点'
        AFTER chart_exit_profit_floor_price,
    ADD COLUMN chart_exit_ma_fast_price DECIMAL(20, 6) NULL
        COMMENT '当前快速均线价格'
        AFTER chart_exit_entry_swing_low,
    ADD COLUMN chart_exit_ma_trend_price DECIMAL(20, 6) NULL
        COMMENT '当前趋势均线价格'
        AFTER chart_exit_ma_fast_price,
    ADD COLUMN chart_exit_ma_long_price DECIMAL(20, 6) NULL
        COMMENT '当前长期均线价格'
        AFTER chart_exit_ma_trend_price,
    ADD COLUMN chart_exit_ma_trend_slope DECIMAL(14, 8) NULL
        COMMENT '趋势均线在配置回看期内的变化率'
        AFTER chart_exit_ma_long_price,
    ADD COLUMN chart_exit_recent_swing_low DECIMAL(20, 6) NULL
        COMMENT '当前最近已确认的摆动低点'
        AFTER chart_exit_ma_trend_slope;
