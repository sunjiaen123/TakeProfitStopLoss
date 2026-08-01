ALTER TABLE tpsl_recommendations
    MODIFY COLUMN take_profit_price DECIMAL(20, 6) NULL COMMENT '建议止盈价格，不建议止盈时为空',
    ADD COLUMN take_profit_enabled TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否建议设置止盈，0 否，1 是'
        AFTER take_profit_price,
    ADD COLUMN take_profit_reason VARCHAR(500) NOT NULL DEFAULT '' COMMENT '是否设置止盈的原因'
        AFTER take_profit_enabled;

ALTER TABLE tpsl_backtest_trades
    MODIFY COLUMN take_profit_price DECIMAL(20, 6) NULL COMMENT '止盈价格，不设置时为空',
    ADD COLUMN take_profit_enabled TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否设置止盈，0 否，1 是'
        AFTER take_profit_price;
