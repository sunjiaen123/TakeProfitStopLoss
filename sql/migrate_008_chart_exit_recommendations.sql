ALTER TABLE tpsl_recommendations
    ADD COLUMN strategy_profile VARCHAR(64) NULL
        COMMENT 'strategy profile name'
        AFTER risk_model_version,
    ADD COLUMN chart_exit_enabled TINYINT(1) NOT NULL DEFAULT 0
        COMMENT 'chart exit enabled flag'
        AFTER strategy_profile,
    ADD COLUMN chart_exit_action VARCHAR(32) NULL
        COMMENT 'chart exit action'
        AFTER chart_exit_enabled,
    ADD COLUMN chart_exit_reason VARCHAR(64) NULL
        COMMENT 'chart exit reason'
        AFTER chart_exit_action,
    ADD COLUMN chart_exit_stop_trigger_price DECIMAL(20, 6) NULL
        COMMENT 'chart hard stop trigger'
        AFTER chart_exit_reason,
    ADD COLUMN chart_exit_stop_limit_price DECIMAL(20, 6) NULL
        COMMENT 'chart hard stop limit'
        AFTER chart_exit_stop_trigger_price,
    ADD COLUMN chart_exit_position_scale DECIMAL(10, 6) NULL
        COMMENT 'entry position scale from chart profile'
        AFTER chart_exit_stop_limit_price,
    ADD COLUMN chart_exit_held_peak_close DECIMAL(20, 6) NULL
        COMMENT 'held peak close used by chart exit'
        AFTER chart_exit_position_scale,
    ADD COLUMN chart_exit_trend_active TINYINT(1) NULL
        COMMENT 'chart trend state flag'
        AFTER chart_exit_held_peak_close,
    ADD COLUMN chart_exit_trade_days INT NULL
        COMMENT 'chart trade days from entry'
        AFTER chart_exit_trend_active,
    ADD COLUMN chart_exit_diagnostic VARCHAR(500) NULL
        COMMENT 'chart exit diagnostics'
        AFTER chart_exit_trade_days;
