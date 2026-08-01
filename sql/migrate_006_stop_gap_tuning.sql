ALTER TABLE tpsl_stop_tuning_results
    ADD COLUMN minimum_stop_gap_pct DECIMAL(10, 6) NOT NULL DEFAULT 0.005
        COMMENT '相对收盘价的最小止损距离'
        AFTER limit_slippage_pct,
    DROP INDEX idx_stop_tuning_results_params,
    ADD KEY idx_stop_tuning_results_params (
        stop_order_type, atr_multiplier, limit_slippage_pct,
        minimum_stop_gap_pct
    );
