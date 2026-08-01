ALTER TABLE tpsl_recommendations
    ADD COLUMN dynamic_stop_gap_pct DECIMAL(14, 8) NULL
        COMMENT '动态风控模型建议的最小止损距离'
        AFTER stop_limit_price,
    ADD COLUMN risk_model_version VARCHAR(64) NULL
        COMMENT '动态风控模型版本'
        AFTER model_version;
