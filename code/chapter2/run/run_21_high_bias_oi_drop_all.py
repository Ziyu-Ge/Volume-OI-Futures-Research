from factor_batch_runner import FactorRunConfig, run_factor_all


if __name__ == "__main__":
    run_factor_all(
        FactorRunConfig(
            factor_id="21",
            factor_name="high_bias_oi_drop",
            factor_script_name="21_high_bias_oi_drop.py",
            output_dir_name="21_high_bias_oi_drop_all_symbols",
            prepare_script_name="01_prepare_data.py",
        )
    )
