from factor_batch_runner import FactorRunConfig, run_factor_all


if __name__ == "__main__":
    run_factor_all(
        FactorRunConfig(
            factor_id="14",
            factor_name="uptrend_crowded_chase",
            factor_script_name="14uptrend_crowded_chase.py",
            output_dir_name="14_uptrend_crowded_chase_all_symbols",
        )
    )
