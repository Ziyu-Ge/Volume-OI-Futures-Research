from factor_batch_runner import FactorRunConfig, run_factor_all


if __name__ == "__main__":
    run_factor_all(
        FactorRunConfig(
            factor_id="12",
            factor_name="price_up_speculation_up",
            factor_script_name="12price_up_speculation_up.py",
            output_dir_name="12_price_up_speculation_up_all_symbols",
        )
    )
