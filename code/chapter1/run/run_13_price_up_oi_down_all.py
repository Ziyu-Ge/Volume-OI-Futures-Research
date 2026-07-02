from factor_batch_runner import FactorRunConfig, run_factor_all


if __name__ == "__main__":
    run_factor_all(
        FactorRunConfig(
            factor_id="13",
            factor_name="price_up_oi_down",
            factor_script_name="13price_up_oi_down.py",
            output_dir_name="13_price_up_oi_down_all_symbols",
        )
    )
