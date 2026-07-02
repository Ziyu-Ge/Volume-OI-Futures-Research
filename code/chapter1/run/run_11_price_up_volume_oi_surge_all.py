from factor_batch_runner import FactorRunConfig, run_factor_all


if __name__ == "__main__":
    run_factor_all(
        FactorRunConfig(
            factor_id="11",
            factor_name="price_up_volume_oi_surge",
            factor_script_name="11price_up_volume_oi_surge.py",
            output_dir_name="11_price_up_volume_oi_surge_all_symbols",
        )
    )
