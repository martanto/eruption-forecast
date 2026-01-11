from eruption_forecast.calculate import Calculate

def main():
    calculate = Calculate(
        station="OJN",
        channel="EHN",
        start_date="2025-07-01",
        end_date="2025-07-31",
        window_size=2,
        day_to_forecast=2,
        remove_outliers=True,
        # value_multiplier=1.e9,
        n_jobs=4,
        overwrite=True,
        verbose=True,
        debug=True,
    )

    calculate.from_sds(
        sds_dir=r"D:\Data\OJN"
    ).run()

if __name__ == "__main__":
    main()
