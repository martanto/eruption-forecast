from eruption_forecast.calculate import Calculate

def main():
    calculate = Calculate(
        station="OJN",
        channel="EHZ",
        start_date="2025-01-01",
        end_date="2025-01-03",
        window_size=2,
        day_to_forecast=2,
        remove_outliers=True,
        value_multiplier=1,
        n_jobs=1,
        overwrite=True,
        verbose=True,
        debug=False,
    )

    calculate.from_sds(
        sds_dir=r"D:\Data\OJN"
    ).run()

if __name__ == "__main__":
    main()
