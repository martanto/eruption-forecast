from eruption_forecast.calculate import Calculate

def main():
    calculate = Calculate(
        station="OJN",
        channel="EHZ",
        start_date="2025-01-01",
        end_date="2025-01-31",
        n_jobs=2,
        verbose=True,
        debug=True,
    )

    calculate.from_sds(
        sds_dir=r"D:\Data\OJN"
    )

    calculate.run()

if __name__ == "__main__":
    main()

