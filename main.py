from eruption_forecast.calculate import Calculate

def main():
    calculate = Calculate(
        station="OJN",
        channel="EHN",
        start_date="2025-11-05",
        end_date="2025-11-10",
        n_jobs=2,
        verbose=True,
        debug=True,
    )

    calculate.from_sds(
        sds_dir=r"D:\Data\OJN"
    ).run()

if __name__ == "__main__":
    main()
