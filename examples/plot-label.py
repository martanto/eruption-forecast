from eruption_forecast.plots.label_plots import (
    LabelScenarioFileEntry,
    plot_label_distribution_comparison_from_files,
)


def main():
    labels: list[LabelScenarioFileEntry] = [
        {
            "name": "Scenario 1",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-1\training\labels\eruption_dates_volcano-2025-01-01-2025-03-31.csv",
        },
        {
            "name": "Scenario 2",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-2\evaluation\prediction\labels\y_true.csv",
        },
        {
            "name": "Scenario 3",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-3\evaluation\prediction\labels\y_true.csv",
        },
        {
            "name": "Scenario 4",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-4\evaluation\prediction\labels\y_true.csv",
        },
        {
            "name": "Scenario 5",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-5\evaluation\prediction\labels\y_true.csv",
        },
    ]

    plot_label_distribution_comparison_from_files(
        labels,
        title="Lewotobi Laki-laki\nLabel Class Dsitribution",
        filepath="label_distribution_comparison_old",
    )


if __name__ == "__main__":
    main()
