from eruption_forecast.plots.label_plots import (
    LabelScenarioFileEntry,
    plot_label_distribution_comparison_from_files,
)


def semeru():
    labels: list[LabelScenarioFileEntry] = [
        {
            "name": "Training 2020",
            "csv": r"D:\Projects\eruption-forecast\output\VG.LEKR.00.EHZ\scenarios\semeru-scenario-1\training\labels\label_2020-09-30_2020-12-02_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Training 2020-2021",
            "csv": r"D:\Projects\eruption-forecast\output\VG.LEKR.00.EHZ\scenarios\semeru-scenario-3\training\labels\label_2020-09-30_2021-12-06_step-6-hours_dtf-2_ie-1.csv",
        },
    ]

    plot_label_distribution_comparison_from_files(
        labels,
        title="Semeru\nLabel Class Dsitribution",
        filepath="LEKR_scenarios",
        bar_width=0.38,
        group_gap=0.24,
        x_label_rotation=270,
    )


def old_scenarios():
    labels: list[LabelScenarioFileEntry] = [
        {
            "name": "Semeru 2020",
            "csv": r"D:\Projects\eruption-forecast\output\VG.LEKR.00.EHZ\scenarios\semeru-scenario-1\training\labels\label_2020-09-30_2020-12-02_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Semeru 2020-2021",
            "csv": r"D:\Projects\eruption-forecast\output\VG.LEKR.00.EHZ\scenarios\semeru-scenario-3\training\labels\label_2020-09-30_2021-12-06_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Lewotobi Scenario 1",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-1\training\labels\label_2025-01-01_2025-03-31_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Lewotobi Scenario 2",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-2\training\labels\label_2025-01-01_2025-04-30_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Lewotobi Scenario 3",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-3\training\labels\label_2025-01-01_2025-05-31_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Lewotobi Scenario 4",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-4\training\labels\label_2025-01-01_2025-06-30_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Lewotobi Scenario 5",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-5\training\labels\label_2025-01-01_2025-07-26_step-6-hours_dtf-2_ie-1.csv",
        },
    ]

    plot_label_distribution_comparison_from_files(
        labels,
        title="Label Class Dsitribution",
        filepath="OJN_old_scenarios",
        x_label_rotation=270,
    )


def new_scenarios():
    labels: list[LabelScenarioFileEntry] = [
        {
            "name": "Scenario 1.1",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-new\scenario-11\training\labels\label_2025-01-01_2025-03-31_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Scenario 2.1",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-new\scenario-21\training\labels\label_2025-01-01_2025-04-30_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Scenario 3.1",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-new\scenario-31\training\labels\label_2025-01-01_2025-05-31_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Scenario 4.1",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-new\scenario-41\training\labels\label_2025-01-01_2025-06-30_step-6-hours_dtf-2_ie-1.csv",
        },
        {
            "name": "Scenario 5.1",
            "csv": r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-new\scenario-51\training\labels\label_2025-01-01_2025-07-26_step-6-hours_dtf-2_ie-1.csv",
        },
    ]

    plot_label_distribution_comparison_from_files(
        labels,
        title="Lewotobi Laki-laki\nLabel Class Dsitribution",
        filepath="OJN_new_scenarios",
    )


if __name__ == "__main__":
    # new_scenarios()
    old_scenarios()
    # semeru()
