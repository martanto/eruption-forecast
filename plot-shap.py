import matplotlib.pyplot as plt

from eruption_forecast.utils.ml import build_classifier_ensemble_summary
from eruption_forecast.model.explanation_model import ExplanationModel
from eruption_forecast.plots.explanation_plots import plot_classifier_waterfall


eruption_dates = [
    "2025-03-20",
    "2025-04-10",
    "2025-04-22",
    "2025-05-18",
    "2025-06-17",
    "2025-07-07",
    "2025-08-02",
    "2025-08-18",
]


def main():
    em = ExplanationModel.from_file(
        filepath=r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios\scenario-8\PredictionModel_2025-07-27_2025-08-22_ws-2.pkl",
        output_dir=r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios\scenario-8",
        n_jobs=8,
        verbose=True,
    ).explain(save_per_seed=True)

    labels = em.model.labels

    for classifier_explanation in em.explanations:
        classifier_name = classifier_explanation["classifier_name"]
        seed_ensemble = em.ClassifierEnsemble.ensembles[classifier_name]

        summary = build_classifier_ensemble_summary(
            seed_ensemble=seed_ensemble,
            labels=labels,
            eruption_dates=eruption_dates,
        )
        plot_classifier_waterfall(summary, classifier_explanation)
        plt.show()


if __name__ == "__main__":
    plt.switch_backend("TkAgg")
    main()
