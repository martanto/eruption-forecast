import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.ensemble import SeedEnsemble


def main(
    model_filepath: str, features_matrix_filepath: str, top_features_filepath: str
) -> None:
    seed_ensemble: SeedEnsemble = joblib.load(model_filepath)

    # get first seed model for test
    model = seed_ensemble[0]["model"]

    top_features_df = pd.read_csv(top_features_filepath, index_col=0)
    feature_names = top_features_df.index.tolist()

    features_df = pd.read_csv(features_matrix_filepath, index_col=0)[feature_names]

    print(features_df.shape)

    explainer = shap.TreeExplainer(model, features_df)
    shap_values = explainer(features_df)
    #
    # fig, ax = shap.partial_dependence_plot(
    #     "entropy__approximate_entropy__m_2__r_0.3",
    #     seed_ensemble.predict_proba,
    #     features_df,
    #     model_expected_value=True,
    #     feature_expected_value=True,
    #     show=False,
    #     ice=False,
    #     shap_values=shap_values[20:21, :],
    # )

    # fig, ax = shap.plots.waterfall(shap_values[1], max_display=20)

    print(shap_values.shape)
    shap.plots.beeswarm(shap_values)

    plt.show()


if __name__ == "__main__":
    plt.switch_backend("TkAgg")

    # _model_filepath = r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios\scenario-1\training\classifiers\GradientBoostingClassifier\stratified-shuffle-split\SeedEnsemble_GradientBoostingClassifier_StratifiedShuffleSplit_seeds-10_features-20.pkl"
    # _features_matrix_filepath = r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios\scenario-1\prediction\features\features-matrix_2025-03-30-2025-04-30.csv"
    # _top_features_filepath = r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios\scenario-1\training\features\stratified-shuffle-split\top_20_features.csv"

    _model_filepath = r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios\scenario-8\training\classifiers\random-forest-classifier\stratified-shuffle-split\SeedEnsemble_RandomForestClassifier_StratifiedShuffleSplit_seeds-100_features-20.pkl"
    _features_matrix_filepath = r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios\scenario-8\prediction\features\features-matrix_2025-07-25-2025-08-22.csv"
    _top_features_filepath = r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios\scenario-8\training\features\stratified-shuffle-split\top_20_features.csv"

    main(_model_filepath, _features_matrix_filepath, _top_features_filepath)
