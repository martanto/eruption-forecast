from eruption_forecast.plots import replot_significant_features

def main():
    replot_significant_features(
        all_features_dir=r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\trainings\model-only\random-forest-classifier\stratified-k-fold\features\all_features",
        overwrite=True,
    )

if __name__ == "__main__":
    main()