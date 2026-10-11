"""Tests for ``eruption_forecast.utils.feature_utils`` and feature-name formatting.

Covers loading curated feature lists, alias maps, legacy-column migration,
ranked-CSV aggregation across seeds and runs, Parquet concatenation, and the
training/prediction matrix merge, plus ``shorten_feature_name`` /
``humanize_feature_name`` / ``get_classifier_label``.
"""

import os

import numpy as np
import pandas as pd
import pytest

from eruption_forecast.utils.formatting import (
    get_classifier_label,
    shorten_feature_name,
    humanize_feature_name,
)
from eruption_forecast.utils.feature_utils import (
    concat_features,
    find_common_features,
    load_feature_aliases,
    load_select_features,
    migrate_score_column,
    merge_features_matrix,
    update_top_features_csv,
    concat_significant_features,
)


# ---------------------------------------------------------------------------
# load_select_features
# ---------------------------------------------------------------------------


class TestLoadSelectFeatures:
    def test_from_list(self) -> None:
        assert load_select_features(["a", "b"]) == ["a", "b"]

    def test_strips_and_drops_blank_entries(self) -> None:
        # The docstring says blank entries raise; the code drops them.
        assert load_select_features([" a ", "", "  ", "b"]) == ["a", "b"]

    def test_truncates(self) -> None:
        assert load_select_features(["a", "b", "c"], number_of_features=2) == ["a", "b"]

    @pytest.mark.parametrize("n", [0, -1])
    def test_non_positive_disables_truncation(self, n: int) -> None:
        assert load_select_features(["a", "b", "c"], number_of_features=n) == [
            "a",
            "b",
            "c",
        ]

    def test_from_csv_index(self, tmp_path) -> None:
        path = tmp_path / "top_20_features.csv"
        pd.DataFrame({"frequency": [3, 1]}, index=pd.Index(["x", "y"], name="features")).to_csv(path)
        assert load_select_features(str(path)) == ["x", "y"]

    def test_missing_csv_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_select_features(str(tmp_path / "nope.csv"))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            load_select_features(["", " "])


# ---------------------------------------------------------------------------
# Aliases / migration
# ---------------------------------------------------------------------------


@pytest.fixture()
def ranked_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"frequency": [5, 3], "alias": ["ft_1", "ft_2"]},
        index=pd.Index(["rsam_f2__mean", "entropy__variance"], name="features"),
    )


class TestLoadFeatureAliases:
    def test_reverse_maps_alias_to_canonical(self, ranked_df) -> None:
        assert load_feature_aliases(ranked_df) == {
            "ft_1": "rsam_f2__mean",
            "ft_2": "entropy__variance",
        }

    def test_forward_maps_canonical_to_alias(self, ranked_df) -> None:
        assert load_feature_aliases(ranked_df, reverse=False)["rsam_f2__mean"] == "ft_1"

    def test_from_csv(self, ranked_df, tmp_path) -> None:
        path = tmp_path / "top.csv"
        ranked_df.to_csv(path)
        assert load_feature_aliases(str(path))["ft_2"] == "entropy__variance"

    def test_missing_alias_column_raises(self) -> None:
        with pytest.raises(ValueError):
            load_feature_aliases(pd.DataFrame({"frequency": [1]}, index=["a"]))

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_feature_aliases(str(tmp_path / "missing.csv"))


class TestMigrateScoreColumn:
    def test_renames_legacy_columns(self) -> None:
        df = migrate_score_column(pd.DataFrame({"score": [1], "mean_score": [0.1]}))
        assert df.columns.tolist() == ["frequency", "score_mean", "score_std"]
        assert np.isnan(df["score_std"].iloc[0])

    def test_current_schema_unchanged(self) -> None:
        df = pd.DataFrame({"frequency": [1], "score_mean": [0.1], "score_std": [0.0]})
        pd.testing.assert_frame_equal(migrate_score_column(df), df)


class TestUpdateTopFeaturesCsv:
    def test_backfills_alias_and_description(self, tmp_path) -> None:
        path = tmp_path / "top.csv"
        pd.DataFrame(
            {"score": [4, 2]}, index=pd.Index(["rsam_f2__mean", "entropy__variance"], name="features")
        ).to_csv(path)

        df = update_top_features_csv(str(path))

        assert df["alias"].tolist() == ["ft_1", "ft_2"]
        assert df["description"].tolist() == [
            "Mean of RSAM 2–5 Hz",
            "Variance of Shannon entropy",
        ]
        assert "frequency" in pd.read_csv(path, index_col=0).columns

    def test_current_file_is_left_alone(self, tmp_path) -> None:
        path = tmp_path / "top.csv"
        df = pd.DataFrame(
            {
                "frequency": [1],
                "score_mean": [0.1],
                "score_std": [0.0],
                "alias": ["custom"],
                "description": ["keep me"],
            },
            index=pd.Index(["rsam_f2__mean"], name="features"),
        )
        df.to_csv(path)
        mtime = os.path.getmtime(path)

        result = update_top_features_csv(str(path))

        assert result["alias"].tolist() == ["custom"]
        assert os.path.getmtime(path) == mtime

    def test_overwrite_regenerates(self, tmp_path) -> None:
        path = tmp_path / "top.csv"
        pd.DataFrame(
            {
                "frequency": [1],
                "score_mean": [0.1],
                "score_std": [0.0],
                "alias": ["custom"],
                "description": ["stale"],
            },
            index=pd.Index(["entropy__variance"], name="features"),
        ).to_csv(path)

        result = update_top_features_csv(str(path), overwrite=True)

        assert result["alias"].tolist() == ["ft_1"]
        assert result["description"].tolist() == ["Variance of Shannon entropy"]

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            update_top_features_csv(str(tmp_path / "x.csv"))


# ---------------------------------------------------------------------------
# Ranked aggregation
# ---------------------------------------------------------------------------


def _write_seed_csv(path, rows: list[tuple[str, float]]) -> str:
    pd.DataFrame(rows, columns=["features", "score"]).to_csv(path, index=False)
    return str(path)


class TestConcatSignificantFeatures:
    def test_ranks_by_frequency_then_score(self, tmp_path) -> None:
        seed_csvs = [
            _write_seed_csv(tmp_path / "s0.csv", [("a", 0.01), ("b", 0.02)]),
            _write_seed_csv(tmp_path / "s1.csv", [("a", 0.03), ("c", 0.001)]),
            _write_seed_csv(tmp_path / "s2.csv", [("b", 0.04), ("c", 0.002)]),
        ]

        ranked = concat_significant_features(
            seed_csvs, str(tmp_path), number_of_features=2
        )

        # All three appear twice; ties break on the smaller mean p-value.
        assert ranked.index.tolist() == ["c", "a", "b"]
        assert ranked["frequency"].tolist() == [2, 2, 2]
        assert ranked["alias"].tolist() == ["ft_1", "ft_2", "ft_3"]
        assert (tmp_path / "significant_features.csv").is_file()
        assert (tmp_path / "top_features.csv").is_file()
        top_2 = pd.read_csv(tmp_path / "top_2_features.csv", index_col=0)
        assert top_2.index.tolist() == ["c", "a"]

    def test_without_number_of_features_returns_raw(self, tmp_path) -> None:
        seed_csvs = [_write_seed_csv(tmp_path / "s0.csv", [("a", 0.01)])]
        raw = concat_significant_features(seed_csvs, str(tmp_path))
        assert raw.columns.tolist() == ["features", "score"]
        assert not (tmp_path / "top_features.csv").exists()

    def test_empty_inputs_raise(self, tmp_path) -> None:
        empty = tmp_path / "empty.csv"
        pd.DataFrame(columns=["features", "score"]).to_csv(empty, index=False)
        with pytest.raises(ValueError):
            concat_significant_features([str(empty)], str(tmp_path))


class TestFindCommonFeatures:
    def _ranked_csv(self, path, data: dict[str, int]) -> str:
        pd.DataFrame(
            {
                "frequency": list(data.values()),
                "score_mean": [0.01] * len(data),
                "score_std": [0.0] * len(data),
            },
            index=pd.Index(list(data), name="features"),
        ).to_csv(path)
        return str(path)

    def test_intersection_and_summed_frequency(self, tmp_path) -> None:
        first = self._ranked_csv(tmp_path / "a.csv", {"x": 5, "y": 1, "only_a": 9})
        second = self._ranked_csv(tmp_path / "b.csv", {"x": 2, "y": 4})

        common = find_common_features([first, second], output_dir=str(tmp_path))

        assert common.index.tolist() == ["x", "y"]
        assert common["frequency"].tolist() == [7, 5]
        assert (tmp_path / "common_top_features.csv").is_file()

    def test_no_overlap_raises(self, tmp_path) -> None:
        first = self._ranked_csv(tmp_path / "a.csv", {"x": 1})
        second = self._ranked_csv(tmp_path / "b.csv", {"y": 1})
        with pytest.raises(ValueError):
            find_common_features([first, second], output_dir=str(tmp_path))

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError):
            find_common_features([])


# ---------------------------------------------------------------------------
# Matrix concat / merge
# ---------------------------------------------------------------------------


class TestConcatFeatures:
    def test_concatenates_columns(self, tmp_path) -> None:
        index = pd.RangeIndex(3, name="id")
        left = tmp_path / "left.parquet"
        pd.DataFrame({"a": [1, 2, 3]}, index=index).to_parquet(left)
        right = pd.DataFrame({"b": [4, 5, 6]}, index=index)
        out = tmp_path / "out.parquet"

        path, df = concat_features([str(left)], str(out), frames=[right])

        assert path == str(out)
        assert df.columns.tolist() == ["a", "b"]
        pd.testing.assert_frame_equal(pd.read_parquet(out), df)

    def test_single_input_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            concat_features([], str(tmp_path / "out.parquet"), frames=[pd.DataFrame()])


class TestMergeFeaturesMatrix:
    @pytest.fixture()
    def matrices(self, tmp_path) -> dict[str, str]:
        train_index = pd.date_range("2025-01-01", periods=4, freq="6h", name="datetime")
        # Prediction starts before training ends, so two rows overlap.
        pred_index = pd.date_range("2025-01-01 12:00", periods=6, freq="3h", name="datetime")

        def write(prefix, index, offset):
            labels = pd.DataFrame({"id": range(len(index)), "is_erupted": 0}, index=index)
            features = pd.DataFrame(
                {"a": np.arange(len(index)) + offset, "b": 0.0, "c": 1.0},
                index=pd.RangeIndex(len(index), name="id"),
            )
            label_csv = tmp_path / f"{prefix}-label.csv"
            matrix = tmp_path / f"{prefix}-matrix.parquet"
            labels.to_csv(label_csv)
            features.to_parquet(matrix)
            return str(label_csv), str(matrix)

        train_label, train_matrix = write("train", train_index, 0)
        pred_label, pred_matrix = write("pred", pred_index, 100)
        return {
            "training_features_matrix": train_matrix,
            "training_label_csv": train_label,
            "prediction_features_matrix": pred_matrix,
            "prediction_label_csv": pred_label,
        }

    def test_merges_and_truncates_overlap(self, matrices, tmp_path) -> None:
        out = tmp_path / "merged.parquet"
        merged = merge_features_matrix(
            **matrices, select_features=["a", "missing"], output_path=str(out)
        )

        assert merged.columns.tolist() == ["a"]
        assert merged.index.is_monotonic_increasing
        assert not merged.index.has_duplicates
        # Training covers up to 18:00; prediction rows at or before that drop.
        assert merged.loc[merged.index <= "2025-01-01 18:00", "a"].tolist() == [0, 1, 2, 3]
        assert (merged.loc[merged.index > "2025-01-01 18:00", "a"] >= 100).all()
        assert out.is_file()

    def test_no_selected_features_present_raises(self, matrices) -> None:
        with pytest.raises(ValueError):
            merge_features_matrix(**matrices, select_features=["zzz"])

    def test_unsupported_suffix_raises(self, matrices) -> None:
        matrices["training_features_matrix"] = "matrix.txt"
        with pytest.raises(ValueError):
            merge_features_matrix(**matrices, select_features=["a"])


# ---------------------------------------------------------------------------
# Feature-name formatting
# ---------------------------------------------------------------------------


class TestShortenFeatureName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ('dsar_f3-f4__fft_coefficient__attr_"abs"__coeff_91', "dsar_f3-f4 | fft_coef(abs, 91)"),
            ("entropy__variance", "entropy | variance"),
            ("rsam_f2__standard_deviation", "rsam_f2 | std"),
            ("not_a_tsfresh_name", "not_a_tsfresh_name"),
            ("rsam_f0__unknown_calc", "rsam_f0 | unknown_calc"),
        ],
    )
    def test_cases(self, name: str, expected: str) -> None:
        assert shorten_feature_name(name) == expected


class TestHumanizeFeatureName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (
                'dsar_f3-f4__fft_coefficient__attr_"abs"__coeff_91',
                "Fourier coefficient (attr=abs, coeff=91) of DSAR ratio 4.5/8 Hz",
            ),
            ("rsam_f2__mean", "Mean of RSAM 2–5 Hz"),
            ("entropy__autocorrelation__lag_1", "Autocorrelation (lag=1) of Shannon entropy"),
            ("plain", "plain"),
            ("rsam_f9__mean", "Mean of rsam f9"),
            ("rsam_f0__brand_new_calc", "Brand new calc of RSAM 0.01–0.1 Hz"),
        ],
    )
    def test_default_bands(self, name: str, expected: str) -> None:
        assert humanize_feature_name(name) == expected

    def test_custom_bands(self) -> None:
        custom = {"f0": (0.05, 0.5), "f1": (0.5, 3.0), "f2": (3.0, 10.0)}
        assert humanize_feature_name("dsar_f0-f1__mean", freq_bands=custom) == (
            "Mean of DSAR ratio 0.05/0.5 Hz"
        )
        assert humanize_feature_name("rsam_f2__mean", freq_bands=custom) == (
            "Mean of RSAM 3–10 Hz"
        )


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("RandomForestClassifier", "Random Forest"),
        ("XGBClassifier", "XGBoost"),
        ("SVC", "svm"),
        ("MyCustomClassifier", "MyCustomClassifier"),
    ],
)
def test_get_classifier_label(name: str, label: str) -> None:
    assert get_classifier_label(name) == label
