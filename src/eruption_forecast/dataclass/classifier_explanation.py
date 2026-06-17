from dataclasses import field, dataclass

import shap


@dataclass(frozen=True, slots=True)
class SeedExplanation:
    """Per-seed SHAP explanation payload."""

    random_state: int
    shap_values: shap.Explanation


@dataclass(slots=True)
class ClassifierExplanation:
    """Aggregated SHAP explanations for one classifier across all seeds."""

    classifier_name: str
    seeds: list[SeedExplanation] = field(default_factory=list)

    def __repr__(self):
        return (
            f"ClassifierExplanation(classifier_name={self.classifier_name}, "
            f"seeds={len(self.seeds)})"
        )
