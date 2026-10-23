"""Tests for the expanded UCI foundation and OOF-fitted ensemble candidate."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.bangladesh_collection import PREDICTOR_FIELDS, load_bangladesh_training_data
from src.combined_ensemble_v2 import (
    COMBINED_PARENT_OCCUPATION_CHOICES,
    UCI_FOUNDATION_FEATURES,
    UCI_OCCUPATION_BY_LABEL,
    UCI_MARITAL_TO_BANGLADESH,
    bangladesh_component_frame,
    calibrator_logit_feature,
    marital_status_to_bangladesh,
    parent_occupation_to_bangladesh,
    parent_occupation_to_uci,
    uci_occupation_code_to_broad,
    uci_harmonized_frame,
)
from src.train_combined_ensemble_18 import calibrator_predict, load_uci
from src.training_data_paths import bangladesh_dataset_path, uci_dataset_path
from src.train_combined_ensemble_18 import FEATURE_COMPATIBILITY
from src.bangladesh_collection import synthetic_example_rows

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "combined_uci_bangladesh_18_v2"
RESULTS_PATH = ROOT / "model_reports" / "combined_uci_bangladesh_18_v2" / "training_results.json"


class FixedProbabilityModel:
    classes_ = np.asarray([0, 1])

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, frame):
        return np.tile([1.0 - self.probability, self.probability], (len(frame), 1))


def test_uci_foundation_uses_only_documented_harmonizable_features():
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    artifact = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")
    assert result["training_counts"]["uci_resolved_records"] == 3630
    assert result["uci_audit"]["excluded_enrolled_rows"] == 794
    assert len(FEATURE_COMPATIBILITY) == 15
    assert sum(item["decision"] == "Use" for item in FEATURE_COMPATIBILITY) == 10
    assert all(item["decision"] == "Exclude" for item in FEATURE_COMPATIBILITY if item["UCI predictor"] in {"Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", "Marital status"})
    assert list(artifact["pipeline"].uci_pipeline.feature_names_in_) == list(UCI_FOUNDATION_FEATURES)


def test_both_local_training_sources_are_readable_without_retraining():
    uci_source = uci_dataset_path()
    if not uci_source.is_file():
        pytest.skip(
            "UCI training CSV is stored outside the source tree; set "
            "CSE_UCI_DATASET_PATH to run this optional source-data check."
        )
    uci_X, uci_y, uci_audit = load_uci()
    bangladesh_source = bangladesh_dataset_path()
    if not bangladesh_source.is_file():
        pytest.skip(
            "Private Bangladesh training workbook is not present; normal Django "
            "inference uses the saved artifact and does not require this source."
        )
    bd_X, bd_y, _, bd_audit = load_bangladesh_training_data(bangladesh_source)
    assert len(uci_X) == len(uci_y) == uci_audit["resolved_rows_used"] == 3630
    assert len(bd_X) == len(bd_y) == bd_audit["records"] == 251
    assert list(bd_X.columns) == list(PREDICTOR_FIELDS)


def test_harmonizer_preserves_unmatched_categories_instead_of_coercing_them():
    frame = synthetic_example_rows().loc[:, list(PREDICTOR_FIELDS)].copy()
    frame.loc[0, "gender_bangladesh"] = "Another identity"
    output = uci_harmonized_frame(frame)
    assert pd.isna(output.loc[0, "gender_bangladesh"])
    assert output.shape == (len(frame), 10)


def test_active_weighted_artifact_accepts_exactly_18_inputs_and_both_signals():
    weighted = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")
    values = synthetic_example_rows().iloc[[0]].loc[:, list(PREDICTOR_FIELDS)]
    pipeline = weighted["pipeline"]
    assert weighted["input_feature_count"] == 18
    assert list(pipeline.feature_names_in_) == list(PREDICTOR_FIELDS)
    local, uci = pipeline.component_probabilities(values)
    assert local.shape == uci.shape == (1,)
    assert np.isfinite(local[0]) and np.isfinite(uci[0])
    probability = pipeline.predict_proba(values)[0, 1]
    assert 0 <= probability <= 1
    assert weighted["pipeline"].blend_weight_uci == .05


def test_final_probability_depends_on_uci_signal_for_each_ensemble():
    weighted = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")["pipeline"]
    frame = synthetic_example_rows().iloc[[0]].loc[:, list(PREDICTOR_FIELDS)]
    real_uci = weighted.uci_pipeline
    try:
        weighted.uci_pipeline = FixedProbabilityModel(.01)
        lower = weighted.predict_proba(frame)[0, 1]
        weighted.uci_pipeline = FixedProbabilityModel(.99)
        upper = weighted.predict_proba(frame)[0, 1]
    finally:
        weighted.uci_pipeline = real_uci
    assert upper != lower


def test_saved_calibrated_output_matches_training_time_logit_transform():
    weighted = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")["pipeline"]
    frame = synthetic_example_rows().iloc[[0]].loc[:, list(PREDICTOR_FIELDS)]
    local, uci = weighted.component_probabilities(frame)
    assert local[0] != uci[0]
    blended = .95 * local + .05 * uci
    expected = calibrator_predict(weighted.calibrator, blended)[0]
    assert weighted.predict_proba(frame)[0, 1] == pytest.approx(expected, abs=1e-12)
    transformed = calibrator_logit_feature(blended)
    direct = weighted.calibrator.predict_proba(transformed.reshape(-1, 1))[0, 1]
    assert expected == pytest.approx(direct, abs=1e-12)
    assert 0 <= expected <= 1


def test_different_component_scores_are_blended_then_calibrated_not_forced_to_raw_blend():
    weighted = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")["pipeline"]
    frame = synthetic_example_rows().iloc[[0]].loc[:, list(PREDICTOR_FIELDS)]
    original_local, original_uci = weighted.bangladesh_pipeline, weighted.uci_pipeline
    try:
        weighted.bangladesh_pipeline = FixedProbabilityModel(.076)
        weighted.uci_pipeline = FixedProbabilityModel(.179)
        raw_blend = .95 * .076 + .05 * .179
        expected = calibrator_predict(weighted.calibrator, np.asarray([raw_blend]))[0]
        actual = weighted.predict_proba(frame)[0, 1]
    finally:
        weighted.bangladesh_pipeline, weighted.uci_pipeline = original_local, original_uci
    assert raw_blend == pytest.approx(.08115)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert 0 <= actual <= 1


def test_source_qualified_occupation_options_have_explicit_component_handling():
    bd_values = {"Not Alive", "Housewife", "Business owner", "Government employee", "Private-sector employee"}
    assert bd_values.issubset(COMBINED_PARENT_OCCUPATION_CHOICES)
    uci_student = "UCI source-only (not used by prediction) — Student"
    assert uci_student in COMBINED_PARENT_OCCUPATION_CHOICES
    assert UCI_OCCUPATION_BY_LABEL[uci_student] == 0
    assert parent_occupation_to_uci(uci_student) == 0
    assert parent_occupation_to_uci("UCI source-only (not used by prediction) — Intellectual and scientific specialists") == 2
    assert uci_occupation_code_to_broad(121) == 2
    assert uci_occupation_code_to_broad(102) == 10
    assert uci_occupation_code_to_broad(99) is None
    assert pd.isna(parent_occupation_to_bangladesh(uci_student))
    assert parent_occupation_to_bangladesh("Not Alive") == "Not Alive"
    assert pd.isna(parent_occupation_to_bangladesh("unrecognized occupation"))
    assert UCI_MARITAL_TO_BANGLADESH == {
        1: "Single / Never married", 2: "Married", 3: "Widowed",
        4: "Divorced", 6: "Separated",
    }
    assert marital_status_to_bangladesh(3) == "Widowed"
    assert marital_status_to_bangladesh(6) == "Separated"
    assert pd.isna(marital_status_to_bangladesh(5))
    frame = synthetic_example_rows().iloc[[0]].loc[:, list(PREDICTOR_FIELDS)].copy()
    frame.loc[:, "mother_occupation_bangladesh"] = uci_student
    masked = bangladesh_component_frame(frame)
    assert pd.isna(masked.loc[masked.index[0], "mother_occupation_bangladesh"])
    assert "mother_occupation_bangladesh" not in uci_harmonized_frame(frame).columns


def test_source_qualified_occupation_runs_through_active_ensemble_without_coercion():
    weighted = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")["pipeline"]
    frame = synthetic_example_rows().iloc[[0]].loc[:, list(PREDICTOR_FIELDS)].copy()
    frame.loc[:, "mother_occupation_bangladesh"] = "UCI source-only (not used by prediction) — Student"
    local, uci = weighted.component_probabilities(frame)
    assert np.isfinite(local[0]) and np.isfinite(uci[0])
    assert 0 <= weighted.predict_proba(frame)[0, 1] <= 1


def test_housewife_is_preserved_but_only_mother_field_is_in_fitted_encoder():
    from src.bangladesh_collection import CATEGORICAL_CHOICES
    local = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")["pipeline"].bangladesh_pipeline
    encoder = local.named_steps["preprocess"].named_transformers_["categorical"].named_steps["onehot"]
    categorical_fields = [field for field in PREDICTOR_FIELDS if field in CATEGORICAL_CHOICES]
    encoded = dict(zip(categorical_fields, encoder.categories_))
    assert "Housewife" in encoded["mother_occupation_bangladesh"]
    assert "Housewife" not in encoded["father_occupation_bangladesh"]


def test_metadata_documents_cohorts_oof_fusion_threshold_and_distinct_targets():
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    assert result["training_counts"]["uci_resolved_records"] == 3630
    assert result["training_counts"]["bangladesh_records"] == 251
    assert "OOF" in result["stacker_fit_procedure"]
    assert "reported current status" in result["warning"].lower()
    assert metadata["active_model"] == "weighted_ensemble_dropout_status_18.joblib"
    assert result["threshold"] == metadata["threshold"]
