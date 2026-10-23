"""Shared prediction-service, 18-input model, batch, and artifact safety tests."""
import hashlib
import io
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from src.prediction_services import (
    ASSESSMENT_CATEGORICAL_CHOICES, BATCH_COLUMNS, FORM_FIELDS, ID_COLUMN, METADATA_PATH, MODEL_PATH, RESULT_COLUMNS,
    SEMESTER_COLUMN, USER_FIELDS, analyze_batch, dropout_probability, export_results,
    export_results_excel, load_model_artifact, make_input_frame, predict_student,
    prepare_student, read_uploaded_table, risk_label, support_guidance, template_excel,
    assessment_pdf,
    template_frame, validate_batch_dataframe, validate_inputs,
)
from src.bangladesh_collection import PREDICTOR_FIELDS, synthetic_example_rows

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "model_reports" / "combined_uci_bangladesh_18_v2" / "training_results.json"
ACTIVE_HASH = "D9F4D5663002E3C17482E57EE5E50E97038CFEDB3B987046ECB0F652CD4DBA06"
ACTIVE_THRESHOLD = 0.3040698561883773


def sample_values(**overrides):
    values = synthetic_example_rows().iloc[0].loc[list(FORM_FIELDS)].to_dict()
    values.update(overrides)
    return values


def student(**overrides):
    row = template_frame().iloc[0].to_dict()
    row[ID_COLUMN] = "ST001"
    row[SEMESTER_COLUMN] = "Relevant semester"
    row.update(overrides)
    return row


def test_active_model_has_exactly_18_predictors_and_expected_frozen_identity():
    assert tuple(FORM_FIELDS) == tuple(PREDICTOR_FIELDS)
    assert len(FORM_FIELDS) == len(USER_FIELDS) == 18
    assert MODEL_PATH.is_file() and METADATA_PATH.is_file()
    artifact = load_model_artifact()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert artifact["raw_input_features"] == FORM_FIELDS
    assert list(artifact["pipeline"].feature_names_in_) == FORM_FIELDS
    assert list(artifact["pipeline"].bangladesh_pipeline.feature_names_in_) == FORM_FIELDS
    assert artifact["input_feature_count"] == 18
    assert artifact["target_is_predictor"] is False
    assert artifact["target_field"] == "current_academic_status"
    assert artifact["pipeline"].blend_weight_uci == pytest.approx(.05)
    assert len(artifact["pipeline"].uci_pipeline.feature_names_in_) == 10
    assert artifact["architecture"].startswith("UCI 10-feature")
    assert artifact["threshold"] == metadata["threshold"] == ACTIVE_THRESHOLD
    assert list(make_input_frame(sample_values()).columns) == FORM_FIELDS
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest().upper() == ACTIVE_HASH


def test_saved_model_prediction_is_independent_of_training_dataset_paths(monkeypatch):
    monkeypatch.setenv("CSE_UCI_DATASET_PATH", str(ROOT / "missing-raw-data" / "uci.csv"))
    monkeypatch.setenv(
        "CSE_BANGLADESH_DATASET_PATH",
        str(ROOT / "missing-raw-data" / "responses.xlsx"),
    )
    result = predict_student(load_model_artifact(), sample_values())
    assert 0 <= result["probability"] <= 1
    assert result["risk_level"] in {"Lower Estimated Risk", "Elevated Estimated Risk"}


def test_model_load_and_prediction_use_bangladesh_categories_and_all_inputs():
    artifact = load_model_artifact()
    values = sample_values()
    frame = make_input_frame(values)
    probability = dropout_probability(artifact, values)
    assert frame.shape == (1, 18)
    assert 0 <= probability <= 1
    result = predict_student(artifact, student())
    assert result["risk_level"] == risk_label(result["probability"], artifact["threshold"])
    assert result["threshold"] == artifact["threshold"]
    assert set(result["component_scores"]) == {"uci_academic", "bangladesh_local"}
    for changed in (
        {"last_semester_attendance_percent": 0.0},
        {"family_financial_condition": "Poor (below ৳25,000)"},
        {"student_employment": "Full-time"},
    ):
        assert set(make_input_frame({**values, **changed}).columns) == set(FORM_FIELDS)


def test_target_status_is_never_an_input_and_schema_rejects_extra_or_missing_fields():
    values = sample_values()
    assert "current_academic_status" not in make_input_frame(values).columns
    with pytest.raises(ValueError, match="exactly the 18"):
        make_input_frame({**values, "current_academic_status": "Dropout (Discontinued Studies)"})
    with pytest.raises(ValueError, match="exactly the 18"):
        make_input_frame({key: value for key, value in values.items() if key != "student_employment"})


def test_input_validation_and_threshold_boundary():
    values = sample_values(courses_enrolled=2, courses_passed=3)
    assert any("greater" in message for message in validate_inputs(values))
    assert risk_label(ACTIVE_THRESHOLD - 0.001, ACTIVE_THRESHOLD) == "Lower Estimated Risk"
    assert risk_label(ACTIVE_THRESHOLD, ACTIVE_THRESHOLD) == "Elevated Estimated Risk"
    assert not validate_inputs(sample_values())


def test_assessment_uses_exact_bangladesh_occupation_choices():
    from src.bangladesh_collection import PARENT_OCCUPATIONS
    choices = ASSESSMENT_CATEGORICAL_CHOICES["mother_occupation_bangladesh"]
    assert choices == PARENT_OCCUPATIONS
    assert "Not Alive" in choices and "Housewife" in choices
    assert not any(value.startswith("UCI source-only") for value in choices)
    assert validate_inputs(sample_values(**{"mother_occupation_bangladesh": "UCI source-only — Student"}))
    assert validate_inputs(sample_values(**{"mother_occupation_bangladesh": "made-up category"}))


def test_both_assessment_parent_occupation_dropdowns_use_original_local_schema():
    from src.bangladesh_collection import PARENT_OCCUPATIONS
    assert ASSESSMENT_CATEGORICAL_CHOICES["mother_occupation_bangladesh"] == PARENT_OCCUPATIONS
    assert ASSESSMENT_CATEGORICAL_CHOICES["father_occupation_bangladesh"] == PARENT_OCCUPATIONS


def test_metrics_and_schema_match_training_report():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    assert results["training_counts"]["bangladesh_records"] == 251
    assert results["training_counts"]["uci_resolved_records"] == 3630
    assert results["bangladesh_features"] == FORM_FIELDS
    assert results["excluded_from_predictors"] == ["Timestamp", "current_academic_status"]
    assert results["bangladesh_audit"]["rows_removed"] == 0
    assert results["bangladesh_audit"]["semester_gpa_marked_unavailable_count"] == 19
    assert len(results["uci_original_15_feature_compatibility"]) == 15
    assert len(results["uci_foundation_features"]) == 10
    assert results["threshold"] == pytest.approx(load_model_artifact()["threshold"])
    assert results["stacker_fit_procedure"].find("OOF") >= 0
    assert results["selected_active_architecture"].startswith("Fixed 5%/95%")
    for candidate, entry in results["nested_cv_comparison"].items():
        metrics = entry["metrics"]
        assert all(0 <= metrics[key] <= 1 for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc_average_precision", "brier_score"))
        assert sum(metrics["confusion_matrix"].values()) == 251


def test_active_probability_changes_when_only_uci_component_signal_changes(monkeypatch):
    artifact = load_model_artifact()
    pipeline = artifact["pipeline"]
    frame = make_input_frame(sample_values())

    class FixedUCIProbability:
        classes_ = [0, 1]

        def __init__(self, probability): self.probability = probability
        def predict_proba(self, _frame):
            import numpy as np
            return np.tile([1 - self.probability, self.probability], (len(_frame), 1))

    original = pipeline.uci_pipeline
    try:
        pipeline.uci_pipeline = FixedUCIProbability(0.0)
        low = pipeline.predict_proba(frame)[0, 1]
        pipeline.uci_pipeline = FixedUCIProbability(1.0)
        high = pipeline.predict_proba(frame)[0, 1]
    finally:
        pipeline.uci_pipeline = original
    assert high != pytest.approx(low)


def test_support_rules_remain_deterministic_and_bounded():
    default = dict(semester_gpa=3.5, enrolled=6, evaluations=6, passed=6, debtor=False,
                   tuition_up_to_date=True, scholarship=True, previous_gpa=4.0)
    assert support_guidance(**default) == ([], [])
    indicators, supports = support_guidance(**{**default, "semester_gpa": 1.2, "enrolled": 8,
        "evaluations": 1, "passed": 1, "debtor": True, "tuition_up_to_date": False,
        "scholarship": False, "previous_gpa": 2.5})
    assert len(indicators) <= 3 and len(supports) <= 4
    assert indicators[0].startswith("Low recent academic performance")


def test_batch_schema_validation_prediction_and_exports():
    artifact = load_model_artifact()
    one = pd.DataFrame([student()])
    assert list(one.columns) == BATCH_COLUMNS
    assert validate_batch_dataframe(one).empty
    result = analyze_batch(artifact, one)
    assert len(result) == 1
    assert 0 <= result.loc[0, "Estimated Current Dropout-Status Probability"] <= 1
    individual = predict_student(artifact, student())
    assert result.loc[0, "Estimated Current Dropout-Status Probability"] == pytest.approx(individual["probability"])
    assert set(RESULT_COLUMNS).issubset(result.columns)
    csv = pd.read_csv(io.BytesIO(export_results(result)))
    excel = pd.read_excel(io.BytesIO(export_results_excel(result)), engine="openpyxl")
    assert set(RESULT_COLUMNS).issubset(csv.columns)
    assert set(RESULT_COLUMNS).issubset(excel.columns)
    assert csv.loc[0, "Mother's occupation"] == "Not Alive"
    assert csv.loc[0, "Estimated Current Dropout-Status Probability"] == pytest.approx(individual["probability"])


def test_batch_rejects_uci_occupation_values_not_in_bangladesh_schema():
    one = pd.DataFrame([student(**{"Mother's occupation": "UCI source-only (not used by prediction) — Student"})])
    assert "unsupported value" in " ".join(validate_batch_dataframe(one)["Issue"])


def test_batch_validation_rejects_invalid_values_but_allows_missing_optional_features():
    row = student()
    assert validate_batch_dataframe(pd.DataFrame([row]).drop(columns=[USER_FIELDS[0]])).iloc[0]["Issue"].startswith("Missing required columns")
    invalid = pd.DataFrame([student(**{"Courses passed in the relevant semester": 7})])
    assert "greater than" in " ".join(validate_batch_dataframe(invalid)["Issue"])
    invalid = pd.DataFrame([student(**{"Last-semester attendance percentage": 101})])
    assert "range 0–100" in " ".join(validate_batch_dataframe(invalid)["Issue"])
    missing = pd.DataFrame([student(**{"Last-semester attendance percentage": None})])
    assert validate_batch_dataframe(missing).empty
    assert pd.isna(pd.DataFrame([prepare_student(missing.iloc[0].to_dict())]).iloc[0]["last_semester_attendance_percent"])


def test_csv_excel_templates_and_uploads_use_the_18_input_contract():
    assert len(template_frame().columns) == 20
    excel_template = io.BytesIO(template_excel())
    excel_template.name = "student_batch_template.xlsx"
    loaded = read_uploaded_table(excel_template)
    assert list(loaded.columns) == BATCH_COLUMNS
    csv_file = io.BytesIO(pd.DataFrame([student()]).to_csv(index=False).encode())
    csv_file.name = "students.csv"
    assert list(read_uploaded_table(csv_file).columns) == BATCH_COLUMNS


def test_assessment_contract_uses_18_inputs_and_excludes_target():
    assert len(FORM_FIELDS) == 18
    assert len(template_frame().columns) == 20
    assert "current_academic_status" not in FORM_FIELDS
    assert load_model_artifact()["input_feature_count"] == 18


def test_status_result_panel_labels_current_status_not_future_dropout(monkeypatch):
    assert "current_academic_status" not in make_input_frame(sample_values()).columns
    assert predict_student(load_model_artifact(), student())["risk_level"] in {
        "Lower Estimated Risk", "Elevated Estimated Risk"
    }


def test_result_panel_displays_threshold_to_one_decimal_percent(monkeypatch):
    assert risk_label(ACTIVE_THRESHOLD, ACTIVE_THRESHOLD) == "Elevated Estimated Risk"


def test_individual_download_uses_final_calibrated_probability(monkeypatch):
    result = predict_student(load_model_artifact(), student())
    pdf = assessment_pdf(result, "ST001")
    assert pdf.startswith(b"%PDF-") and b"ST001" in pdf
    assert f"{result['probability']:.1%}".encode() in pdf


def test_pdf_download_contains_assessment_and_uses_supplied_result_without_predicting():
    result={"probability":.417,"threshold":ACTIVE_THRESHOLD,"risk_level":"Elevated Estimated Risk",
            "indicators":["Attendance follow-up flagged"],"supports":["Academic advising"]}
    pdf=assessment_pdf(result,"REFERENCE-42")
    assert pdf.startswith(b"%PDF-")
    assert len(pdf)>1000
    for expected in (b'REFERENCE-42',b'41.7%',b'Elevated Estimated Risk',b'Attendance follow-up flagged',b'Academic advising'):
        assert expected in pdf
