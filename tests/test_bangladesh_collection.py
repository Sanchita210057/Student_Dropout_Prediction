"""Tests for the isolated Bangladesh 18-predictor plus status-target schema."""
import io
import json
import re
from pathlib import Path

import pandas as pd
from src.prediction_services import FORM_FIELDS, MODEL_PATH, bangladesh_research_excel
from src.bangladesh_collection import (
    BANGLADESH_EXTENDED_FEATURES,
    EXPORT_COLUMNS,
    FIELD_TITLES,
    FINANCIAL_CONDITION_OPTIONS,
    PARENT_OCCUPATIONS,
    PREDICTOR_FIELDS,
    SCHEMA_VERSION,
    STATUS_FIELD,
    academic_status_target,
    SHARED_ACADEMIC_FEATURES,
    SHARED_CORE_FEATURES,
    STUDENT_EMPLOYMENT_OPTIONS,
    clean_google_forms_responses,
    future_comparison_feature_sets,
    synthetic_example_rows,
    validate_bangladesh_responses,
)

ROOT = Path(__file__).resolve().parents[1]


def codes(frame: pd.DataFrame) -> set[str]:
    issues = validate_bangladesh_responses(frame)
    return set(issues.loc[issues["Severity"].eq("error"), "Field"])


def test_schema_has_18_predictors_plus_final_status_target_and_keeps_uci_separate():
    assert SCHEMA_VERSION == "3.0.0"
    assert EXPORT_COLUMNS == (*PREDICTOR_FIELDS, STATUS_FIELD)
    assert len(PREDICTOR_FIELDS) == 18
    assert len(EXPORT_COLUMNS) == 19
    assert list(FIELD_TITLES) == list(EXPORT_COLUMNS)
    assert tuple(FORM_FIELDS) == tuple(PREDICTOR_FIELDS)
    assert PARENT_OCCUPATIONS == ("Not Alive", "Retired", "Housewife", "Business owner", "Government employee", "Private-sector employee", "Self-employed / Daily-wage worker", "Other", "Unknown / Prefer not to say")
    assert MODEL_PATH.is_file()


def test_parent_occupation_categories_are_exact():
    assert PARENT_OCCUPATIONS == (
        "Not Alive", "Retired", "Housewife", "Business owner",
        "Government employee", "Private-sector employee",
        "Self-employed / Daily-wage worker", "Other", "Unknown / Prefer not to say",
    )


def test_synthetic_training_frame_uses_18_features_and_applies_zero_gpa_rule():
    from src.bangladesh_collection import prepare_bangladesh_training_data
    raw = synthetic_example_rows()
    raw.loc[0, "semester_gpa_native"] = 0
    raw.loc[0, "courses_passed"] = 3
    raw.loc[1, "semester_gpa_native"] = 0
    raw.loc[1, "courses_passed"] = raw.loc[1, "courses_enrolled"]
    X, y, status, audit = prepare_bangladesh_training_data(raw)
    assert X.shape == (3, 18)
    assert STATUS_FIELD not in X.columns and "Timestamp" not in X.columns
    assert y.tolist() == [0, 0, 1]
    assert status.tolist() == [
        "Running (Currently Studying)",
        "Passed (Graduated)",
        "Dropout (Discontinued Studies)",
    ]
    assert audit["rows_removed"] == 0 and audit["course_counts_changed"] is False
    assert audit["semester_gpa_marked_unavailable_count"] == 1
    assert pd.isna(X.loc[0, "semester_gpa_native"])
    assert X.loc[1, "semester_gpa_native"] == 0


def test_training_source_configuration_is_optional_for_inference_and_clear_when_missing(monkeypatch):
    from src.training_data_paths import require_training_source, uci_dataset_path, UCI_DATA_ENV
    missing = ROOT / "private-archive-not-present" / "private-uci.csv"
    monkeypatch.setenv(UCI_DATA_ENV, str(missing))
    assert uci_dataset_path() == missing
    try:
        require_training_source(missing, UCI_DATA_ENV, "UCI")
    except FileNotFoundError as error:
        assert UCI_DATA_ENV in str(error)
        assert "Normal application predictions do not require raw training data" in str(error)
    else:
        raise AssertionError("Missing training source should fail with setup guidance")


def test_attendance_boundaries_and_missing_answers():
    valid = synthetic_example_rows()
    valid.loc[0, "last_semester_attendance_percent"] = 0
    valid.loc[1, "last_semester_attendance_percent"] = 100
    assert "last_semester_attendance_percent" not in codes(valid)
    for value in (-0.01, 100.01):
        invalid = valid.copy()
        invalid.loc[0, "last_semester_attendance_percent"] = value
        assert "last_semester_attendance_percent" in codes(invalid)
    valid.loc[0, "last_semester_attendance_percent"] = pd.NA
    assert "last_semester_attendance_percent" not in codes(valid)


def test_income_and_employment_options_are_validated():
    frame = synthetic_example_rows()
    frame["family_financial_condition"] = list(FINANCIAL_CONDITION_OPTIONS[:3])
    frame["student_employment"] = list(STUDENT_EMPLOYMENT_OPTIONS[:3])
    assert not {"family_financial_condition", "student_employment"}.intersection(codes(frame))
    frame.loc[0, "family_financial_condition"] = "Official poverty category"
    frame.loc[1, "student_employment"] = "Unknown occupation code"
    assert {"family_financial_condition", "student_employment"}.issubset(codes(frame))


def test_google_forms_csv_headers_normalize_to_18_predictors_plus_required_target():
    source = synthetic_example_rows()
    google = source.rename(columns=FIELD_TITLES)
    google.insert(0, "Timestamp", ["2026-01-10 09:00:00"] * len(google))
    google["Unrelated metadata"] = "ignored"
    cleaned = clean_google_forms_responses(google)
    assert tuple(cleaned.columns) == EXPORT_COLUMNS
    assert validate_bangladesh_responses(google).empty
    assert len(cleaned.columns) == 19

    dictionary = json.loads((ROOT / "data" / "bangladesh_questionnaire" / "v3" / "data_dictionary_v3.json").read_text(encoding="utf-8"))
    assert dictionary["schema_version"] == SCHEMA_VERSION
    assert dictionary["export_columns"] == list(EXPORT_COLUMNS)
    assert [item["name"] for item in dictionary["predictors"]] == list(PREDICTOR_FIELDS)
    assert dictionary["target"]["field"] == STATUS_FIELD
    assert dictionary["target"]["predictor"] is False

    excel = pd.read_excel(io.BytesIO(bangladesh_research_excel(cleaned)), engine="openpyxl")
    assert list(excel.columns) == list(EXPORT_COLUMNS)


def test_missing_columns_are_warned_and_filled_blank_but_not_required():
    full = synthetic_example_rows()
    partial = full.drop(columns=["last_semester_attendance_percent", "student_employment"])
    cleaned = clean_google_forms_responses(partial)
    assert cleaned["last_semester_attendance_percent"].isna().all()
    assert cleaned["student_employment"].isna().all()
    issues = validate_bangladesh_responses(partial)
    assert (issues["Severity"] == "warning").all()

    malformed = full.copy()
    malformed["last_semester_attendance_percent"] = malformed["last_semester_attendance_percent"].astype(object)
    malformed.loc[0, "last_semester_attendance_percent"] = "seventy"
    assert "last_semester_attendance_percent" in codes(malformed)


def test_status_target_is_required_and_uses_requested_binary_mapping():
    frame = synthetic_example_rows()
    assert academic_status_target(frame).tolist() == [0, 0, 1]
    assert STATUS_FIELD not in future_comparison_feature_sets(frame)["bangladesh_expanded"].columns

    missing_header = frame.drop(columns=[STATUS_FIELD])
    issues = validate_bangladesh_responses(missing_header)
    assert ((issues["Field"] == STATUS_FIELD) & (issues["Severity"] == "error")).any()

    missing_value = frame.copy()
    missing_value.loc[0, STATUS_FIELD] = pd.NA
    issues = validate_bangladesh_responses(missing_value)
    assert ((issues["Field"] == STATUS_FIELD) & (issues["Severity"] == "error")).any()

    invalid = frame.copy()
    invalid.loc[0, STATUS_FIELD] = "Withdrawn"
    assert STATUS_FIELD in codes(invalid)


def test_unknown_categories_are_preserved_as_missing():
    frame = synthetic_example_rows()
    frame.loc[0, "mother_occupation_bangladesh"] = "Unknown / Prefer not to say"
    frame.loc[1, "family_financial_condition"] = "Undisclosed"
    cleaned = clean_google_forms_responses(frame)
    assert pd.isna(cleaned.loc[0, "mother_occupation_bangladesh"])
    assert pd.isna(cleaned.loc[1, "family_financial_condition"])
    assert validate_bangladesh_responses(frame).empty


def test_feature_sets_are_predictor_only_and_have_expected_widths():
    frames = future_comparison_feature_sets(synthetic_example_rows())
    assert tuple(frames["shared_academic"].columns) == SHARED_ACADEMIC_FEATURES
    assert len(SHARED_ACADEMIC_FEATURES) == 5
    assert tuple(frames["shared_core"].columns) == SHARED_CORE_FEATURES
    assert len(SHARED_CORE_FEATURES) == 13
    assert len(BANGLADESH_EXTENDED_FEATURES) == 5
    assert tuple(frames["bangladesh_expanded"].columns) == PREDICTOR_FIELDS


def test_synthetic_examples_have_18_predictors_plus_target_and_are_valid():
    sample = synthetic_example_rows()
    assert list(sample.columns) == list(EXPORT_COLUMNS)
    assert len(sample) == 3
    assert validate_bangladesh_responses(sample).empty


def test_current_status_target_remains_separate_from_18_predictors():
    assert tuple(EXPORT_COLUMNS[:-1]) == tuple(PREDICTOR_FIELDS)
    assert EXPORT_COLUMNS[-1] == STATUS_FIELD
    assert len(EXPORT_COLUMNS) == 19


def test_collection_page_is_separate_and_does_not_change_production_schema():
    assert len(FORM_FIELDS) == 18


def test_questionnaire_export_cleaning_preserves_schema():
    cleaned = clean_google_forms_responses(synthetic_example_rows())
    assert list(cleaned.columns) == list(EXPORT_COLUMNS)
    assert len(FORM_FIELDS) == 18
