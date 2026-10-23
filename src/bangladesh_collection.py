"""Bangladesh-specific questionnaire schema and import validation.

This 18-predictor plus one-target schema is separate from the production UCI model.
It contains no identity, consent, institution, date, or research metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "3.0.0"
STATUS_FIELD = "current_academic_status"
STATUS_TITLE = "Current Academic Status"
ACADEMIC_STATUS_OPTIONS = (
    "Running (Currently Studying)",
    "Passed (Graduated)",
    "Dropout (Discontinued Studies)",
)

PARENT_OCCUPATIONS = (
    "Not Alive", "Retired", "Housewife", "Business owner",
    "Government employee", "Private-sector employee",
    "Self-employed / Daily-wage worker", "Other", "Unknown / Prefer not to say",
)
PARENT_EDUCATION = (
    "No formal education", "Primary", "Lower secondary", "Secondary / SSC",
    "Higher secondary / HSC", "Bachelor's degree", "Postgraduate degree",
    "Other", "Unknown / Prefer not to say",
)
GENDER_OPTIONS = ("Female", "Male", "Another identity", "Unknown / Prefer not to say")
MARITAL_STATUS_OPTIONS = (
    "Single / Never married", "Married", "Separated", "Divorced", "Widowed",
    "Other", "Unknown / Prefer not to say",
)
YES_NO_UNKNOWN = ("Yes", "No", "Unknown / Prefer not to say")
FINANCIAL_CONDITION_OPTIONS = (
    "Poor (below ৳25,000)", "Below Average (৳25,000–39,999)",
    "Average (৳40,000–69,999)", "Higher Middle Class (৳70,000–149,999)",
    "Rich (৳150,000+)", "Undisclosed",
)
STUDENT_EMPLOYMENT_OPTIONS = (
    "Unemployed", "Part-time", "Full-time", "Private Tuition",
    "Freelancing/Outsourcing", "Own Business", "Undisclosed",
)

# Order is the Google Form order and the canonical CSV/Excel order.
PREDICTOR_FIELDS = (
    "hsc_equivalent_gpa",
    "age_at_enrollment",
    "mother_occupation_bangladesh",
    "father_occupation_bangladesh",
    "mother_education_bangladesh",
    "father_education_bangladesh",
    "gender_bangladesh",
    "marital_status_bangladesh",
    "scholarship_holder",
    "debtor",
    "tuition_fees_up_to_date",
    "courses_enrolled",
    "evaluations_completed",
    "courses_passed",
    "semester_gpa_native",
    "last_semester_attendance_percent",
    "family_financial_condition",
    "student_employment",
)
EXPORT_COLUMNS = (*PREDICTOR_FIELDS, STATUS_FIELD)

# Categories remain locally defined; they are not UCI occupation/education codes.
SHARED_CORE_FEATURES = (
    "hsc_equivalent_gpa", "age_at_enrollment", "mother_education_bangladesh",
    "father_education_bangladesh", "gender_bangladesh", "marital_status_bangladesh",
    "scholarship_holder", "debtor", "tuition_fees_up_to_date", "courses_enrolled",
    "evaluations_completed", "courses_passed", "semester_gpa_native",
)
SHARED_ACADEMIC_FEATURES = (
    "hsc_equivalent_gpa", "courses_enrolled", "evaluations_completed",
    "courses_passed", "semester_gpa_native",
)
BANGLADESH_EXTENDED_FEATURES = (
    "mother_occupation_bangladesh", "father_occupation_bangladesh",
    "last_semester_attendance_percent", "family_financial_condition", "student_employment",
)

FIELD_TITLES = {
    "hsc_equivalent_gpa": "HSC / Equivalent GPA",
    "age_at_enrollment": "Age at enrollment",
    "mother_occupation_bangladesh": "Mother's occupation",
    "father_occupation_bangladesh": "Father's occupation",
    "mother_education_bangladesh": "Mother's highest education",
    "father_education_bangladesh": "Father's highest education",
    "gender_bangladesh": "Gender",
    "marital_status_bangladesh": "Marital status",
    "scholarship_holder": "Scholarship holder",
    "debtor": "Have any debt",
    "tuition_fees_up_to_date": "Tuition fees up to date",
    "courses_enrolled": "Courses enrolled in the relevant semester",
    "evaluations_completed": "Evaluations completed in the relevant semester",
    "courses_passed": "Courses passed in the relevant semester",
    "semester_gpa_native": "Semester GPA (native university scale)",
    "last_semester_attendance_percent": "Last-semester attendance percentage",
    "family_financial_condition": "Family financial condition (monthly household income bracket)",
    "student_employment": "Student employment",
    STATUS_FIELD: STATUS_TITLE,
}
TITLE_TO_FIELD = {title.casefold(): field for field, title in FIELD_TITLES.items()}
TITLE_TO_FIELD["have any debt"] = "debtor"
TITLE_TO_FIELD["outstanding university fees / debt"] = "debtor"
TARGET_FIELD = STATUS_FIELD

_CATEGORICAL_CHOICES = {
    "mother_occupation_bangladesh": PARENT_OCCUPATIONS,
    "father_occupation_bangladesh": PARENT_OCCUPATIONS,
    "mother_education_bangladesh": PARENT_EDUCATION,
    "father_education_bangladesh": PARENT_EDUCATION,
    "gender_bangladesh": GENDER_OPTIONS,
    "marital_status_bangladesh": MARITAL_STATUS_OPTIONS,
    "scholarship_holder": YES_NO_UNKNOWN,
    "debtor": YES_NO_UNKNOWN,
    "tuition_fees_up_to_date": YES_NO_UNKNOWN,
    "family_financial_condition": FINANCIAL_CONDITION_OPTIONS,
    "student_employment": STUDENT_EMPLOYMENT_OPTIONS,
    STATUS_FIELD: ACADEMIC_STATUS_OPTIONS,
}
CATEGORICAL_CHOICES = _CATEGORICAL_CHOICES
_NUMERIC_RANGES = {
    "hsc_equivalent_gpa": (0.0, 5.0, False),
    "age_at_enrollment": (15.0, 100.0, True),
    "courses_enrolled": (0.0, 500.0, True),
    "evaluations_completed": (0.0, 1000.0, True),
    "courses_passed": (0.0, 500.0, True),
    "semester_gpa_native": (0.0, 5.0, False),
    "last_semester_attendance_percent": (0.0, 100.0, False),
}
NUMERIC_RANGES = _NUMERIC_RANGES
_MISSING_TOKENS = {
    "", "unknown", "unknown / prefer not to say", "prefer not to say",
    "undisclosed", "not reported", "not available", "n/a", "na", "nan",
}


def questionnaire_template_frame() -> pd.DataFrame:
    """Return the 18 predictor columns followed by the status target."""
    return pd.DataFrame(columns=EXPORT_COLUMNS)


def validate_predictor_record(values: dict[str, Any]) -> list[str]:
    """Validate one complete 18-input prediction/collection record.

    This shares the versioned questionnaire categories and numeric limits,
    while keeping Current Academic Status out of student submissions.
    """
    if set(values) != set(PREDICTOR_FIELDS):
        missing = [field for field in PREDICTOR_FIELDS if field not in values]
        extra = [field for field in values if field not in PREDICTOR_FIELDS]
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        return ["Student answers must contain exactly the 18 model inputs (" + "; ".join(details) + ")."]

    issues: list[str] = []
    for field, choices in _CATEGORICAL_CHOICES.items():
        if field == STATUS_FIELD:
            continue
        value = values[field]
        if pd.isna(value) or str(value).strip() not in choices:
            issues.append(f"{FIELD_TITLES[field]} must use one of the listed options.")
    for field, (minimum, maximum, integer) in _NUMERIC_RANGES.items():
        value = values[field]
        try:
            number = float(value)
        except (TypeError, ValueError):
            issues.append(f"{FIELD_TITLES[field]} must be numeric.")
            continue
        if not number == number or number < minimum or number > maximum or (integer and not number.is_integer()):
            kind = "a whole number" if integer else "a number"
            issues.append(f"{FIELD_TITLES[field]} must be {kind} from {minimum:g} to {maximum:g}.")
    try:
        if float(values["courses_passed"]) > float(values["courses_enrolled"]):
            issues.append("Courses passed cannot exceed courses enrolled.")
    except (TypeError, ValueError):
        pass
    return issues


def _canonical_field_name(column: Any) -> str | None:
    text = str(column).strip()
    if text in EXPORT_COLUMNS:
        return text
    return TITLE_TO_FIELD.get(text.casefold())


def clean_google_forms_responses(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Google Forms titles or machine names to the v3 schema.

    Incidental Google metadata (such as its automatic Timestamp column) and
    other unrelated columns are ignored. Missing predictor columns are blank.
    """
    renamed: dict[Any, str] = {}
    for column in frame.columns:
        canonical = _canonical_field_name(column)
        if canonical is not None:
            if canonical in renamed.values():
                raise ValueError(f"Duplicate Google Forms column for {canonical!r}.")
            renamed[column] = canonical
    clean = frame.rename(columns=renamed)
    clean = clean.loc[:, [column for column in clean.columns if column in EXPORT_COLUMNS]].copy()
    for column in EXPORT_COLUMNS:
        if column not in clean:
            clean[column] = pd.NA
    clean = clean.loc[:, EXPORT_COLUMNS]

    for column in PREDICTOR_FIELDS:
        if column in _NUMERIC_RANGES:
            clean[column] = pd.to_numeric(clean[column], errors="coerce").astype("Float64")
        else:
            clean[column] = clean[column].map(
                lambda value: pd.NA
                if pd.isna(value) or str(value).strip().casefold() in _MISSING_TOKENS
                else str(value).strip()
            ).astype("string")
            if column == "marital_status_bangladesh":
                clean[column] = clean[column].replace({"Single": "Single / Never married"})
    clean[STATUS_FIELD] = clean[STATUS_FIELD].map(
        lambda value: pd.NA
        if pd.isna(value) or str(value).strip().casefold() in _MISSING_TOKENS
        else str(value).strip()
    ).astype("string")
    return clean


def validate_bangladesh_responses(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate imported predictors and the required status target."""
    issues: list[dict[str, Any]] = []
    raw_by_field: dict[str, pd.Series] = {}
    for column in frame.columns:
        canonical = _canonical_field_name(column)
        if canonical is not None:
            if canonical in raw_by_field:
                issues.append({"Row": "File", "Field": canonical, "Severity": "error", "Issue": "More than one column maps to this question."})
            raw_by_field[canonical] = frame[column]

    if not raw_by_field:
        issues.append({"Row": "File", "Field": "", "Severity": "error", "Issue": "No recognized Bangladesh predictor columns were found."})
    for field in EXPORT_COLUMNS:
        if field not in raw_by_field:
            severity = "error" if field == STATUS_FIELD else "warning"
            message = f"Required target column is missing: {FIELD_TITLES[field]}." if field == STATUS_FIELD else f"Question column is absent and will be blank: {FIELD_TITLES[field]}."
            issues.append({"Row": "File", "Field": field, "Severity": severity, "Issue": message})

    clean = clean_google_forms_responses(frame)
    for position, (index, row) in enumerate(clean.iterrows()):
        row_number = index + 2 if isinstance(index, int) else str(index)

        def add(field: str, message: str, severity: str = "error") -> None:
            issues.append({"Row": row_number, "Field": field, "Severity": severity, "Issue": message})

        if pd.isna(row[STATUS_FIELD]):
            add(STATUS_FIELD, f"{STATUS_TITLE} is required.")

        for field, choices in _CATEGORICAL_CHOICES.items():
            value = row[field]
            if not pd.isna(value) and str(value) not in choices:
                add(field, f"Unsupported value {value!r}.")

        for field, (minimum, maximum, integer) in _NUMERIC_RANGES.items():
            value = row[field]
            raw_value = raw_by_field[field].iloc[position] if field in raw_by_field else pd.NA
            if pd.isna(value):
                if not pd.isna(raw_value) and str(raw_value).strip().casefold() not in _MISSING_TOKENS:
                    add(field, f"Must be numeric in the range {minimum:g}–{maximum:g}.")
                continue
            number = float(value)
            if number < minimum or number > maximum or (integer and not number.is_integer()):
                unit = "whole number" if integer else "number"
                add(field, f"Must be a {unit} in the range {minimum:g}–{maximum:g}.")

        if not pd.isna(row["courses_passed"]) and not pd.isna(row["courses_enrolled"]):
            if row["courses_passed"] > row["courses_enrolled"]:
                add("courses_passed", "Courses passed cannot exceed courses enrolled.")

    return pd.DataFrame(issues, columns=["Row", "Field", "Severity", "Issue"])


def future_comparison_feature_sets(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare feature-only matrices; outcomes must come from a separate source."""
    clean = clean_google_forms_responses(frame)
    return {
        "shared_academic": clean.loc[:, SHARED_ACADEMIC_FEATURES].copy(),
        "shared_core": clean.loc[:, SHARED_CORE_FEATURES].copy(),
        "bangladesh_expanded": clean.loc[:, PREDICTOR_FIELDS].copy(),
    }


def academic_status_target(frame: pd.DataFrame) -> pd.Series:
    """Map the separate required academic-status label to the requested target.

    Running and Passed are coded 0, while Dropout is coded 1. This is a
    current-status label, not evidence of a future dropout outcome.
    """
    clean = clean_google_forms_responses(frame)
    mapping = {
        "Running (Currently Studying)": 0,
        "Passed (Graduated)": 0,
        "Dropout (Discontinued Studies)": 1,
    }
    return clean[STATUS_FIELD].map(mapping).astype("Int64").rename("dropout_target")


def prepare_bangladesh_training_data(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Normalize raw questionnaire responses into 18 features and a separate label.

    Timestamp and Current Academic Status never enter ``X``. A zero Semester GPA
    is treated as unpublished only where at least one, but not all, enrolled
    courses were passed. Rows and the three course counts are preserved.
    """
    normalized = clean_google_forms_responses(raw)
    issues = validate_bangladesh_responses(raw)
    errors = issues.loc[issues["Severity"].eq("error")] if not issues.empty else issues
    if not errors.empty:
        summary = errors[["Row", "Field", "Issue"]].head(10).to_dict("records")
        raise ValueError(f"Invalid Bangladesh training data: {summary}")

    original_status = normalized[STATUS_FIELD].copy().rename("original_current_academic_status")
    y = academic_status_target(normalized)
    X = normalized.loc[:, PREDICTOR_FIELDS].copy()
    unavailable_gpa = (
        X["semester_gpa_native"].eq(0)
        & X["courses_passed"].gt(0)
        & X["courses_passed"].lt(X["courses_enrolled"])
    ).fillna(False)
    zero_gpa_count = int(X["semester_gpa_native"].eq(0).sum())
    affected_status_counts = original_status.loc[unavailable_gpa].value_counts().to_dict()
    X.loc[unavailable_gpa, "semester_gpa_native"] = pd.NA

    audit = {
        "records": int(len(X)),
        "predictor_count": int(X.shape[1]),
        "target_column_excluded_from_features": STATUS_FIELD not in X.columns,
        "timestamp_column_excluded_from_features": "Timestamp" not in X.columns,
        "source_zero_semester_gpa_count": zero_gpa_count,
        "semester_gpa_marked_unavailable_count": int(unavailable_gpa.sum()),
        "semester_gpa_marked_unavailable_status_counts": {str(key): int(value) for key, value in affected_status_counts.items()},
        "rows_removed": 0,
        "course_counts_changed": False,
        "status_counts": {str(key): int(value) for key, value in original_status.value_counts(dropna=False).items()},
        "target_counts": {str(int(key)): int(value) for key, value in y.value_counts(dropna=False).sort_index().items()},
    }
    return X, y, original_status, audit


def load_bangladesh_training_data(
    path: str | Path,
    sheet_name: str = "Form Responses 1",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Read the authoritative raw Google Forms response sheet and prepare it."""
    raw = pd.read_excel(path, sheet_name=sheet_name)
    return prepare_bangladesh_training_data(raw)


def synthetic_example_rows() -> pd.DataFrame:
    """Return invented demonstration rows using only the 18 predictors."""
    rows = [
        {
            "hsc_equivalent_gpa": 4.25, "age_at_enrollment": 19,
            "mother_occupation_bangladesh": "Private-sector employee", "father_occupation_bangladesh": "Business owner",
            "mother_education_bangladesh": "Secondary / SSC", "father_education_bangladesh": "Bachelor's degree",
            "gender_bangladesh": "Female", "marital_status_bangladesh": "Single / Never married",
            "scholarship_holder": "No", "debtor": "No", "tuition_fees_up_to_date": "Yes",
            "courses_enrolled": 5, "evaluations_completed": 5, "courses_passed": 4,
            "semester_gpa_native": 3.18, "last_semester_attendance_percent": 87.5,
            "family_financial_condition": "Average (৳40,000–69,999)", "student_employment": "Private Tuition",
            STATUS_FIELD: "Running (Currently Studying)",
        },
        {
            "hsc_equivalent_gpa": 3.70, "age_at_enrollment": 21,
            "mother_occupation_bangladesh": "Housewife", "father_occupation_bangladesh": "Government employee",
            "mother_education_bangladesh": "Primary", "father_education_bangladesh": "Higher secondary / HSC",
            "gender_bangladesh": "Male", "marital_status_bangladesh": "Single / Never married",
            "scholarship_holder": "Yes", "debtor": "Unknown / Prefer not to say", "tuition_fees_up_to_date": "Yes",
            "courses_enrolled": 6, "evaluations_completed": 4, "courses_passed": 3,
            "semester_gpa_native": 2.54, "last_semester_attendance_percent": pd.NA,
            "family_financial_condition": "Below Average (৳25,000–39,999)", "student_employment": "Part-time",
            STATUS_FIELD: "Passed (Graduated)",
        },
        {
            "hsc_equivalent_gpa": 4.60, "age_at_enrollment": 18,
            "mother_occupation_bangladesh": "Unknown / Prefer not to say", "father_occupation_bangladesh": "Self-employed / Daily-wage worker",
            "mother_education_bangladesh": "Unknown / Prefer not to say", "father_education_bangladesh": "Secondary / SSC",
            "gender_bangladesh": "Unknown / Prefer not to say", "marital_status_bangladesh": "Single / Never married",
            "scholarship_holder": "No", "debtor": "No", "tuition_fees_up_to_date": "Unknown / Prefer not to say",
            "courses_enrolled": 4, "evaluations_completed": 3, "courses_passed": 3,
            "semester_gpa_native": 3.70, "last_semester_attendance_percent": 91.0,
            "family_financial_condition": "Undisclosed", "student_employment": "Undisclosed",
            STATUS_FIELD: "Dropout (Discontinued Studies)",
        },
    ]
    return clean_google_forms_responses(pd.DataFrame(rows))
