"""Inspect the active UCI student dropout dataset.

This script performs descriptive and timing/leakage checks only. It does not
transform the saved CSV, train a model, or run the Django application.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from src.training_data_paths import uci_dataset_path

DEFAULT_DATASET = uci_dataset_path()
TARGET = "Target"

# UCI stores these semantic categories as integer codes.
CATEGORICAL_CODE_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",  # Official UCI spelling in the CSV.
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Application order is an ordered code rather than a nominal category. It is
# listed here only so its documented 0-9 values are shown during inspection.
CODE_VALUE_COLUMNS = CATEGORICAL_CODE_COLUMNS + ["Application order"]

ADMISSION_TIME_FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
]

FIRST_SEMESTER_FEATURES = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

SECOND_SEMESTER_FEATURES = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

CONTEXTUAL_FEATURES = ["Unemployment rate", "Inflation rate", "GDP"]

BINARY_CODE_COLUMNS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

GRADE_COLUMNS = [
    "Previous qualification (grade)",
    "Admission grade",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"CSV path (default: {DEFAULT_DATASET})",
    )
    return parser.parse_args()


def heading(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def detect_delimiter(path: Path) -> str:
    first_line = path.read_text(encoding="utf-8-sig", errors="strict").splitlines()[0]
    return ";" if first_line.count(";") > first_line.count(",") else ","


def load_dataset(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    delimiter = detect_delimiter(path)
    df = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")
    original_columns = [str(column) for column in df.columns]
    df.columns = [str(column).strip() for column in df.columns]
    normalized = [
        f"{old!r} -> {new!r}"
        for old, new in zip(original_columns, df.columns, strict=True)
        if old != new
    ]
    return df, delimiter, normalized


def feature_stage(column: str) -> str:
    if column == TARGET:
        return "Target"
    if column in ADMISSION_TIME_FEATURES:
        return "Admission-time feature"
    if column in FIRST_SEMESTER_FEATURES:
        return "First-semester feature"
    if column in SECOND_SEMESTER_FEATURES:
        return "Second-semester feature"
    if column in CONTEXTUAL_FEATURES:
        return "Contextual/economic feature"
    return "Potential leakage or undocumented"


def find_identifier_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(
        r"(?:^unnamed:|^index$|^id$|student.?id|registration.?id|identifier|^name$)",
        flags=re.IGNORECASE,
    )
    return [str(column) for column in df.columns if pattern.search(str(column))]


def print_target_distribution(df: pd.DataFrame) -> None:
    distribution = df[TARGET].value_counts(dropna=False)
    percentages = df[TARGET].value_counts(dropna=False, normalize=True) * 100
    summary = pd.DataFrame({"count": distribution, "percentage": percentages})
    print("Original three-class target:")
    print(summary.to_string(float_format=lambda value: f"{value:.2f}%"))

    resolved = df[df[TARGET].isin(["Dropout", "Graduate"])].copy()
    resolved["dropout_binary"] = (resolved[TARGET] == "Dropout").astype("int8")
    binary_counts = resolved["dropout_binary"].value_counts().sort_index()
    binary_pct = resolved["dropout_binary"].value_counts(normalize=True).sort_index() * 100
    binary_summary = pd.DataFrame({"count": binary_counts, "percentage": binary_pct})
    print("\nRecommended resolved-outcome binary target (0 = Graduate, 1 = Dropout):")
    print(binary_summary.to_string(float_format=lambda value: f"{value:.2f}%"))
    print(f"Rows retained: {len(resolved)}; Enrolled rows excluded: {(df[TARGET] == 'Enrolled').sum()}")


def print_unusual_values(df: pd.DataFrame, normalized_headers: list[str]) -> None:
    if normalized_headers:
        print("Header whitespace normalized in memory (saved CSV left unchanged):")
        for change in normalized_headers:
            print(f"- {change}")

    print(f"- Age at enrollment range: {df['Age at enrollment'].min()} to {df['Age at enrollment'].max()}")
    print(
        "- The official column name 'Nacionality' is misspelled; it is preserved for schema compatibility."
    )
    print(
        "- UCI code 34 means 'Unknown' in parental qualification fields; code 99 means "
        "blank in parental occupation fields. These are coded unknowns, not pandas nulls."
    )

    for column in GRADE_COLUMNS:
        minimum = df[column].min()
        maximum = df[column].max()
        expected_max = 200 if column in {"Previous qualification (grade)", "Admission grade"} else 20
        invalid = ((df[column] < 0) | (df[column] > expected_max)).sum()
        print(
            f"- {column}: range {minimum:g} to {maximum:g}; "
            f"values outside 0-{expected_max}: {int(invalid)}"
        )

    for column in FIRST_SEMESTER_FEATURES + SECOND_SEMESTER_FEATURES:
        zero_count = int((df[column] == 0).sum())
        print(
            f"- {column}: min={df[column].min():g}, max={df[column].max():g}, "
            f"zeros={zero_count} ({zero_count / len(df) * 100:.2f}%)"
        )

    invalid_binary = {
        column: sorted(set(df[column].dropna().unique()) - {0, 1})
        for column in BINARY_CODE_COLUMNS
        if set(df[column].dropna().unique()) - {0, 1}
    }
    print(f"- Invalid values in documented binary-code fields: {invalid_binary or 'none'}")

    rare_codes: list[str] = []
    for column in CATEGORICAL_CODE_COLUMNS:
        counts = df[column].value_counts(dropna=False)
        rare = {str(code): int(count) for code, count in counts.items() if count < 10}
        if rare:
            rare_codes.append(f"{column}: {rare}")
    print("- Rare categorical codes (fewer than 10 students):")
    for item in rare_codes:
        print(f"  {item}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df, delimiter, normalized_headers = load_dataset(dataset_path)
    missing_columns = {
        *ADMISSION_TIME_FEATURES,
        *FIRST_SEMESTER_FEATURES,
        *SECOND_SEMESTER_FEATURES,
        *CONTEXTUAL_FEATURES,
        TARGET,
    }.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset schema is missing expected columns: {sorted(missing_columns)}")

    categorical = CATEGORICAL_CODE_COLUMNS + [TARGET]
    numerical = [column for column in df.columns if column not in categorical]

    heading("DATASET")
    print(f"Path: {dataset_path}")
    print(f"Detected delimiter: {delimiter!r}")
    print(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Predictor count: {df.shape[1] - 1}; target count: 1")

    heading("FEATURE NAMES AND AVAILABILITY")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:>2}. {column} [{feature_stage(column)}]")

    heading("DATA TYPES")
    print(df.dtypes.to_string())

    heading("FIRST 10 ROWS")
    with pd.option_context("display.max_columns", None, "display.width", 300):
        print(df.head(10).to_string(index=False))

    heading("MISSING VALUES")
    print(df.isna().sum().to_string())
    print(f"Total missing cells: {int(df.isna().sum().sum())}")
    print("Coded unknown/blank values are assessed separately under unusual values.")

    heading("DUPLICATE ROWS")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")

    heading("CATEGORICAL / CODE-BASED UNIQUE VALUES")
    for column in CODE_VALUE_COLUMNS + [TARGET]:
        values = sorted(df[column].drop_duplicates().tolist(), key=lambda value: str(value))
        print(f"{column} ({len(values)} unique): {values}")

    heading("TARGET DISTRIBUTION")
    print_target_distribution(df)

    heading("SEMANTIC FEATURE TYPES")
    print(f"Categorical/code-based features ({len(CATEGORICAL_CODE_COLUMNS)}):")
    print(CATEGORICAL_CODE_COLUMNS)
    print(f"\nNumerical/ordinal/count features ({len(numerical)}):")
    print(numerical)

    heading("SUSPICIOUS IDENTIFIERS")
    identifiers = find_identifier_columns(df)
    print(identifiers if identifiers else "None detected. Each row represents a student, but no row ID is supplied.")

    heading("UNUSUAL VALUES AND DATA-QUALITY WARNINGS")
    print_unusual_values(df, normalized_headers)

    heading("POSSIBLE DATA LEAKAGE")
    print("Recommended prediction point: at enrollment, before first-semester teaching begins.")
    print(f"- Direct leakage: {TARGET} is the outcome and must never enter the feature matrix.")
    print(
        "- Post-cutoff fields: all first- and second-semester curricular-unit fields are "
        "unavailable at enrollment and must be excluded from the recommended model. They are "
        "legitimate predictors at a later cutoff, not inherently leaking variables."
    )
    print(
        "- Timing ambiguity: UCI provides final status but no dropout date. At an end-of-second-"
        "semester cutoff, some dropout outcomes may already have occurred; zero enrollments, "
        "evaluations, approvals, or grades could then encode an existing dropout rather than future risk."
    )
    print(
        "- Administrative timing check: confirm that Debtor and Tuition fees up to date are true "
        "enrollment-time snapshots in the intended deployment system. UCI groups them with enrollment "
        "information, but the field descriptions do not state an exact timestamp."
    )
    print("- No identifier-like predictor was detected.")

    heading("RECOMMENDED INPUT FEATURES")
    final_features = ADMISSION_TIME_FEATURES + CONTEXTUAL_FEATURES
    print(f"Count: {len(final_features)}")
    for column in final_features:
        print(f"- {column}")


if __name__ == "__main__":
    main()
