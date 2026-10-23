"""Framework-independent prediction, validation, and export services.

This module provides framework-independent inference and batch services for the Django site.
It loads the saved model only; it never trains or modifies an artifact.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.bangladesh_collection import (
    CATEGORICAL_CHOICES,
    FIELD_TITLES,
    PREDICTOR_FIELDS,
    STATUS_FIELD,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "combined_uci_bangladesh_18_v2" / "weighted_ensemble_dropout_status_18.joblib"
METADATA_PATH = ROOT / "models" / "combined_uci_bangladesh_18_v2" / "model_metadata.json"
FORM_FIELDS = list(PREDICTOR_FIELDS)
USER_FIELDS = [FIELD_TITLES[field] for field in FORM_FIELDS]
USER_FIELD_TO_KEY = dict(zip(USER_FIELDS, FORM_FIELDS))
ASSESSMENT_CATEGORICAL_CHOICES = {key: value for key, value in CATEGORICAL_CHOICES.items() if key in FORM_FIELDS}
ID_COLUMN = "Anonymous Follow-Up ID"
SEMESTER_COLUMN = "Current Semester / Academic Year"
BATCH_COLUMNS = [ID_COLUMN, SEMESTER_COLUMN, *USER_FIELDS]
RESULT_COLUMNS = [
    ID_COLUMN, SEMESTER_COLUMN,
    "Estimated Current Dropout-Status Probability",
    "UCI Academic Foundation Signal (research estimate)",
    "Bangladesh Local Model Signal (research estimate)",
    "Risk Level", "Main Indicator", "Key Risk Indicators",
    "Suggested Institute Support",
]


@lru_cache(maxsize=1)
def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Saved 18-input ensemble is missing: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact.get("pipeline")
    if (
        artifact.get("raw_input_features") != FORM_FIELDS
        or list(getattr(pipeline, "feature_names_in_", [])) != FORM_FIELDS
        or artifact.get("input_feature_count") != 18
        or artifact.get("target_is_predictor") is not False
        or artifact.get("target_field") != STATUS_FIELD
        or not getattr(pipeline, "blend_weight_uci", 0)
    ):
        raise RuntimeError("The saved ensemble does not match the approved 18-predictor schema.")
    return artifact


def make_input_frame(values: dict[str, Any]) -> pd.DataFrame:
    if set(values) != set(FORM_FIELDS):
        raise ValueError("Answers must contain exactly the 18 model input fields.")
    normalized = {field: values[field] for field in FORM_FIELDS}
    if normalized.get("marital_status_bangladesh") == "Single":
        normalized["marital_status_bangladesh"] = "Single / Never married"
    frame = pd.DataFrame([normalized], columns=FORM_FIELDS)
    for field in FORM_FIELDS:
        if field in CATEGORICAL_CHOICES:
            frame[field] = frame[field].astype(object).where(frame[field].notna(), float("nan"))
        else:
            frame[field] = pd.to_numeric(frame[field], errors="coerce").astype(float)
    return frame


def validate_inputs(values: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if set(values) != set(FORM_FIELDS):
        return ["Answers must contain exactly the 18 model input fields."]
    for field, low, high in (
        ("hsc_equivalent_gpa", 0, 5), ("age_at_enrollment", 15, 100),
        ("semester_gpa_native", 0, 5), ("last_semester_attendance_percent", 0, 100),
    ):
        try:
            value = float(values[field])
            if value < low or value > high:
                issues.append(f"{FIELD_TITLES[field]} must be between {low:g} and {high:g}.")
        except (TypeError, ValueError):
            issues.append(f"{FIELD_TITLES[field]} must be numeric.")
    for field in ("courses_enrolled", "evaluations_completed", "courses_passed"):
        try:
            if int(values[field]) != float(values[field]) or float(values[field]) < 0:
                issues.append(f"{FIELD_TITLES[field]} must be a non-negative whole number.")
        except (TypeError, ValueError):
            issues.append(f"{FIELD_TITLES[field]} must be numeric.")
    try:
        if float(values["courses_passed"]) > float(values["courses_enrolled"]):
            issues.append("Courses passed cannot be greater than courses enrolled.")
    except (TypeError, ValueError):
        pass
    for field, choices in ASSESSMENT_CATEGORICAL_CHOICES.items():
        if str(values[field]) not in choices:
            issues.append(f"{FIELD_TITLES[field]} has an unsupported value.")
    return issues


def dropout_probability(artifact: dict[str, Any], values: dict[str, Any]) -> float:
    if artifact.get("raw_input_features") != FORM_FIELDS or artifact.get("target_is_predictor") is not False:
        raise ValueError("The saved model does not support the exact combined 18-input schema.")
    pipeline = artifact["pipeline"]
    return float(pipeline.predict_proba(make_input_frame(values))[0, list(pipeline.classes_).index(1)])


def risk_label(probability: float, threshold: float | None = None) -> str:
    if threshold is None:
        threshold = load_model_artifact()["threshold"]
    return "Elevated Estimated Risk" if probability >= threshold else "Lower Estimated Risk"


def support_guidance(*, semester_gpa: float, enrolled: int, evaluations: int, passed: int,
                     debtor: bool, tuition_up_to_date: bool, scholarship: bool,
                     previous_gpa: float) -> tuple[list[str], list[str]]:
    pass_rate = passed / enrolled if enrolled else 0.0
    evaluation_rate = evaluations / enrolled if enrolled else 0.0
    candidates: list[tuple[int, str, list[str]]] = []
    if semester_gpa < 2.0:
        candidates.append((10, f"Low recent academic performance — Semester GPA: {semester_gpa:.2f} on the reported scale", ["Academic advising", "Tutoring / study support"]))
    if not enrolled:
        candidates.append((9, "No course enrollment was reported", ["Study-plan review"]))
    elif pass_rate < 0.5:
        candidates.append((9, f"Low course completion — {passed} of {enrolled} courses passed", ["Course-specific academic support", "Study-plan review"]))
    elif pass_rate < 0.7:
        candidates.append((5, f"Moderate course completion — {passed} of {enrolled} courses passed", ["Study-plan review"]))
    if not tuition_up_to_date:
        candidates.append((8, "Tuition fees are not currently up to date", ["Financial-aid review", "Installment/payment support if available"]))
    if debtor:
        candidates.append((8, "Outstanding debtor status was reported", ["Financial counselling", "Review possible payment arrangements"]))
    if enrolled and evaluation_rate < 0.75:
        candidates.append((7, f"Low evaluation completion — {evaluations} evaluations for {enrolled} enrolled courses", ["Student engagement follow-up", "Check attendance / participation difficulties"]))
    if not scholarship and (debtor or not tuition_up_to_date):
        candidates.append((4, "No scholarship was reported alongside a financial warning", ["Review scholarship or student-aid eligibility"]))
    if previous_gpa < 3.0:
        candidates.append((3, f"Weaker previous academic background — HSC / Equivalent GPA: {previous_gpa:.2f} / 5.00", ["Foundation/remedial academic support if needed"]))
    selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:3]
    supports: list[str] = []
    for _, _, recommendations in selected:
        if recommendations[0] not in supports:
            supports.append(recommendations[0])
    for _, _, recommendations in selected:
        for recommendation in recommendations[1:]:
            if recommendation not in supports and len(supports) < 4:
                supports.append(recommendation)
    return [item[1] for item in selected], supports


def prepare_student(record: dict[str, Any]) -> dict[str, Any]:
    if set(USER_FIELDS).issubset(record):
        values = {USER_FIELD_TO_KEY[title]: record[title] for title in USER_FIELDS}
    elif set(FORM_FIELDS).issubset(record):
        values = {field: record[field] for field in FORM_FIELDS}
    else:
        missing = [title for title in USER_FIELDS if title not in record]
        raise ValueError("Missing 18-input assessment fields: " + ", ".join(missing))
    integer_fields = {"age_at_enrollment", "courses_enrolled", "evaluations_completed", "courses_passed"}
    numeric_fields = {"hsc_equivalent_gpa", "semester_gpa_native", "last_semester_attendance_percent"}
    for field in integer_fields | numeric_fields:
        value = values[field]
        if pd.isna(value) or str(value).strip() == "":
            values[field] = float("nan")
        elif field in integer_fields:
            values[field] = int(float(value))
        else:
            values[field] = float(value)
    return values


def predict_student(artifact: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    values = prepare_student(record)
    probability = dropout_probability(artifact, values)
    local_signal, uci_signal = artifact["pipeline"].component_probabilities(make_input_frame(values))
    indicators, supports = support_guidance(
        semester_gpa=float(values["semester_gpa_native"]), enrolled=int(values["courses_enrolled"]),
        evaluations=int(values["evaluations_completed"]), passed=int(values["courses_passed"]),
        debtor=str(values["debtor"]) == "Yes", tuition_up_to_date=str(values["tuition_fees_up_to_date"]) == "Yes",
        scholarship=str(values["scholarship_holder"]) == "Yes", previous_gpa=float(values["hsc_equivalent_gpa"]),
    )
    attendance = values["last_semester_attendance_percent"]
    if attendance < 75:
        indicators.append(f"Low reported last-semester attendance — {attendance:.1f}%")
        supports.insert(0, "Attendance follow-up and student check-in")
    if str(values["family_financial_condition"]).startswith(("Poor", "Below Average")):
        indicators.append("Reported household income is in a lower provisional survey bracket")
        supports.append("Financial-aid review")
    if str(values["student_employment"]) in {"Full-time", "Part-time"}:
        indicators.append(f"Student reported {values['student_employment'].lower()} employment")
        supports.append("Review study/work schedule and flexibility")
    return {
        "probability": probability, "risk_level": risk_label(probability, artifact["threshold"]),
        "threshold": artifact["threshold"],
        "component_scores": {"uci_academic": float(uci_signal[0]), "bangladesh_local": float(local_signal[0])},
        "indicators": indicators[:3], "supports": list(dict.fromkeys(supports))[:4],
    }


def validate_batch_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in USER_FIELDS if column not in frame.columns]
    if missing:
        return pd.DataFrame([{"Row": "File", "Issue": "Missing required columns: " + ", ".join(missing)}])
    issues: list[dict[str, Any]] = []
    ranges = {
        "HSC / Equivalent GPA": (0, 5, False), "Age at enrollment": (15, 100, True),
        "Courses enrolled in the relevant semester": (0, 500, True),
        "Evaluations completed in the relevant semester": (0, 1000, True),
        "Courses passed in the relevant semester": (0, 500, True),
        "Semester GPA (native university scale)": (0, 5, False),
        "Last-semester attendance percentage": (0, 100, False),
    }
    for position, (_, row) in enumerate(frame.iterrows(), start=2):
        canonical = {USER_FIELD_TO_KEY[col]: row[col] for col in USER_FIELDS}
        def issue(message: str) -> None:
            issues.append({"Row": position, "Issue": message})
        for title, (minimum, maximum, integer) in ranges.items():
            value = row[title]
            if pd.isna(value) or str(value).strip() == "":
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                issue(f"{title} must be numeric.")
                continue
            if number < minimum or number > maximum or (integer and not number.is_integer()):
                kind = "whole number" if integer else "number"
                issue(f"{title} must be a {kind} in range {minimum:g}–{maximum:g}.")
        for field, choices in ASSESSMENT_CATEGORICAL_CHOICES.items():
            title = FIELD_TITLES[field]
            value = row[title]
            if pd.isna(value) or str(value).strip() == "":
                continue
            normalized = str(value).strip()
            if field == "marital_status_bangladesh" and normalized == "Single":
                normalized = "Single / Never married"
            if normalized not in choices:
                issue(f"{title} has an unsupported value.")
        try:
            if not pd.isna(canonical["courses_passed"]) and not pd.isna(canonical["courses_enrolled"]) and float(canonical["courses_passed"]) > float(canonical["courses_enrolled"]):
                issue("Courses passed cannot be greater than courses enrolled.")
        except (TypeError, ValueError):
            pass
    return pd.DataFrame(issues, columns=["Row", "Issue"])


def indicator_group(indicator: str) -> str:
    return "Financial" if any(word in indicator.lower() for word in ["tuition", "debtor", "scholarship"]) else "Academic"


def metadata_text(value: Any, fallback: str = "") -> str:
    return fallback if value is None or pd.isna(value) or not str(value).strip() else str(value).strip()


def analyze_batch(artifact: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    errors = validate_batch_dataframe(frame)
    if not errors.empty:
        raise ValueError(errors.to_json(orient="records"))
    rows = []
    for position, (_, source) in enumerate(frame.iterrows(), start=1):
        record = source.to_dict()
        result = predict_student(artifact, record)
        indicators = result["indicators"] if result["risk_level"].startswith("Elevated") else []
        rows.append({
            **record,
            ID_COLUMN: metadata_text(record.get(ID_COLUMN), f"Student {position}"),
            SEMESTER_COLUMN: metadata_text(record.get(SEMESTER_COLUMN)),
            "Estimated Current Dropout-Status Probability": result["probability"],
            "UCI Academic Foundation Signal (research estimate)": result["component_scores"]["uci_academic"],
            "Bangladesh Local Model Signal (research estimate)": result["component_scores"]["bangladesh_local"],
            "Risk Level": result["risk_level"], "Main Indicator": indicators[0] if indicators else "-",
            "Key Risk Indicators": " | ".join(indicators) if indicators else "-",
            "Suggested Institute Support": " | ".join(result["supports"]) if indicators else "-",
            "Warning Type": indicator_group(indicators[0]) if indicators else "-",
        })
    return pd.DataFrame(rows)


def export_results(frame: pd.DataFrame) -> bytes:
    columns = [*BATCH_COLUMNS, *RESULT_COLUMNS[2:]]
    return frame[[column for column in columns if column in frame.columns]].to_csv(index=False).encode("utf-8")


def template_frame() -> pd.DataFrame:
    example: dict[str, Any] = {ID_COLUMN: "EXAMPLE-001", SEMESTER_COLUMN: "Relevant semester"}
    for field in FORM_FIELDS:
        title = FIELD_TITLES[field]
        choices = ASSESSMENT_CATEGORICAL_CHOICES.get(field)
        if choices:
            example[title] = choices[0]
        elif field == "hsc_equivalent_gpa":
            example[title] = 4.2
        elif field == "age_at_enrollment":
            example[title] = 19
        elif field in {"courses_enrolled", "evaluations_completed", "courses_passed"}:
            example[title] = 5
        elif field == "semester_gpa_native":
            example[title] = 3.25
        elif field == "last_semester_attendance_percent":
            example[title] = 85.0
    return pd.DataFrame([example], columns=BATCH_COLUMNS)


def dataframe_to_excel(frame: pd.DataFrame, sheet_name: str) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def template_excel() -> bytes:
    return dataframe_to_excel(template_frame(), "Student Template")


def bangladesh_research_excel(frame: pd.DataFrame) -> bytes:
    """Export the versioned Bangladesh research schema without collection metadata."""
    return dataframe_to_excel(frame, "Bangladesh Survey")


def export_results_excel(frame: pd.DataFrame) -> bytes:
    columns = [*BATCH_COLUMNS, *RESULT_COLUMNS[2:]]
    return dataframe_to_excel(frame[[column for column in columns if column in frame.columns]], "Analysis Results")


def dataframe_to_excel(frame: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Data")
    return buffer.getvalue()


def read_uploaded_table(uploaded: Any, filename: str | None = None) -> pd.DataFrame:
    name = str(filename or getattr(uploaded, "name", "")).lower()
    if hasattr(uploaded, "seek"):
        uploaded.seek(0)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded, engine="openpyxl")
    raise ValueError("Unsupported file type. Please provide a CSV or Excel (.xlsx) file.")


def assessment_pdf(result: dict[str, Any], student_id: str) -> bytes:
    """Create a PDF from an already-calculated assessment without re-predicting."""
    commands: list[str] = []
    def text(x: float, y: float, value: Any, size: int = 10, color: tuple[float, float, float] = (.18, .25, .34), font: str = "F1") -> None:
        content = str(value).encode("cp1252", "replace").decode("cp1252")
        escaped = content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.append(f"BT /{font} {size} Tf {color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg {x:.1f} {y:.1f} Td ({escaped}) Tj ET")
    def rect(x: float, y: float, width: float, height: float, color: tuple[float, float, float], stroke: tuple[float, float, float] | None = None) -> None:
        commands.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg")
        if stroke:
            commands.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG 0.8 w")
        commands.append(f"{x:.1f} {y:.1f} {width:.1f} {height:.1f} re " + ("B" if stroke else "f"))
    def wrap(value: Any, limit: int = 82) -> list[str]:
        lines: list[str] = []
        line = ""
        for word in str(value).split():
            candidate = f"{line} {word}".strip()
            if len(candidate) > limit and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        return lines + ([line] if line else []) or [""]
    rect(0, 632, 612, 160, (.035, .095, .16)); rect(42, 632, 5, 160, (.10, .76, .88))
    text(62, 742, "STUDENT RISK ASSESSMENT", 9, (.36, .84, .94), "F2")
    text(62, 708, "Student Dropout Early Warning System", 19, (1, 1, 1), "F2")
    rect(42, 536, 528, 74, (.93, .965, .98), (.75, .83, .88))
    text(58, 583, "STUDENT REFERENCE", 8, (.30, .42, .53), "F2")
    for index, line in enumerate(wrap(student_id, 27)[:2]):
        text(58, 562 - index * 13, line)
    text(246, 583, "ESTIMATED RISK", 8, (.30, .42, .53), "F2")
    text(246, 552, f"{float(result['probability']):.1%}", 22, (.05, .30, .44), "F2")
    text(402, 583, "CLASSIFICATION", 8, (.30, .42, .53), "F2")
    for index, line in enumerate(wrap(result["risk_level"], 24)[:2]):
        text(402, 562 - index * 13, line, 10, (.13, .39, .45), "F2")
    y = 490
    for heading, key, fallback in (("KEY RISK INDICATORS", "indicators", "No specific indicators were reported."), ("RECOMMENDED INSTITUTE SUPPORT", "supports", "Continue normal academic monitoring.")):
        text(46, y, heading, 11, (.04, .45, .58), "F2"); y -= 12
        commands.append(f"0.12 0.79 0.87 RG 1.4 w 46 {y:.1f} m 566 {y:.1f} l S"); y -= 19
        for item in result.get(key) or [fallback]:
            for index, line in enumerate(wrap(item)):
                text(52, y, "•" if index == 0 else " ", 10, (.07, .62, .70), "F2")
                text(68, y, line); y -= 15
            y -= 4
        y -= 12
    text(46, 38, "Assessment formatted from the result displayed in the application.", 8, (.45, .52, .60))
    stream = ("\n".join(commands) + "\n").encode("cp1252", "replace")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream"]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"); offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf)); pdf.extend(f"{number} 0 obj\n".encode()); pdf.extend(obj); pdf.extend(b"\nendobj\n")
    xref = len(pdf); pdf.extend(f"xref\n0 {len(objects)+1}\n".encode()); pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return bytes(pdf)
