"""Framework-independent collection validation and record adaptation."""
from __future__ import annotations

from datetime import date, datetime, timezone
import secrets
from typing import Any

import pandas as pd

from src.bangladesh_collection import FIELD_TITLES, PREDICTOR_FIELDS, validate_predictor_record

STUDENT_ID_FIELD = "student_id"
SEMESTER_FIELD = "semester"
SEMESTER_OPTIONS = tuple([f"Semester {number}" for number in range(1, 13)] + ["Summer", "Trimester 1", "Trimester 2", "Trimester 3"])


class WorkspaceError(ValueError):
    """Expected, user-displayable collection or authorization error."""


class DuplicateStudentIDError(WorkspaceError):
    pass


class CollectionClosedError(WorkspaceError):
    pass


class CollectionNotFoundError(WorkspaceError):
    pass


class TeacherAccessError(WorkspaceError):
    pass


def generate_collection_id() -> str:
    """Create a URL-safe, 256-bit unpredictable student collection token."""
    return secrets.token_urlsafe(32)


def normalize_student_id(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise WorkspaceError("Student ID is required.")
    if len(text) > 100:
        raise WorkspaceError("Student ID must be 100 characters or fewer.")
    return text


def validate_collection_definition(title: Any, department: Any,
                                   closing_date: date | str | None) -> tuple[str, str, list[str], str | None]:
    clean_title = " ".join(str(title or "").split())
    clean_department = " ".join(str(department or "").split())
    if not clean_title:
        raise WorkspaceError("Collection title is required.")
    if not clean_department:
        raise WorkspaceError("Department/class is required.")
    if len(clean_title) > 120 or len(clean_department) > 120:
        raise WorkspaceError("Collection title and department/class must be 120 characters or fewer.")
    if closing_date in (None, ""):
        closing = None
    elif isinstance(closing_date, date):
        closing = closing_date.isoformat()
    else:
        try:
            closing = date.fromisoformat(str(closing_date)).isoformat()
        except ValueError as error:
            raise WorkspaceError("Closing date must be a valid date.") from error
    if closing and date.fromisoformat(closing) < datetime.now(timezone.utc).date():
        raise WorkspaceError("Closing date cannot be in the past.")
    return clean_title, clean_department, list(SEMESTER_OPTIONS), closing


def validate_student_submission(collection: dict[str, Any], student_id: Any, semester: Any,
                                predictors: dict[str, Any], *, today: date | None = None) -> dict[str, Any]:
    if collection.get("status") != "open":
        raise CollectionClosedError("This collection is closed and is no longer accepting submissions.")
    close_value = collection.get("closing_date")
    if close_value and date.fromisoformat(str(close_value)) < (today or datetime.now(timezone.utc).date()):
        raise CollectionClosedError("The closing date for this collection has passed.")
    clean_student_id = normalize_student_id(student_id)
    clean_semester = " ".join(str(semester or "").split())
    if not clean_semester:
        raise WorkspaceError("Semester is required.")
    if clean_semester not in SEMESTER_OPTIONS:
        raise WorkspaceError("Choose a semester from the supported semester list.")
    errors = validate_predictor_record(predictors)
    if errors:
        raise WorkspaceError("Please correct the student information: " + " ".join(errors))
    return {STUDENT_ID_FIELD: clean_student_id, SEMESTER_FIELD: clean_semester,
            **{field: predictors[field] for field in PREDICTOR_FIELDS}}


def make_collection(*, title: str, department: str, closing_date: str | None,
                    semesters: list[str] | None = None,
                    created_by: str, collection_id: str | None = None) -> dict[str, Any]:
    clean_title, clean_department, clean_semesters, closing = validate_collection_definition(
        title, department, closing_date)
    if not str(created_by or "").strip():
        raise TeacherAccessError("An authenticated teacher account is required to create a collection.")
    return {
        "collection_id": collection_id or generate_collection_id(), "title": clean_title,
        "department": clean_department, "semesters": clean_semesters,
        "closing_date": closing, "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": str(created_by).strip().casefold(),
    }


def records_to_batch_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Adapt private collection rows to the application's shared batch schema."""
    rows = []
    for record in records:
        row = {"Anonymous Follow-Up ID": record[STUDENT_ID_FIELD],
               "Current Semester / Academic Year": record[SEMESTER_FIELD]}
        row.update({FIELD_TITLES[field]: record[field] for field in PREDICTOR_FIELDS})
        rows.append(row)
    columns = ["Anonymous Follow-Up ID", "Current Semester / Academic Year",
               *[FIELD_TITLES[field] for field in PREDICTOR_FIELDS]]
    return pd.DataFrame(rows, columns=columns)
