from datetime import date, timedelta

import pytest

from src.bangladesh_collection import PREDICTOR_FIELDS, synthetic_example_rows
from src.collection_core import (
    CollectionClosedError,
    WorkspaceError,
    generate_collection_id,
    make_collection,
    normalize_student_id,
    records_to_batch_frame,
    validate_student_submission,
)


def predictors():
    return synthetic_example_rows().iloc[0].loc[list(PREDICTOR_FIELDS)].to_dict()


def collection(**overrides):
    values = {
        "title": "Semester 1 Support", "department": "Computer Science",
        "semesters": ["Semester 1", "Semester 2"], "closing_date": None,
        "created_by": "teacher@example.edu",
    }
    values.update(overrides)
    return make_collection(**values)


def test_collection_ids_are_unpredictable_and_submission_schema_is_exact():
    one, two = generate_collection_id(), generate_collection_id()
    assert one != two and len(one) >= 40
    saved = validate_student_submission(collection(), " CS 001 ", "Semester 1", predictors())
    assert saved["student_id"] == "CS 001"
    assert tuple(field for field in saved if field not in {"student_id", "semester"}) == PREDICTOR_FIELDS
    assert len(saved) == 20 and "current_academic_status" not in saved


def test_collection_and_submission_validation_reject_invalid_values():
    with pytest.raises(WorkspaceError, match="title"):
        collection(title="")
    with pytest.raises(WorkspaceError, match="Student ID"):
        validate_student_submission(collection(), "", "Semester 1", predictors())
    allowed_by_default = validate_student_submission(collection(), "CS-1", "Semester 12", predictors())
    assert allowed_by_default["semester"] == "Semester 12"
    with pytest.raises(WorkspaceError, match="supported semester"):
        validate_student_submission(collection(), "CS-1", "Semester 99", predictors())
    invalid = predictors()
    invalid["last_semester_attendance_percent"] = 101
    with pytest.raises(WorkspaceError, match="attendance"):
        validate_student_submission(collection(), "CS-1", "Semester 1", invalid)


def test_collection_status_closing_date_and_student_id_normalization():
    with pytest.raises(WorkspaceError, match="100 characters"):
        normalize_student_id("x" * 101)
    closed = {**collection(), "status": "closed"}
    with pytest.raises(CollectionClosedError, match="closed"):
        validate_student_submission(closed, "CS-1", "Semester 1", predictors())
    expired = {**collection(), "closing_date": (date.today() - timedelta(days=1)).isoformat()}
    with pytest.raises(CollectionClosedError, match="passed"):
        validate_student_submission(expired, "CS-1", "Semester 1", predictors(), today=date.today())


def test_collected_records_adapt_to_the_existing_18_input_batch_schema():
    record = {"student_id": "CS-001", "semester": "Semester 1", **predictors()}
    frame = records_to_batch_frame([record])
    assert frame.shape == (1, 20)
    assert frame.loc[0, "Anonymous Follow-Up ID"] == "CS-001"
    assert frame.loc[0, "Mother's occupation"] == record["mother_occupation_bangladesh"]
    assert "Current Academic Status" not in frame.columns
