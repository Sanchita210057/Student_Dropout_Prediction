"""Database-backed collection services using the shared Project A schema."""
from __future__ import annotations

from datetime import date
import uuid

from django.db import IntegrityError, transaction
from django.utils import timezone

from src.bangladesh_collection import PREDICTOR_FIELDS
from src.collection_core import (
    CollectionClosedError, DuplicateStudentIDError, normalize_student_id,
    validate_student_submission, make_collection as build_collection,
    SEMESTER_OPTIONS,
)

from .models import Collection, EditHistory, StudentResponse, Teacher


def collection_payload(collection: Collection) -> dict:
    return {
        "collection_id": collection.collection_id,
        "title": collection.title,
        "department": collection.department,
        # Stored legacy semester selections are retained for history only; all
        # supported semesters are now available on every collection link.
        "semesters": list(SEMESTER_OPTIONS),
        "closing_date": collection.closing_date.isoformat() if collection.closing_date else None,
        "status": collection.status,
        "created_by": collection.owner.user.get_username(),
    }


def create_collection(*, owner: Teacher, title: str, department: str, closing_date: date | None,
                      semesters: list[str] | None = None) -> Collection:
    normalized = build_collection(title=title, department=department, semesters=semesters,
                                  closing_date=closing_date, created_by=owner.user.get_username())
    return Collection.objects.create(
        collection_id=normalized["collection_id"], owner=owner, title=normalized["title"],
        department=normalized["department"], semesters=normalized["semesters"],
        closing_date=closing_date, status=Collection.Status.OPEN,
    )


def set_collection_status(*, collection: Collection, status: str) -> Collection:
    if status not in {Collection.Status.OPEN, Collection.Status.CLOSED}:
        raise ValueError("Collection status must be open or closed.")
    collection.status = status
    collection.save(update_fields=("status",))
    return collection


def _student_payload(collection: Collection, student_id: str, semester: str, predictors: dict) -> dict:
    return validate_student_submission(collection_payload(collection), student_id, semester, predictors,
                                       today=timezone.localdate())


@transaction.atomic
def submit_response(*, collection_id: str, student_id: str, semester: str, predictors: dict) -> StudentResponse:
    try:
        collection = Collection.objects.select_for_update().get(collection_id=collection_id)
    except Collection.DoesNotExist as error:
        raise Collection.DoesNotExist("This collection link is invalid or no longer available.") from error
    payload = _student_payload(collection, student_id, semester, predictors)
    normalized_id = payload["student_id"]
    response = StudentResponse(
        collection=collection, record_id=uuid.uuid4(), student_id=normalized_id,
        student_id_key=normalize_student_id(normalized_id).casefold(), semester=payload["semester"],
        predictors={field: payload[field] for field in PREDICTOR_FIELDS},
    )
    try:
        response.save()
    except IntegrityError as error:
        raise DuplicateStudentIDError("That Student ID has already submitted to this collection. Contact your teacher if a correction is needed.") from error
    return response


def response_as_record(response: StudentResponse) -> dict:
    return {
        "record_id": str(response.record_id),
        "collection_id": response.collection.collection_id,
        "student_id": response.student_id,
        "semester": response.semester,
        **response.predictors,
        "submitted_at": response.submitted_at.isoformat(),
    }


def list_collection_records(*, collection: Collection) -> list[dict]:
    return [response_as_record(row) for row in collection.responses.all()]


@transaction.atomic
def edit_response(*, collection_id: str, record_id: uuid.UUID, teacher: Teacher,
                  student_id: str, semester: str, predictors: dict) -> StudentResponse:
    collection_query = Collection.objects.select_for_update().filter(collection_id=collection_id)
    if teacher.role != Teacher.Role.ADMIN:
        collection_query = collection_query.filter(owner=teacher)
    collection = collection_query.get()
    response = StudentResponse.objects.select_for_update().get(collection=collection, record_id=record_id)
    # Corrections remain possible after collection close or its deadline.
    collection_for_validation = collection_payload(collection)
    collection_for_validation["status"] = Collection.Status.OPEN
    collection_for_validation["closing_date"] = None
    payload = validate_student_submission(collection_for_validation, student_id, semester, predictors)
    before = {"student_id": response.student_id, "semester": response.semester, **response.predictors}
    normalized_id = payload["student_id"]
    response.student_id = normalized_id
    response.student_id_key = normalize_student_id(normalized_id).casefold()
    response.semester = payload["semester"]
    response.predictors = {field: payload[field] for field in PREDICTOR_FIELDS}
    try:
        response.save(update_fields=("student_id", "student_id_key", "semester", "predictors", "updated_at"))
    except IntegrityError as error:
        raise DuplicateStudentIDError("Another record in this collection already uses that Student ID.") from error
    after = {"student_id": response.student_id, "semester": response.semester, **response.predictors}
    EditHistory.objects.create(collection=collection, response=response, editor=teacher,
                               previous_values=before, new_values=after)
    return response
