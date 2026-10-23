from __future__ import annotations

from django.conf import settings
from django.db import models


class Teacher(models.Model):
    """Institute role profile; Django staff/superuser flags remain independent."""
    class Role(models.TextChoices):
        ADMIN = "admin", "Institute Administrator"
        TEACHER = "teacher", "Teacher"
        STAFF = "staff", "Staff"

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="teacher_profile")
    role = models.CharField(max_length=12, choices=Role.choices, default=Role.TEACHER)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.user.get_username()


class Collection(models.Model):
    class Status(models.TextChoices):
        OPEN = "open", "Open"
        CLOSED = "closed", "Closed"

    collection_id = models.CharField(max_length=64, unique=True, db_index=True)
    owner = models.ForeignKey(Teacher, on_delete=models.PROTECT, related_name="collections")
    title = models.CharField(max_length=120)
    department = models.CharField(max_length=120)
    semesters = models.JSONField(default=list)
    closing_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=8, choices=Status.choices, default=Status.OPEN)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self) -> str:
        return f"{self.title} ({self.status})"


class StudentResponse(models.Model):
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name="responses")
    record_id = models.UUIDField(unique=True, editable=False)
    student_id = models.CharField(max_length=100)
    student_id_key = models.CharField(max_length=100)
    semester = models.CharField(max_length=40)
    predictors = models.JSONField()
    submitted_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("submitted_at", "pk")
        constraints = [models.UniqueConstraint(fields=("collection", "student_id_key"), name="unique_student_id_per_collection")]

    def __str__(self) -> str:
        return f"{self.student_id} · {self.semester}"


class EditHistory(models.Model):
    collection = models.ForeignKey(Collection, on_delete=models.PROTECT, related_name="edit_history")
    response = models.ForeignKey(StudentResponse, on_delete=models.PROTECT, related_name="edit_history")
    editor = models.ForeignKey(Teacher, on_delete=models.PROTECT, related_name="edits")
    edited_at = models.DateTimeField(auto_now_add=True)
    previous_values = models.JSONField()
    new_values = models.JSONField()

    class Meta:
        ordering = ("-edited_at",)

    def __str__(self) -> str:
        return f"Edit {self.response.record_id} by {self.editor}"
