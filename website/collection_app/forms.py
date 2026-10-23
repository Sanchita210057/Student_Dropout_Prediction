from __future__ import annotations

from django import forms
from django.core.exceptions import ValidationError

from src.bangladesh_collection import CATEGORICAL_CHOICES, FIELD_TITLES, PREDICTOR_FIELDS, NUMERIC_RANGES, validate_predictor_record
from src.collection_core import SEMESTER_OPTIONS, validate_collection_definition


class CollectionCreateForm(forms.Form):
    title = forms.CharField(max_length=120, label="Collection title")
    department = forms.CharField(max_length=120, label="Department / class")
    closing_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}), label="Optional closing date")

    def clean(self):
        cleaned = super().clean()
        if not all(name in cleaned for name in ("title", "department")):
            return cleaned
        try:
            validate_collection_definition(cleaned["title"], cleaned["department"], cleaned.get("closing_date"))
        except ValueError as error:
            raise ValidationError(str(error)) from error
        return cleaned


class PredictorForm(forms.Form):
    """The single canonical 18-predictor form shared by assessments and collection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in PREDICTOR_FIELDS:
            title = FIELD_TITLES[field]
            choices = CATEGORICAL_CHOICES.get(field)
            if choices:
                self.fields[field] = forms.ChoiceField(
                    required=True, label=title,
                    choices=[("", "Select an option"), *[(choice, choice) for choice in choices]],
                )
                continue
            minimum, maximum, integer = NUMERIC_RANGES[field]
            widget = forms.NumberInput(attrs={"step": "1" if integer else "0.01"})
            field_class = forms.IntegerField if integer else forms.FloatField
            self.fields[field] = field_class(required=True, label=title, min_value=minimum,
                                             max_value=maximum, widget=widget)

    def clean_student_id(self):
        value = " ".join(self.cleaned_data["student_id"].split())
        if not value:
            raise ValidationError("Student ID is required.")
        return value

    def clean(self):
        cleaned = super().clean()
        if not all(field in cleaned for field in PREDICTOR_FIELDS):
            return cleaned
        errors = validate_predictor_record({field: cleaned[field] for field in PREDICTOR_FIELDS})
        if errors:
            raise ValidationError(errors)
        return cleaned


class IndividualAssessmentForm(PredictorForm):
    reference_id = forms.CharField(max_length=100, required=False, label="Student reference ID (optional)")


class StudentSubmissionForm(PredictorForm):
    student_id = forms.CharField(max_length=100, required=True, label="Student ID")
    semester = forms.ChoiceField(required=True, label="Semester")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["semester"].choices = [(semester, semester) for semester in SEMESTER_OPTIONS]

    def clean_student_id(self):
        value = " ".join(self.cleaned_data["student_id"].split())
        if not value:
            raise ValidationError("Student ID is required.")
        return value


class StudentResponseEditForm(StudentSubmissionForm):
    pass


class BatchUploadForm(forms.Form):
    file = forms.FileField(label="CSV or Excel file (.csv, .xlsx)")

    def clean_file(self):
        uploaded = self.cleaned_data["file"]
        name = uploaded.name.lower()
        if not (name.endswith(".csv") or name.endswith(".xlsx")):
            raise ValidationError("Upload a CSV or Excel (.xlsx) file.")
        if uploaded.size > 10 * 1024 * 1024:
            raise ValidationError("The upload must be 10 MB or smaller.")
        return uploaded
