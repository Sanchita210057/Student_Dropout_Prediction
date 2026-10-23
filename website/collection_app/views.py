from __future__ import annotations

import json
import hashlib
from io import StringIO
from uuid import UUID

import pandas as pd
from django.contrib import messages
from django.core.cache import cache
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST
from django.views.decorators.http import require_GET

from src.bangladesh_collection import FIELD_TITLES, PREDICTOR_FIELDS
from src.collection_core import DuplicateStudentIDError, WorkspaceError
from src import prediction_services as prediction

from .forms import (BatchUploadForm, CollectionCreateForm, IndividualAssessmentForm,
                    StudentResponseEditForm, StudentSubmissionForm)
from .access import institute_profile, institute_user_required, public_site_required
from .models import Collection, StudentResponse, Teacher
from .services import (create_collection as create_collection_record, edit_response,
                       list_collection_records, set_collection_status, submit_response)

FORM_SECTIONS = (
    ("Academic Background", ("hsc_equivalent_gpa", "age_at_enrollment")),
    ("Family Information", ("mother_occupation_bangladesh", "father_occupation_bangladesh",
                             "mother_education_bangladesh", "father_education_bangladesh",
                             "gender_bangladesh", "marital_status_bangladesh")),
    ("Academic Performance", ("courses_enrolled", "evaluations_completed", "courses_passed",
                              "semester_gpa_native", "last_semester_attendance_percent")),
    ("Financial and Employment Information", ("scholarship_holder", "debtor",
                                               "tuition_fees_up_to_date", "family_financial_condition",
                                               "student_employment")),
)
MAX_BATCH_ROWS = 500
MAX_BATCH_BYTES = 10 * 1024 * 1024


teacher_required = institute_user_required


def _owned_collection(request, collection_id: str) -> Collection:
    query = Collection.objects.filter(collection_id=collection_id)
    if request.teacher.role != Teacher.Role.ADMIN:
        query = query.filter(owner=request.teacher)
    return get_object_or_404(query)


def _form_sections(form):
    return [(heading, [form[field] for field in fields]) for heading, fields in FORM_SECTIONS]


def home(request):
    metrics = {}
    training_counts = {}
    try:
        with (prediction.ROOT / "models" / "combined_uci_bangladesh_18_v2" / "model_metadata.json").open(encoding="utf-8") as source:
            metadata = json.load(source)
        metrics = metadata["nested_cv_comparison"]["Fixed 5%/95% blend (10-feature UCI)"]["metrics"]
        training_counts = metadata["training_counts"]
    except (OSError, KeyError, ValueError, TypeError):
        metadata = {}
    dataset_counts = {
        "uci": f"{int(training_counts.get('uci_resolved_records', 3630)):,}",
        "bangladesh": f"{int(training_counts.get('bangladesh_records', 251)):,}",
    }
    return render(request, "collection_app/home.html", {
        "metrics": metrics, "training_counts": training_counts, "dataset_counts": dataset_counts,
        "performance_metrics": ([
            {"label": label, "value": f"{float(metrics[key]):.1%}"}
            for key, label in (("accuracy", "Accuracy"), ("precision", "Precision"),
                               ("recall", "Recall"), ("f1", "F1-score"))
        ] if metrics else []),
        "model_available": prediction.MODEL_PATH.is_file(),
    })


@public_site_required
def dashboard(request):
    return home(request)


def _assessment_sections(form):
    fields = {
        "Academic Background": ("hsc_equivalent_gpa", "age_at_enrollment"),
        "Family Information": ("mother_occupation_bangladesh", "father_occupation_bangladesh",
                               "mother_education_bangladesh", "father_education_bangladesh",
                               "gender_bangladesh", "marital_status_bangladesh"),
        "Academic Performance": ("courses_enrolled", "evaluations_completed", "courses_passed",
                                 "semester_gpa_native", "last_semester_attendance_percent"),
        "Financial and Employment Information": ("scholarship_holder", "debtor",
                                                   "tuition_fees_up_to_date", "family_financial_condition",
                                                   "student_employment"),
    }
    return [(title, [form[field] for field in names]) for title, names in fields.items()]


@public_site_required
@require_http_methods(["GET", "POST"])
def individual_assessment(request):
    form = IndividualAssessmentForm(request.POST or None)
    result = None
    reference_id = ""
    if request.method == "POST" and form.is_valid():
        values = {field: form.cleaned_data[field] for field in PREDICTOR_FIELDS}
        reference_id = form.cleaned_data.get("reference_id") or "Individual Assessment"
        try:
            result = prediction.predict_student(prediction.load_model_artifact(), values)
        except (RuntimeError, ValueError, OSError) as error:
            form.add_error(None, f"Assessment is unavailable: {error}")
        else:
            result["probability_percent"] = result["probability"] * 100
            result["threshold_percent"] = result["threshold"] * 100
            result["uci_percent"] = result["component_scores"]["uci_academic"] * 100
            result["local_percent"] = result["component_scores"]["bangladesh_local"] * 100
            request.session["last_assessment_report"] = {"result": result, "reference_id": reference_id}
            request.session.set_expiry(3600)
    return render(request, "collection_app/assessment.html", {
        "form": form, "form_sections": _assessment_sections(form),
        "result": result, "reference_id": reference_id,
    })


@public_site_required
@require_GET
def assessment_pdf_download(request):
    saved = request.session.get("last_assessment_report")
    if not saved:
        raise Http404("Run an individual assessment before downloading its report.")
    response = HttpResponse(prediction.assessment_pdf(saved["result"], saved["reference_id"]), content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="student_risk_assessment.pdf"'
    return response


@institute_user_required
@require_http_methods(["GET", "POST"])
def batch_analysis(request):
    form = BatchUploadForm(request.POST or None, request.FILES or None)
    user_profile = institute_profile(request.user) if request.user.is_authenticated else None
    results = None
    validation_issues = []
    upload_name = ""
    if request.method == "POST" and form.is_valid():
        uploaded = form.cleaned_data["file"]
        upload_name = uploaded.name
        try:
            frame = prediction.read_uploaded_table(uploaded, upload_name)
            if len(frame) > MAX_BATCH_ROWS:
                raise ValueError(f"A batch may contain at most {MAX_BATCH_ROWS} student rows.")
            errors = prediction.validate_batch_dataframe(frame)
            if not errors.empty:
                validation_issues = errors.to_dict(orient="records")
            else:
                results = prediction.analyze_batch(prediction.load_model_artifact(), frame)
                request.session["last_batch_results"] = results.to_json(orient="split", date_format="iso")
                request.session["last_batch_filename"] = upload_name.rsplit(".", 1)[0]
                request.session.set_expiry(3600)
        except (RuntimeError, ValueError, OSError, ImportError) as error:
            form.add_error("file", str(error))
    elif request.method == "POST":
        pass
    if results is None and request.session.get("last_batch_results") and request.method == "GET":
        try:
            results = pd.read_json(StringIO(request.session["last_batch_results"]), orient="split")
        except (ValueError, TypeError):
            request.session.pop("last_batch_results", None)
    context = {
        "form": form, "results": results,
        "upload_name": upload_name or request.session.get("last_batch_filename", ""),
        "result_rows": ([{
            "student_id": row.get(prediction.ID_COLUMN, ""),
            "semester": row.get(prediction.SEMESTER_COLUMN, ""),
            "probability_percent": float(row.get("Estimated Current Dropout-Status Probability", 0)) * 100,
            "risk_level": row.get("Risk Level", ""),
            "indicators": row.get("Key Risk Indicators", ""),
            "support": row.get("Suggested Institute Support", ""),
        } for row in results.to_dict(orient="records")] if results is not None else []),
        "total": len(results) if results is not None else 0,
        "elevated": int((results["Risk Level"].str.startswith("Elevated")).sum()) if results is not None and not results.empty else 0,
        "lower": int((results["Risk Level"].str.startswith("Lower")).sum()) if results is not None and not results.empty else 0,
        "average_risk": float(results["Estimated Current Dropout-Status Probability"].mean()) * 100 if results is not None and not results.empty else 0,
        "validation_issues": validation_issues,
        "download_ready": bool(request.session.get("last_batch_results")),
        "show_collection_tools": bool(user_profile and request.user.is_active),
    }
    return render(request, "collection_app/batch.html", context)


@institute_user_required
@require_GET
def batch_template_download(request, file_type):
    if file_type == "csv":
        response = HttpResponse(prediction.template_frame().to_csv(index=False), content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = 'attachment; filename="student_batch_template.csv"'
        return response
    if file_type == "xlsx":
        response = HttpResponse(prediction.template_excel(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        response["Content-Disposition"] = 'attachment; filename="student_batch_template.xlsx"'
        return response
    raise Http404("Unknown template format.")


@institute_user_required
@require_GET
def batch_result_download(request, file_type):
    serialized = request.session.get("last_batch_results")
    if not serialized:
        raise Http404("Analyze a batch before downloading results.")
    frame = pd.read_json(StringIO(serialized), orient="split")
    if file_type == "csv":
        response = HttpResponse(prediction.export_results(frame), content_type="text/csv; charset=utf-8")
        filename = "student_batch_results.csv"
    elif file_type == "xlsx":
        response = HttpResponse(prediction.export_results_excel(frame), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        filename = "student_batch_results.xlsx"
    else:
        raise Http404("Unknown export format.")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@require_GET
@public_site_required
def about_project(request):
    try:
        with (prediction.ROOT / "models" / "combined_uci_bangladesh_18_v2" / "model_metadata.json").open(encoding="utf-8") as source:
            metadata = json.load(source)
        metrics = metadata["nested_cv_comparison"]["Fixed 5%/95% blend (10-feature UCI)"]["metrics"]
        confusion = metrics["confusion_matrix"]
        counts = metadata["training_counts"]
        threshold = metadata["threshold"]
    except (OSError, KeyError, ValueError, TypeError):
        metrics, confusion, counts, threshold = {}, {}, {}, None
    dataset_counts = {
        "uci": f"{int(counts.get('uci_resolved_records', 3630)):,}",
        "bangladesh": f"{int(counts.get('bangladesh_records', 251)):,}",
    }
    return render(request, "collection_app/about.html", {
        "metrics": metrics, "confusion": confusion, "training_counts": counts,
        "dataset_counts": dataset_counts,
        "threshold": threshold, "threshold_percent": threshold * 100 if threshold is not None else None,
        "performance_bars": ([
            {"label": label, "percent": float(metrics[key]) * 100,
             "value": f"{float(metrics[key]):.1%}"}
            for key, label in (("accuracy", "Accuracy"), ("precision", "Precision"),
                               ("recall", "Recall"), ("f1", "F1-score"))
        ] if metrics else []),
        "supporting_metrics": ([
            {"label": "ROC-AUC", "value": f"{float(metrics['roc_auc']):.1%}"},
            {"label": "PR-AUC", "value": f"{float(metrics['pr_auc_average_precision']):.1%}"},
            {"label": "Brier score", "value": f"{float(metrics['brier_score']):.3f}"},
        ] if metrics else []),
        "model_available": prediction.MODEL_PATH.is_file(),
    })


@teacher_required
def workspace(request):
    collections = Collection.objects.all() if request.teacher.role == Teacher.Role.ADMIN else Collection.objects.filter(owner=request.teacher)
    collections = collections.order_by("-created_at")
    return render(request, "collection_app/workspace.html", {"collections": collections})


@teacher_required
@require_http_methods(["GET", "POST"])
def collection_create(request):
    form = CollectionCreateForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        collection = create_collection_record(
            owner=request.teacher, title=form.cleaned_data["title"],
            department=form.cleaned_data["department"],
            closing_date=form.cleaned_data.get("closing_date"),
        )
        messages.success(request, "Collection created. Share the student link below.")
        return redirect("collection_detail", collection_id=collection.collection_id)
    return render(request, "collection_app/collection_form.html", {"form": form})


@teacher_required
def collection_detail(request, collection_id: str):
    collection = _owned_collection(request, collection_id)
    records = list_collection_records(collection=collection)
    display_records = [{
        "student_id": row["student_id"], "semester": row["semester"],
        "submitted_at": row["submitted_at"], "record_id": row["record_id"],
        "predictor_values": [row[field] for field in PREDICTOR_FIELDS],
    } for row in records]
    return render(request, "collection_app/collection_detail.html", {
        "collection": collection,
        "student_url": request.build_absolute_uri(reverse("student_collect", args=(collection.collection_id,))),
        "records": display_records,
        "predictors": [(field, FIELD_TITLES[field]) for field in PREDICTOR_FIELDS],
    })


@teacher_required
@require_POST
def collection_status(request, collection_id: str):
    collection = _owned_collection(request, collection_id)
    status = request.POST.get("status", "")
    try:
        set_collection_status(collection=collection, status=status)
    except ValueError as error:
        messages.error(request, str(error))
    else:
        messages.success(request, f"Collection is now {status}.")
    return redirect("collection_detail", collection_id=collection.collection_id)


@teacher_required
@require_http_methods(["GET", "POST"])
def response_edit(request, collection_id: str, record_id: UUID):
    collection = _owned_collection(request, collection_id)
    response = get_object_or_404(StudentResponse, collection=collection, record_id=record_id)
    initial = {"student_id": response.student_id, "semester": response.semester, **response.predictors}
    form = StudentResponseEditForm(request.POST or None, initial=initial)
    if request.method == "POST" and form.is_valid():
        values = {field: form.cleaned_data[field] for field in PREDICTOR_FIELDS}
        try:
            edit_response(collection_id=collection.collection_id, record_id=record_id,
                          teacher=request.teacher, student_id=form.cleaned_data["student_id"],
                          semester=form.cleaned_data["semester"], predictors=values)
        except DuplicateStudentIDError as error:
            form.add_error("student_id", str(error))
        except WorkspaceError as error:
            form.add_error(None, str(error))
        else:
            messages.success(request, "Corrected record saved. An edit history entry was recorded.")
            return redirect("collection_detail", collection_id=collection.collection_id)
    return render(request, "collection_app/student_form.html", {
        "form": form, "collection": collection, "form_sections": _form_sections(form),
        "editing": True,
    })


@require_http_methods(["GET", "POST"])
def student_collect(request, collection_id: str):
    collection = get_object_or_404(Collection, collection_id=collection_id)
    if collection.status != Collection.Status.OPEN:
        return render(request, "collection_app/collection_closed.html", {"collection": collection}, status=410)
    if collection.closing_date and collection.closing_date < timezone.localdate():
        return render(request, "collection_app/collection_closed.html", {"collection": collection}, status=410)
    form = StudentSubmissionForm(request.POST or None)
    if request.method == "POST":
        address = request.META.get("REMOTE_ADDR", "unknown")
        address_key = hashlib.sha256(address.encode("utf-8")).hexdigest()[:24]
        minute_key = f"student-collect-minute:{collection.pk}:{address_key}:{timezone.now():%Y%m%d%H%M}"
        hour_key = f"student-collect-hour:{collection.pk}:{address_key}:{timezone.now():%Y%m%d%H}"
        for key, limit, timeout in ((minute_key, 30, 120), (hour_key, 1000, 3700)):
            cache.add(key, 0, timeout=timeout)
            count = cache.incr(key)
            if count > limit:
                return render(request, "collection_app/collection_closed.html", {
                    "collection": collection, "rate_limited": True,
                }, status=429)
    if request.method == "POST" and form.is_valid():
        values = {field: form.cleaned_data[field] for field in PREDICTOR_FIELDS}
        try:
            submit_response(collection_id=collection.collection_id,
                            student_id=form.cleaned_data["student_id"],
                            semester=form.cleaned_data["semester"], predictors=values)
        except DuplicateStudentIDError as error:
            form.add_error("student_id", str(error))
        except Collection.DoesNotExist:
            raise Http404("This collection link is no longer available.")
        except WorkspaceError as error:
            form.add_error(None, str(error))
        else:
            return render(request, "collection_app/submission_success.html", {"collection": collection})
    return render(request, "collection_app/student_form.html", {
        "form": form, "collection": collection, "form_sections": _form_sections(form),
        "editing": False,
    })


def _batch_frame(collection: Collection) -> pd.DataFrame:
    from src.collection_core import records_to_batch_frame
    return records_to_batch_frame(list_collection_records(collection=collection))


def _analysis(collection: Collection) -> pd.DataFrame:
    if not collection.responses.exists():
        return pd.DataFrame()
    return prediction.analyze_batch(prediction.load_model_artifact(), _batch_frame(collection))


@teacher_required
def export_collection_csv(request, collection_id: str):
    collection = _owned_collection(request, collection_id)
    frame = _batch_frame(collection)
    response = HttpResponse(frame.to_csv(index=False), content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="collection_{collection.collection_id}_records.csv"'
    return response


@teacher_required
def export_collection_excel(request, collection_id: str):
    collection = _owned_collection(request, collection_id)
    response = HttpResponse(prediction.dataframe_to_excel(_batch_frame(collection), "Student Records"),
                            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = f'attachment; filename="collection_{collection.collection_id}_records.xlsx"'
    return response


@teacher_required
def collection_analysis(request, collection_id: str):
    collection = _owned_collection(request, collection_id)
    try:
        results = _analysis(collection)
    except Exception as error:
        messages.error(request, f"Analysis could not be completed: {error}")
        results = pd.DataFrame()
    rows = [{
        "student_id": row.get("Anonymous Follow-Up ID", ""),
        "semester": row.get("Current Semester / Academic Year", ""),
        "probability_percent": float(row.get("Estimated Current Dropout-Status Probability", 0)) * 100,
        "risk_level": row.get("Risk Level", ""),
        "indicators": row.get("Key Risk Indicators", ""),
        "support": row.get("Suggested Institute Support", ""),
    } for row in results.to_dict(orient="records")] if not results.empty else []
    counts = results["Risk Level"].value_counts().to_dict() if not results.empty else {}
    return render(request, "collection_app/analysis.html", {
        "collection": collection, "rows": rows, "counts": counts,
        "analysis_available": not results.empty,
    })


@teacher_required
def analysis_download(request, collection_id: str, file_type: str):
    collection = _owned_collection(request, collection_id)
    try:
        results = _analysis(collection)
    except Exception as error:
        messages.error(request, f"Analysis could not be completed: {error}")
        return redirect("collection_detail", collection_id=collection.collection_id)
    if results.empty:
        messages.info(request, "There are no student responses to analyze yet.")
        return redirect("collection_detail", collection_id=collection.collection_id)
    if file_type == "csv":
        response = HttpResponse(prediction.export_results(results), content_type="text/csv; charset=utf-8")
        filename = "collection_analysis.csv"
    else:
        response = HttpResponse(prediction.export_results_excel(results), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        filename = "collection_analysis.xlsx"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response
