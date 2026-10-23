from __future__ import annotations

import io
import json
import pandas as pd
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse

from src.bangladesh_collection import FIELD_TITLES, PREDICTOR_FIELDS, synthetic_example_rows
from src import prediction_services as prediction
from src.collection_core import SEMESTER_OPTIONS

from .models import Collection, EditHistory, StudentResponse, Teacher
from .services import create_collection


def sample_predictors():
    return synthetic_example_rows().iloc[0].loc[list(PREDICTOR_FIELDS)].to_dict()


def post_payload(*, student_id="CSE-1001", semester="Semester 1", **changes):
    values = {**sample_predictors(), "student_id": student_id, "semester": semester}
    values.update(changes)
    return values


class DjangoCollectionWorkflowTests(TestCase):
    def setUp(self):
        cache.clear()
        User = get_user_model()
        self.user = User.objects.create_user(username="teacher1", password="GoodPassword!246", email="teacher@example.edu")
        self.teacher = Teacher.objects.create(user=self.user)
        self.collection = create_collection(owner=self.teacher, title="Semester 1 Support",
                                            department="Computer Science", semesters=["Semester 1", "Semester 2"],
                                            closing_date=None)
        self.client = Client()

    def sign_in(self):
        assert self.client.login(username="teacher1", password="GoodPassword!246")

    def submit_student(self, **changes):
        return self.client.post(reverse("student_collect", args=(self.collection.collection_id,)),
                                post_payload(**changes))

    def test_student_get_is_public_and_uses_exact_shared_schema(self):
        response = self.client.get(reverse("student_collect", args=(self.collection.collection_id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Student Identification")
        self.assertContains(response, "Academic Background")
        self.assertContains(response, "Family Information")
        self.assertContains(response, "Academic Performance")
        self.assertContains(response, "Financial and Employment Information")
        self.assertEqual(len(PREDICTOR_FIELDS), 18)
        self.assertNotContains(response, "Current Academic Status")
        form = response.context["form"]
        actual_predictors = tuple(field for field in form.fields if field not in {"student_id", "semester"})
        self.assertEqual(actual_predictors, PREDICTOR_FIELDS)
        self.assertEqual(tuple(form.fields["semester"].choices), tuple((item, item) for item in SEMESTER_OPTIONS))

    def test_student_submission_persists_private_record_and_confirms_after_save(self):
        response = self.submit_student()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "saved successfully")
        saved = StudentResponse.objects.get(collection=self.collection)
        self.assertEqual(saved.student_id, "CSE-1001")
        self.assertEqual(saved.semester, "Semester 1")
        self.assertEqual(tuple(saved.predictors), PREDICTOR_FIELDS)
        self.assertEqual(len(saved.predictors), 18)
        self.assertNotIn("current_academic_status", saved.predictors)
        self.assertTrue(saved.record_id)
        self.assertTrue(saved.submitted_at)
        # A student success page contains no teacher-only response list or risk assessment.
        self.assertNotContains(response, "Risk Level")

    def test_duplicate_student_id_case_and_whitespace_normalized_within_collection(self):
        self.submit_student(student_id="CSE 1001")
        response = self.submit_student(student_id=" cse   1001 ", semester="Semester 2")
        self.assertContains(response, "already submitted")
        self.assertEqual(StudentResponse.objects.filter(collection=self.collection).count(), 1)

    def test_invalid_predictor_values_and_semester_are_rejected_without_persistence(self):
        response = self.submit_student(last_semester_attendance_percent=101)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "100")
        response = self.submit_student(semester="Semester 99")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(StudentResponse.objects.count(), 0)

    def test_teacher_login_session_and_unprovisioned_access_gate(self):
        response = self.client.get(reverse("workspace"))
        self.assertRedirects(response, f"{reverse('login')}?next={reverse('workspace')}")
        self.sign_in()
        self.assertEqual(self.client.get(reverse("workspace")).status_code, 200)
        self.client.post(reverse("logout"))
        self.assertFalse("_auth_user_id" in self.client.session)
        stranger = get_user_model().objects.create_user(username="stranger", password="GoodPassword!246")
        self.client.force_login(stranger)
        self.assertEqual(self.client.get(reverse("workspace")).status_code, 403)

    def test_collection_creation_ownership_and_open_close_actions(self):
        self.sign_in()
        response = self.client.post(reverse("collection_create"), {
            "title": "Year 2 support", "department": "CSE", "closing_date": "",
        })
        created = Collection.objects.get(title="Year 2 support")
        self.assertEqual(created.semesters, list(SEMESTER_OPTIONS))
        self.assertRedirects(response, reverse("collection_detail", args=(created.collection_id,)))
        detail = self.client.get(reverse("collection_detail", args=(created.collection_id,)))
        self.assertContains(detail, f"/collect/{created.collection_id}/")
        self.assertNotContains(detail, "Allowed semesters")
        student_form = self.client.get(reverse("student_collect", args=(created.collection_id,))).context["form"]
        self.assertEqual(tuple(student_form.fields["semester"].choices), tuple((item, item) for item in SEMESTER_OPTIONS))
        self.client.post(reverse("collection_status", args=(created.collection_id,)), {"status": "closed"})
        created.refresh_from_db()
        self.assertEqual(created.status, Collection.Status.CLOSED)
        stranger_user = get_user_model().objects.create_user(username="teacher2", password="GoodPassword!246")
        stranger_teacher = Teacher.objects.create(user=stranger_user)
        self.client.force_login(stranger_user)
        self.assertEqual(self.client.get(reverse("collection_detail", args=(created.collection_id,))).status_code, 404)
        self.assertEqual(self.client.get(reverse("export_collection_csv", args=(created.collection_id,))).status_code, 404)
        self.assertEqual(self.client.get(reverse("collection_analysis", args=(created.collection_id,))).status_code, 404)
        self.assertFalse(Collection.objects.filter(owner=stranger_teacher, collection_id=created.collection_id).exists())

    def test_teacher_edit_creates_audit_record_and_cannot_take_duplicate_id(self):
        self.submit_student()
        first = StudentResponse.objects.get(collection=self.collection)
        changed = {**first.predictors, "last_semester_attendance_percent": 63.5}
        self.sign_in()
        edit_url = reverse("response_edit", args=(self.collection.collection_id, first.record_id))
        response = self.client.post(edit_url, {**changed, "student_id": "CSE-1001", "semester": "Semester 2"})
        self.assertRedirects(response, reverse("collection_detail", args=(self.collection.collection_id,)))
        first.refresh_from_db()
        self.assertEqual(first.predictors["last_semester_attendance_percent"], 63.5)
        self.assertEqual(first.semester, "Semester 2")
        history = EditHistory.objects.get(response=first)
        self.assertEqual(history.previous_values["semester"], "Semester 1")
        self.assertEqual(history.new_values["semester"], "Semester 2")

        self.submit_student(student_id="CSE-2002")
        second = StudentResponse.objects.get(student_id="CSE-2002")
        response = self.client.post(reverse("response_edit", args=(self.collection.collection_id, second.record_id)),
                                    {**second.predictors, "student_id": "cse-1001", "semester": "Semester 1"})
        self.assertContains(response, "already uses that Student ID")
        self.assertEqual(EditHistory.objects.count(), 1)

    def test_exports_match_existing_batch_schema_and_excel_is_readable(self):
        self.submit_student()
        self.sign_in()
        csv_response = self.client.get(reverse("export_collection_csv", args=(self.collection.collection_id,)))
        csv_frame = pd.read_csv(io.BytesIO(csv_response.content))
        expected = ["Anonymous Follow-Up ID", "Current Semester / Academic Year", *[FIELD_TITLES[field] for field in PREDICTOR_FIELDS]]
        self.assertEqual(list(csv_frame.columns), expected)
        excel_response = self.client.get(reverse("export_collection_excel", args=(self.collection.collection_id,)))
        excel_frame = pd.read_excel(io.BytesIO(excel_response.content), engine="openpyxl")
        self.assertEqual(list(excel_frame.columns), expected)
        self.assertEqual(excel_frame.loc[0, "Anonymous Follow-Up ID"], "CSE-1001")

    def test_existing_model_prediction_analysis_and_downloads(self):
        self.submit_student()
        self.sign_in()
        response = self.client.get(reverse("collection_analysis", args=(self.collection.collection_id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Student results and support indicators")
        self.assertContains(response, "CSE-1001")
        self.assertEqual(self.client.get(reverse("analysis_csv", args=(self.collection.collection_id,))).status_code, 200)
        self.assertEqual(self.client.get(reverse("analysis_excel", args=(self.collection.collection_id,))).status_code, 200)

    def test_closed_collection_does_not_accept_students(self):
        self.collection.status = Collection.Status.CLOSED
        self.collection.save(update_fields=("status",))
        response = self.client.get(reverse("student_collect", args=(self.collection.collection_id,)))
        self.assertEqual(response.status_code, 410)
        self.assertContains(response, "closed", status_code=410)
        self.assertEqual(StudentResponse.objects.count(), 0)

    def test_public_student_posts_are_rate_limited_by_network_and_collection(self):
        url = reverse("student_collect", args=(self.collection.collection_id,))
        responses = [self.client.post(url, {}) for _ in range(31)]
        self.assertTrue(all(response.status_code == 200 for response in responses[:30]))
        self.assertEqual(responses[-1].status_code, 429)
        self.assertContains(responses[-1], "Too many submissions", status_code=429)
        self.assertEqual(StudentResponse.objects.count(), 0)


class PublicProjectWebsiteTests(TestCase):
    def setUp(self):
        self.predictors = synthetic_example_rows().iloc[0].loc[list(PREDICTOR_FIELDS)].to_dict()
        self.client.post(reverse("continue_as_guest"))

    def sign_in_teacher(self):
        User = get_user_model()
        user = User.objects.create_user(username="batch-teacher", email="batch@example.edu",
                                        password="BatchPass123")
        Teacher.objects.create(user=user, role=Teacher.Role.TEACHER)
        response = self.client.post(reverse("login"), {"username": user.username, "password": "BatchPass123"})
        self.assertEqual(response.status_code, 302)

    def test_dashboard_and_about_present_saved_project_information(self):
        dashboard = self.client.get(reverse("home"))
        self.assertEqual(dashboard.status_code, 200)
        self.assertContains(dashboard, "Student Dropout Early Warning System")
        self.assertContains(dashboard, "Assess One Student")
        about = self.client.get(reverse("about_project"))
        self.assertEqual(about.status_code, 200)
        self.assertContains(about, "Five-fold nested cross-validation")
        self.assertContains(about, "193")
        self.assertContains(about, "251")

    def test_dashboard_and_about_percentages_match_active_saved_metadata(self):
        metadata_path = prediction.ROOT / "models" / "combined_uci_bangladesh_18_v2" / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata["nested_cv_comparison"]["Fixed 5%/95% blend (10-feature UCI)"]["metrics"]

        dashboard = self.client.get(reverse("dashboard"))
        self.assertEqual(dashboard.status_code, 200)
        dashboard_metrics = {item["label"]: item["value"] for item in dashboard.context["performance_metrics"]}
        for key, label in (("accuracy", "Accuracy"), ("precision", "Precision"),
                           ("recall", "Recall"), ("f1", "F1-score")):
            formatted = f"{expected[key]:.1%}"
            self.assertEqual(dashboard_metrics[label], formatted)
            self.assertContains(dashboard, formatted)
        self.assertEqual(dashboard_metrics["Accuracy"], "96.4%")
        self.assertContains(dashboard, "nested CV on Bangladesh records")
        self.assertContains(dashboard, "reported current dropout-status patterns")
        self.assertContains(dashboard, "not been prospectively validated")
        self.assertContains(dashboard, "Logistic Regression Ensemble")
        self.assertContains(dashboard, "18")
        self.assertContains(dashboard, "3,630")
        self.assertContains(dashboard, "251")

        about = self.client.get(reverse("about_project"))
        self.assertEqual(about.status_code, 200)
        self.assertEqual(about.context["metrics"], expected)
        self.assertContains(about, f"{expected['roc_auc']:.1%}")
        self.assertContains(about, f"{expected['pr_auc_average_precision']:.1%}")
        self.assertContains(about, f"{expected['brier_score']:.3f}")
        self.assertContains(about, "Five-fold nested cross-validation · Bangladesh records")
        self.assertContains(about, "not pooled as one training set")
        self.assertContains(about, "98.6%")
        self.assertContains(about, "93.5%")
        self.assertEqual(about.context["threshold_percent"], metadata["threshold"] * 100)
        self.assertContains(about, f"{metadata['threshold'] * 100:.1f}%")

    def test_individual_assessment_uses_exact_18_inputs_and_saved_ensemble(self):
        response = self.client.get(reverse("individual_assessment"))
        form = response.context["form"]
        self.assertEqual(len([field for field in form.fields if field != "reference_id"]), 18)
        self.assertNotIn("current_academic_status", form.fields)
        self.assertContains(response, "Not Alive")
        self.assertContains(response, "Housewife")

        expected = prediction.predict_student(prediction.load_model_artifact(), self.predictors)
        posted = {**self.predictors, "reference_id": "DEMO-001"}
        result = self.client.post(reverse("individual_assessment"), posted)
        self.assertEqual(result.status_code, 200)
        self.assertContains(result, f"{expected['probability'] * 100:.1f}%")
        self.assertContains(result, f"{expected['component_scores']['uci_academic'] * 100:.1f}%")
        self.assertContains(result, f"{expected['component_scores']['bangladesh_local'] * 100:.1f}%")
        self.assertContains(result, "threshold 30.4%")
        self.assertContains(result, expected["risk_level"])
        self.assertContains(result, "Model details")

        report = self.client.get(reverse("assessment_pdf_download"))
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report["Content-Type"], "application/pdf")
        self.assertTrue(report.content.startswith(b"%PDF-1.4"))

    def _post_batch(self, filename, content, content_type):
        upload = SimpleUploadedFile(filename, content, content_type=content_type)
        return self.client.post(reverse("batch_analysis"), {"file": upload})

    def test_batch_csv_upload_templates_and_exports(self):
        self.sign_in_teacher()
        frame = prediction.template_frame()
        csv_content = frame.to_csv(index=False).encode("utf-8")
        response = self._post_batch("sample.csv", csv_content, "text/csv")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Total students")
        self.assertContains(response, "EXAMPLE-001")
        expected_batch = prediction.analyze_batch(prediction.load_model_artifact(), frame)
        expected_probability = float(expected_batch.iloc[0]["Estimated Current Dropout-Status Probability"])
        self.assertContains(response, f"{expected_probability * 100:.1f}%")
        self.assertEqual(response.context["average_risk"], expected_probability * 100)
        self.assertEqual(self.client.get(reverse("batch_template_download", args=("csv",))).status_code, 200)
        self.assertEqual(self.client.get(reverse("batch_template_download", args=("xlsx",))).status_code, 200)
        csv_export = self.client.get(reverse("batch_result_download", args=("csv",)))
        self.assertEqual(csv_export.status_code, 200)
        result = pd.read_csv(io.BytesIO(csv_export.content))
        self.assertIn("Risk Level", result.columns)
        excel_export = self.client.get(reverse("batch_result_download", args=("xlsx",)))
        self.assertEqual(excel_export.status_code, 200)
        excel_result = pd.read_excel(io.BytesIO(excel_export.content), engine="openpyxl")
        self.assertIn("Suggested Institute Support", excel_result.columns)

    def test_batch_excel_upload_is_supported_and_validation_is_reported(self):
        self.sign_in_teacher()
        excel_content = prediction.dataframe_to_excel(prediction.template_frame(), "Students")
        response = self._post_batch("sample.xlsx", excel_content,
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Total students")

        invalid = prediction.template_frame()
        invalid.loc[0, "Last-semester attendance percentage"] = 101
        bad_response = self._post_batch("invalid.csv", invalid.to_csv(index=False).encode("utf-8"), "text/csv")
        self.assertContains(bad_response, "range 0–100")
