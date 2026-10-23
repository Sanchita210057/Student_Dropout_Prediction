from __future__ import annotations

from django.contrib.auth import get_user_model
from django.contrib.staticfiles import finders
from django.test import TestCase
from django.urls import reverse

from src.bangladesh_collection import PREDICTOR_FIELDS, synthetic_example_rows

from .models import Collection, StudentResponse, Teacher
from .services import create_collection, submit_response

User = get_user_model()


def make_teacher(username="staff1", role=Teacher.Role.TEACHER, password="Initial#Pass-824"):
    user = User.objects.create_user(username=username, email=f"{username}@example.edu", password=password)
    profile = Teacher.objects.create(user=user, role=role)
    return user, profile


def predictor_values():
    return synthetic_example_rows().iloc[0].loc[list(PREDICTOR_FIELDS)].to_dict()


class AccountWorkflowTests(TestCase):
    strong_password = "Temporary#Access-938"

    def setUp(self):
        self.admin = User.objects.create_superuser(
            username="instituteadmin", email="admin@example.edu", password="Admin#Password-284"
        )
        self.admin_profile = self.admin.teacher_profile
        self.client = self.client_class()

    def admin_login(self):
        response = self.client.post(reverse("login"), {
            "username": self.admin.email, "password": "Admin#Password-284",
        })
        self.assertRedirects(response, reverse("dashboard"))

    def create_account(self, username="newteacher", role=Teacher.Role.TEACHER, **overrides):
        payload = {
            "full_name": "Taylor Instructor",
            "username": username,
            "email": f"{username}@example.edu",
            "role": role,
            "temporary_password": self.strong_password,
            "confirm_password": self.strong_password,
        }
        payload.update(overrides)
        return self.client.post(reverse("user_create"), payload)

    def test_createsuperuser_is_bootstrapped_as_separate_institute_admin_role(self):
        self.assertTrue(self.admin.is_superuser)
        self.assertTrue(self.admin.is_staff)
        self.assertEqual(self.admin_profile.role, Teacher.Role.ADMIN)
        self.assertEqual(self.client.get(reverse("user_management")).status_code, 302)
        self.admin_login()
        self.assertEqual(self.client.get(reverse("user_management")).status_code, 200)
        self.assertContains(self.client.get(reverse("dashboard")), "Admin Management")

    def test_root_is_direct_login_and_logo_has_missing_file_fallback(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Student Dropout Early Warning System")
        self.assertContains(response, "Username or Email")
        self.assertContains(response, "Continue as Guest")
        self.assertContains(response, "Sign In", count=1)
        self.assertContains(response, "@media(max-width:640px)")
        self.assertContains(response, "collection_app/images/university_logo.png")
        self.assertContains(response, "login-logo-fallback")
        self.assertContains(response, "onerror=")
        self.assertEqual(self.client.get(reverse("login")).status_code, 200)
        self.assertTrue(finders.find("collection_app/images/university_logo.png"))
        self.assertNotContains(response, "For institute users")
        self.assertNotContains(response, "For guests")
        self.assertNotContains(response, "Batch Analysis")
        self.assertEqual(self.client.get("/accounts/register/").status_code, 404)

    def test_guest_mode_allows_only_dashboard_assessment_and_about(self):
        denied_dashboard = self.client.get(reverse("dashboard"))
        self.assertEqual(denied_dashboard.status_code, 302)
        self.assertIn(reverse("login"), denied_dashboard["Location"])
        guest_start = self.client.post(reverse("continue_as_guest"))
        self.assertRedirects(guest_start, reverse("dashboard"))
        self.assertTrue(self.client.session.get("guest_mode"))
        self.assertContains(self.client.get(reverse("login")), "Sign In", count=1)

        for route in ("dashboard", "individual_assessment", "about_project"):
            self.assertEqual(self.client.get(reverse(route)).status_code, 200)
        dashboard = self.client.get(reverse("dashboard"))
        self.assertNotContains(dashboard, "Batch Analysis")
        self.assertNotContains(dashboard, "Analyze Student Batch")
        self.assertContains(dashboard, "Assess One Student")

        protected = (
            reverse("batch_analysis"),
            reverse("batch_template_download", args=("csv",)),
            reverse("batch_template_download", args=("xlsx",)),
            reverse("batch_result_download", args=("csv",)),
            reverse("batch_result_download", args=("xlsx",)),
            reverse("workspace"),
            reverse("user_management"),
            reverse("account_settings"),
        )
        owner, owner_profile = make_teacher("guest-private-owner")
        collection = create_collection(owner=owner_profile, title="Private guest test", department="CSE",
                                       semesters=["Semester 1"], closing_date=None)
        response = submit_response(collection_id=collection.collection_id, student_id="PRIVATE-1",
                                   semester="Semester 1", predictors=predictor_values())
        protected += (
            reverse("collection_create"),
            reverse("collection_detail", args=(collection.collection_id,)),
            reverse("export_collection_csv", args=(collection.collection_id,)),
            reverse("export_collection_excel", args=(collection.collection_id,)),
            reverse("collection_analysis", args=(collection.collection_id,)),
            reverse("analysis_csv", args=(collection.collection_id,)),
            reverse("analysis_excel", args=(collection.collection_id,)),
            reverse("response_edit", args=(collection.collection_id, response.record_id)),
        )
        for url in protected:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302, url)
            self.assertIn(reverse("login"), response["Location"])
        self.assertContains(self.client.get(reverse("login")), "Sign in with an institute account")

    def test_guest_can_make_an_individual_assessment_without_persisting_records(self):
        self.client.post(reverse("continue_as_guest"))
        values = synthetic_example_rows().iloc[0].loc[list(PREDICTOR_FIELDS)].to_dict()
        response = self.client.post(reverse("individual_assessment"), values)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "FINAL ASSESSMENT")
        self.assertEqual(StudentResponse.objects.count(), 0)

    def test_admin_creates_teacher_or_staff_with_simple_hashed_password_and_immediate_login(self):
        self.admin_login()
        response = self.create_account(role=Teacher.Role.STAFF, temporary_password="1234", confirm_password="1234")
        self.assertRedirects(response, reverse("user_management"))
        user = User.objects.get(username="newteacher")
        profile = user.teacher_profile
        self.assertTrue(user.is_active)
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)
        self.assertEqual(profile.role, Teacher.Role.STAFF)
        self.assertNotEqual(user.password, "1234")
        self.client.post(reverse("logout"))
        login_response = self.client.post(reverse("login"), {
            "username": user.email, "password": "1234",
        })
        self.assertRedirects(login_response, reverse("dashboard"))
        self.assertEqual(self.client.get(reverse("workspace")).status_code, 200)

    def test_admin_creation_rejects_duplicate_username_and_email_and_admin_role(self):
        self.admin_login()
        make_teacher(username="taken")
        duplicate_username = self.create_account(username="taken")
        self.assertContains(duplicate_username, "already in use")
        duplicate_email = self.create_account(username="other", email="TAKEN@example.edu")
        self.assertContains(duplicate_email, "already in use")
        forbidden_role = self.create_account(username="elevated", role=Teacher.Role.ADMIN)
        self.assertContains(forbidden_role, "Select a valid choice")
        self.assertFalse(User.objects.filter(username__in=("other", "elevated")).exists())

    def test_account_passwords_must_be_nonempty_and_confirmed(self):
        self.admin_login()
        empty = self.create_account(username="empty-pass", temporary_password="", confirm_password="")
        self.assertContains(empty, "This field is required")
        mismatch = self.create_account(username="mismatch-pass", temporary_password="abc", confirm_password="xyz")
        self.assertContains(mismatch, "passwords do not match")
        self.assertFalse(User.objects.filter(username__in=("empty-pass", "mismatch-pass")).exists())

    def test_admin_can_edit_managed_user_without_elevating_django_privileges(self):
        self.admin_login()
        self.create_account(username="editable-user")
        user = User.objects.get(username="editable-user")
        response = self.client.post(reverse("user_edit", args=(user.pk,)), {
            "full_name": "Alex Edited", "username": "edited-user", "email": "edited@example.edu",
            "role": Teacher.Role.STAFF,
        })
        self.assertRedirects(response, reverse("user_management"))
        user.refresh_from_db()
        self.assertEqual(user.get_full_name(), "Alex Edited")
        self.assertEqual(user.email, "edited@example.edu")
        self.assertEqual(user.teacher_profile.role, Teacher.Role.STAFF)
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)

    def test_teacher_and_staff_can_sign_in_with_username_or_email_but_not_manage_users(self):
        teacher, _ = make_teacher("teacher-login", Teacher.Role.TEACHER)
        self.client.post(reverse("continue_as_guest"))
        self.assertEqual(self.client.post(reverse("login"), {
            "username": teacher.username, "password": "Initial#Pass-824",
        }).status_code, 302)
        self.assertFalse(self.client.session.get("guest_mode", False))
        authenticated_dashboard = self.client.get(reverse("dashboard"))
        self.assertContains(authenticated_dashboard, "Batch Analysis")
        self.assertContains(authenticated_dashboard, "My Account")
        self.assertContains(authenticated_dashboard, "Logout")
        self.assertRedirects(self.client.get("/"), reverse("dashboard"))
        self.client.post(reverse("logout"))
        self.assertEqual(self.client.post(reverse("login"), {
            "username": teacher.email, "password": "Initial#Pass-824",
        }).status_code, 302)
        self.assertEqual(self.client.get(reverse("user_management")).status_code, 403)
        staff, profile = make_teacher("staff-login", Teacher.Role.STAFF)
        self.client.post(reverse("logout"))
        staff_login = self.client.post(reverse("login"), {
            "username": staff.email, "password": "Initial#Pass-824",
        })
        self.assertRedirects(staff_login, reverse("dashboard"))
        self.assertEqual(self.client.get(reverse("workspace")).status_code, 200)
        self.assertNotContains(self.client.get(reverse("dashboard")), "Admin Management")
        self.assertEqual(self.client.get(reverse("user_management")).status_code, 403)
        self.assertEqual(profile.role, Teacher.Role.STAFF)

    def test_repeated_failed_sign_ins_are_throttled(self):
        responses = [self.client.post(reverse("login"), {
            "username": "locked-login@example.edu", "password": "incorrect",
        }) for _ in range(9)]
        self.assertEqual(responses[-1].status_code, 200)
        self.assertContains(responses[-1], "Too many unsuccessful sign-in attempts")

    def test_my_account_only_updates_username_and_email_and_password_change_is_blocked(self):
        user, _ = make_teacher("account-owner")
        self.client.force_login(user)
        denied = self.client.post(reverse("account_update"), {
            "action": "password", "current_password": "wrong", "new_password1": "Next#Password-184",
            "new_password2": "Next#Password-184",
        })
        self.assertEqual(denied.status_code, 403)
        self.assertTrue(user.check_password("Initial#Pass-824"))

        self.client.post(reverse("account_update"), {
            "action": "username", "username": "renamed-user",
        })
        self.client.post(reverse("account_update"), {
            "action": "email", "new_email": "new-address@example.edu",
        })
        user.refresh_from_db()
        self.assertEqual(user.username, "renamed-user")
        self.assertEqual(user.email, "new-address@example.edu")
        self.assertTrue(user.check_password("Initial#Pass-824"))

    def test_admin_can_reset_password_deactivate_reactivate_and_login_immediately(self):
        self.admin_login()
        self.create_account(username="reset-target")
        user = User.objects.get(username="reset-target")
        url = reverse("user_password_reset", args=(user.pk,))
        response = self.client.post(url, {
            "temporary_password": "42",
            "confirm_password": "42",
        })
        self.assertRedirects(response, reverse("user_management"))
        user.refresh_from_db()
        self.assertTrue(user.check_password("42"))
        self.client.post(reverse("user_status", args=(user.pk,)), {"active": "0"})
        user.refresh_from_db()
        self.assertFalse(user.is_active)
        self.client.post(reverse("user_status", args=(user.pk,)), {"active": "1"})
        user.refresh_from_db()
        self.assertTrue(user.is_active)
        self.client.post(reverse("logout"))
        self.assertRedirects(self.client.post(reverse("login"), {
            "username": user.username, "password": "42",
        }), reverse("dashboard"))

    def test_account_and_admin_management_ui_excludes_forced_password_change_controls(self):
        self.admin_login()
        account = self.client.get(reverse("account_settings"))
        self.assertContains(account, "Username")
        self.assertContains(account, "Email Address")
        self.assertNotContains(account, "Change Password")
        self.assertNotContains(account, "New Password")
        self.assertContains(account, "Reset Administrator Password")

        self.create_account(username="table-teacher")
        listing = self.client.get(reverse("user_management"))
        self.assertContains(listing, "Full Name")
        self.assertContains(listing, "Date Created")
        self.assertContains(listing, "row-actions")
        self.assertContains(listing, "Reset My Password")
        self.assertNotContains(listing, "Password update required")

    def test_admin_can_reset_own_password_and_staff_cannot_access_reset_endpoint(self):
        self.admin_login()
        response = self.client.post(reverse("user_password_reset", args=(self.admin.pk,)), {
            "temporary_password": "7", "confirm_password": "7",
        })
        self.assertRedirects(response, reverse("user_management"))
        self.admin.refresh_from_db()
        self.assertTrue(self.admin.check_password("7"))
        self.assertTrue(self.client.login(username=self.admin.username, password="7"))

        teacher, _ = make_teacher("cannot-reset")
        self.client.force_login(teacher)
        self.assertEqual(self.client.get(reverse("user_password_reset", args=(teacher.pk,))).status_code, 403)
        self.assertEqual(self.client.post(reverse("user_password_reset", args=(teacher.pk,)), {
            "temporary_password": "new", "confirm_password": "new",
        }).status_code, 403)

    def test_admin_password_reset_requires_nonempty_matching_values_and_is_admin_only(self):
        self.admin_login()
        url = reverse("user_password_reset", args=(self.admin.pk,))
        empty = self.client.post(url, {"temporary_password": "", "confirm_password": ""})
        self.assertContains(empty, "This field is required")
        mismatch = self.client.post(url, {"temporary_password": "simple", "confirm_password": "different"})
        self.assertContains(mismatch, "passwords do not match")
        self.assertFalse(self.admin.check_password("simple"))

    def test_admin_can_view_and_edit_other_teachers_collection_but_teacher_cannot(self):
        teacher, teacher_profile = make_teacher("collection-owner")
        collection = create_collection(owner=teacher_profile, title="Owned by teacher", department="CSE",
                                       semesters=["Semester 1"], closing_date=None)
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get(reverse("collection_detail", args=(collection.collection_id,))).status_code, 200)
        self.client.force_login(teacher)
        self.assertEqual(self.client.get(reverse("collection_detail", args=(collection.collection_id,))).status_code, 200)

        # Another teacher must not read, edit, export, or analyze this collection.
        other, _ = make_teacher("other-teacher")
        self.client.force_login(other)
        self.assertEqual(self.client.get(reverse("collection_detail", args=(collection.collection_id,))).status_code, 404)
        self.assertEqual(self.client.get(reverse("export_collection_csv", args=(collection.collection_id,))).status_code, 404)

    def test_admin_can_edit_export_and_analyze_another_teachers_responses(self):
        _, teacher_profile = make_teacher("managed-owner")
        collection = create_collection(owner=teacher_profile, title="Institute-wide", department="CSE",
                                       semesters=["Semester 1"], closing_date=None)
        record = submit_response(collection_id=collection.collection_id, student_id="S-22",
                                 semester="Semester 1", predictors=predictor_values())
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get(reverse("export_collection_csv", args=(collection.collection_id,))).status_code, 200)
        self.assertEqual(self.client.get(reverse("collection_analysis", args=(collection.collection_id,))).status_code, 200)
        updated = {**record.predictors, "student_id": "S-22", "semester": "Semester 1",
                   "last_semester_attendance_percent": 70.0}
        edited = self.client.post(reverse("response_edit", args=(collection.collection_id, record.record_id)), updated)
        self.assertEqual(edited.status_code, 302)
        record.refresh_from_db()
        self.assertEqual(record.predictors["last_semester_attendance_percent"], 70.0)

    def test_student_collection_link_remains_public_and_is_not_part_of_account_verification(self):
        teacher, profile = make_teacher("public-link-owner")
        collection = create_collection(owner=profile, title="Student submissions", department="CSE",
                                       semesters=["Semester 1"], closing_date=None)
        response = self.client.get(reverse("student_collect", args=(collection.collection_id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Student ID")
        self.assertContains(response, "Submit student information")
