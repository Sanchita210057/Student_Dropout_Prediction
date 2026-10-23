from __future__ import annotations

from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm
from django.core.exceptions import ValidationError

from .models import Teacher
from .access import institute_profile

User = get_user_model()


def _unique_value(field: str, value: str, *, exclude_pk=None) -> bool:
    query = User.objects.filter(**{f"{field}__iexact": value.strip()})
    if exclude_pk is not None:
        query = query.exclude(pk=exclude_pk)
    return not query.exists()


class EmailAuthenticationForm(AuthenticationForm):
    username = forms.CharField(label="Username or Email", max_length=254, widget=forms.TextInput(attrs={"autocomplete": "username"}))

    def clean(self):
        submitted = self.cleaned_data.get("username", "").strip()
        address = self.request.META.get("REMOTE_ADDR", "unknown") if self.request else "unknown"
        from .auth_services import allow_login_attempt, clear_login_failures, record_login_failure
        allowed, key = allow_login_attempt(submitted, address)
        self.login_attempt_key = key
        if not allowed:
            raise ValidationError("Too many unsuccessful sign-in attempts. Try again later.")
        if "@" in submitted:
            user = User.objects.filter(email__iexact=submitted).order_by("pk").first()
            if user:
                self.cleaned_data["username"] = user.get_username()
        try:
            cleaned = super().clean()
        except ValidationError:
            record_login_failure(key)
            raise
        clear_login_failures(key)
        return cleaned

    def confirm_login_allowed(self, user):
        profile = institute_profile(user)
        if not profile:
            raise ValidationError("Invalid username/email or password.", code="invalid_login")
        if not user.is_active:
            raise ValidationError("Invalid username/email or password.", code="invalid_login")


class AdministratorCreateUserForm(forms.Form):
    full_name = forms.CharField(max_length=150, label="Full Name")
    username = forms.CharField(max_length=150)
    email = forms.EmailField(label="Email Address")
    role = forms.ChoiceField(choices=((Teacher.Role.TEACHER, "Teacher"), (Teacher.Role.STAFF, "Staff")))
    temporary_password = forms.CharField(label="Password", widget=forms.PasswordInput)
    confirm_password = forms.CharField(label="Confirm Password", widget=forms.PasswordInput)

    def clean_username(self):
        value = self.cleaned_data["username"].strip()
        if not _unique_value("username", value):
            raise ValidationError("That username is already in use.")
        return value

    def clean_email(self):
        value = self.cleaned_data["email"].strip().lower()
        if not _unique_value("email", value):
            raise ValidationError("That email address is already in use.")
        return value

    def clean(self):
        cleaned = super().clean()
        password = cleaned.get("temporary_password")
        if password and cleaned.get("confirm_password") and password != cleaned["confirm_password"]:
            self.add_error("confirm_password", "The passwords do not match.")
        return cleaned

    def save(self):
        if not self.is_valid():
            raise ValueError("Cannot save an invalid account form.")
        name = self.cleaned_data["full_name"].strip().split(maxsplit=1)
        user = User.objects.create_user(
            username=self.cleaned_data["username"], email=self.cleaned_data["email"],
            password=self.cleaned_data["temporary_password"], first_name=name[0],
            last_name=name[1] if len(name) > 1 else "", is_active=True,
        )
        Teacher.objects.create(user=user, role=self.cleaned_data["role"])
        return user


class AdministratorEditUserForm(forms.Form):
    full_name = forms.CharField(max_length=150, label="Full Name")
    username = forms.CharField(max_length=150)
    email = forms.EmailField(label="Email Address")
    role = forms.ChoiceField(choices=((Teacher.Role.TEACHER, "Teacher"), (Teacher.Role.STAFF, "Staff")))

    def __init__(self, *args, user, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_username(self):
        value = self.cleaned_data["username"].strip()
        if not _unique_value("username", value, exclude_pk=self.user.pk):
            raise ValidationError("That username is already in use.")
        return value

    def clean_email(self):
        value = self.cleaned_data["email"].strip().lower()
        if not _unique_value("email", value, exclude_pk=self.user.pk):
            raise ValidationError("That email address is already in use.")
        return value


class UsernameChangeForm(forms.Form):
    username = forms.CharField(max_length=150)

    def __init__(self, *args, user, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_username(self):
        value = self.cleaned_data["username"].strip()
        if not _unique_value("username", value, exclude_pk=self.user.pk):
            raise ValidationError("That username is already in use.")
        return value


class EmailChangeForm(forms.Form):
    new_email = forms.EmailField(label="New email address")

    def __init__(self, *args, user, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_new_email(self):
        value = self.cleaned_data["new_email"].strip().lower()
        if not _unique_value("email", value, exclude_pk=self.user.pk):
            raise ValidationError("That email address is already in use.")
        if value.casefold() == self.user.email.casefold():
            raise ValidationError("Enter a different email address.")
        return value


class AdministratorPasswordResetForm(forms.Form):
    temporary_password = forms.CharField(label="Password", widget=forms.PasswordInput)
    confirm_password = forms.CharField(label="Confirm password", widget=forms.PasswordInput)

    def clean(self):
        cleaned = super().clean()
        password = cleaned.get("temporary_password")
        if password and not password.strip():
            self.add_error("temporary_password", "Enter a nonempty password.")
        if password and cleaned.get("confirm_password") and password != cleaned["confirm_password"]:
            self.add_error("confirm_password", "The passwords do not match.")
        return cleaned
