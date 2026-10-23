from __future__ import annotations

from functools import wraps

from django.contrib import messages
from django.contrib.auth import get_user_model, update_session_auth_hash
from django.contrib.auth.views import LoginView, LogoutView
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.views.decorators.http import require_POST

from .access import institute_profile, institute_user_required
from .account_forms import (
    AdministratorCreateUserForm, AdministratorEditUserForm, AdministratorPasswordResetForm,
    EmailChangeForm, EmailAuthenticationForm, UsernameChangeForm,
)
from .models import Teacher

User = get_user_model()


class WebsiteLoginView(LoginView):
    template_name = "registration/login.html"
    authentication_form = EmailAuthenticationForm
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy("dashboard")

    def form_valid(self, form):
        self.request.session.pop("guest_mode", None)
        return super().form_valid(form)


class WebsiteLogoutView(LogoutView):
    next_page = reverse_lazy("welcome")


def welcome(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return WebsiteLoginView.as_view()(request)


@require_POST
def continue_as_guest(request):
    if not request.user.is_authenticated:
        request.session["guest_mode"] = True
    return redirect("dashboard")


@institute_user_required
def account_settings(request):
    profile = institute_profile(request.user)
    if not profile:
        return HttpResponseForbidden("This account is not provisioned by the institute administrator.")
    return render(request, "registration/account_settings.html", {
        "profile": profile,
        "username_form": UsernameChangeForm(user=request.user, initial={"username": request.user.username}),
        "email_form": EmailChangeForm(user=request.user, initial={"new_email": request.user.email}),
    })


@institute_user_required
@require_POST
def account_update(request):
    profile = institute_profile(request.user)
    if not profile:
        return HttpResponseForbidden("This account is not provisioned by the institute administrator.")
    action = request.POST.get("action")
    if action == "username":
        form = UsernameChangeForm(request.POST, user=request.user)
        if form.is_valid():
            request.user.username = form.cleaned_data["username"]
            request.user.save(update_fields=("username",))
            messages.success(request, "Username updated.")
        else:
            messages.error(request, "Please correct the username form.")
            return render(request, "registration/account_settings.html", {
                "profile": profile, "username_form": form,
                "email_form": EmailChangeForm(user=request.user),
            })
    elif action == "email":
        form = EmailChangeForm(request.POST, user=request.user)
        if form.is_valid():
            request.user.email = form.cleaned_data["new_email"]
            request.user.save(update_fields=("email",))
            messages.success(request, "Email address updated.")
        else:
            messages.error(request, "Please correct the email form.")
            return render(request, "registration/account_settings.html", {
                "profile": profile, "username_form": UsernameChangeForm(user=request.user),
                "email_form": form,
            })
    else:
        return HttpResponseForbidden("Unknown account action.")
    return redirect("account_settings")


def administrator_required(view_func):
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.info(request, "Sign in with the institute administrator account to manage users.")
            from django.contrib.auth.views import redirect_to_login
            return redirect_to_login(request.get_full_path(), login_url=reverse_lazy("login"))
        profile = institute_profile(request.user)
        if not profile or profile.role != Teacher.Role.ADMIN:
            return HttpResponseForbidden("Institute administrator access is required.")
        request.institute_admin = profile
        return view_func(request, *args, **kwargs)
    return wrapped


@administrator_required
def user_management(request):
    accounts = User.objects.filter(teacher_profile__role__in=(Teacher.Role.TEACHER, Teacher.Role.STAFF)).select_related("teacher_profile").order_by("username")
    return render(request, "registration/user_management.html", {"accounts": accounts})


@administrator_required
def user_create(request):
    form = AdministratorCreateUserForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        messages.success(request, f"{user.username} was created and can sign in now. Share the assigned password securely.")
        return redirect("user_management")
    return render(request, "registration/user_form.html", {"form": form, "heading": "Add Teacher / Staff"})


@administrator_required
def user_edit(request, user_id: int):
    user = get_object_or_404(User, pk=user_id, is_superuser=False, teacher_profile__role__in=(Teacher.Role.TEACHER, Teacher.Role.STAFF))
    profile = user.teacher_profile
    initial = {"full_name": user.get_full_name(), "username": user.username, "email": user.email, "role": profile.role}
    form = AdministratorEditUserForm(request.POST or None, user=user, initial=initial)
    if request.method == "POST" and form.is_valid():
        names = form.cleaned_data["full_name"].strip().split(maxsplit=1)
        user.first_name = names[0] if names else ""
        user.last_name = names[1] if len(names) > 1 else ""
        user.username = form.cleaned_data["username"]
        user.email = form.cleaned_data["email"]
        user.save(update_fields=("first_name", "last_name", "username", "email"))
        profile.role = form.cleaned_data["role"]
        profile.save(update_fields=("role",))
        messages.success(request, f"{user.username}'s account was updated.")
        return redirect("user_management")
    return render(request, "registration/user_form.html", {"form": form, "heading": "Edit account", "account": user})


@administrator_required
@require_POST
def user_status(request, user_id: int):
    user = get_object_or_404(User, pk=user_id, is_superuser=False, teacher_profile__role__in=(Teacher.Role.TEACHER, Teacher.Role.STAFF))
    user.is_active = request.POST.get("active") == "1"
    user.save(update_fields=("is_active",))
    messages.success(request, f"{user.username} is now {'active' if user.is_active else 'deactivated'}.")
    return redirect("user_management")


@administrator_required
def user_password_reset(request, user_id: int):
    if user_id == request.user.pk:
        user = request.user
    else:
        user = get_object_or_404(User, pk=user_id, is_superuser=False,
                                  teacher_profile__role__in=(Teacher.Role.TEACHER, Teacher.Role.STAFF))
    form = AdministratorPasswordResetForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user.set_password(form.cleaned_data["temporary_password"])
        user.save(update_fields=("password",))
        if user.pk == request.user.pk:
            update_session_auth_hash(request, user)
        messages.success(request, f"The password for {user.username} was reset and is ready to use.")
        return redirect("user_management")
    return render(request, "registration/user_password_reset.html", {"account": user, "form": form,
                                                                       "self_reset": user.pk == request.user.pk})
