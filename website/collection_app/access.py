from __future__ import annotations

from functools import wraps

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import redirect_to_login
from django.http import HttpResponseForbidden
from django.urls import reverse

from .models import Teacher


def institute_profile(user):
    try:
        profile = user.teacher_profile
    except Teacher.DoesNotExist:
        if user.is_superuser:
            profile, _ = Teacher.objects.get_or_create(user=user, defaults={"role": Teacher.Role.ADMIN})
        else:
            return None
    if user.is_superuser and profile.role != Teacher.Role.ADMIN:
        profile.role = Teacher.Role.ADMIN
        profile.save(update_fields=("role",))
    if not user.is_active:
        return None
    return profile


def institute_admin_required(view_func):
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return _sign_in_redirect(request)
        profile = institute_profile(request.user)
        if not profile or profile.role != Teacher.Role.ADMIN:
            return HttpResponseForbidden("Institute administrator access is required.")
        request.teacher = profile
        return view_func(request, *args, **kwargs)
    return wrapped


def institute_user_required(view_func):
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return _sign_in_redirect(request)
        profile = institute_profile(request.user)
        if not profile:
            return HttpResponseForbidden("A provisioned institute account is required.")
        request.teacher = profile
        return view_func(request, *args, **kwargs)
    return wrapped


def _sign_in_redirect(request):
    messages.info(request, "Sign in with an institute account to use this feature.")
    return redirect_to_login(request.get_full_path(), login_url=reverse("login"))


def public_site_required(view_func):
    """Allow signed-in users and visitors who explicitly entered Guest mode."""
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        if request.user.is_authenticated or request.session.get("guest_mode"):
            return view_func(request, *args, **kwargs)
        return _sign_in_redirect(request)
    return wrapped
