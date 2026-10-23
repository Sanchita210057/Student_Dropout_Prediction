from django.conf import settings


def institute_branding(request):
    from .access import institute_profile
    profile = institute_profile(request.user) if request.user.is_authenticated else None
    return {
        "institute_name": settings.INSTITUTE_NAME,
        "institute_role": profile.role if profile else "",
        "guest_mode": bool(request.session.get("guest_mode")),
    }
