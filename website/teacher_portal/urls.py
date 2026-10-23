from django.contrib import admin
from django.urls import include, path

from collection_app import auth_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", auth_views.welcome, name="welcome"),
    path("guest/", auth_views.continue_as_guest, name="continue_as_guest"),
    path("accounts/login/", auth_views.WebsiteLoginView.as_view(), name="login"),
    path("accounts/logout/", auth_views.WebsiteLogoutView.as_view(), name="logout"),
    path("accounts/me/", auth_views.account_settings, name="account_settings"),
    path("accounts/me/update/", auth_views.account_update, name="account_update"),
    path("admin-management/", auth_views.user_management, name="user_management"),
    path("admin-management/add/", auth_views.user_create, name="user_create"),
    path("admin-management/<int:user_id>/edit/", auth_views.user_edit, name="user_edit"),
    path("admin-management/<int:user_id>/status/", auth_views.user_status, name="user_status"),
    path("admin-management/<int:user_id>/password/", auth_views.user_password_reset, name="user_password_reset"),
    path("", include("collection_app.urls")),
]
