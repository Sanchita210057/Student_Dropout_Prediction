from django.contrib import admin
from .models import Collection, EditHistory, StudentResponse, Teacher


@admin.register(Teacher)
class TeacherAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at")
    search_fields = ("user__username", "user__email")


@admin.register(Collection)
class CollectionAdmin(admin.ModelAdmin):
    list_display = ("title", "department", "owner", "status", "created_at")
    list_filter = ("status", "department")
    search_fields = ("title", "collection_id", "owner__user__username")


@admin.register(StudentResponse)
class StudentResponseAdmin(admin.ModelAdmin):
    list_display = ("student_id", "semester", "collection", "submitted_at")
    search_fields = ("student_id", "collection__title")
    readonly_fields = ("collection", "record_id", "student_id", "student_id_key", "semester",
                       "predictors", "submitted_at", "updated_at")

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(EditHistory)
class EditHistoryAdmin(admin.ModelAdmin):
    list_display = ("response", "editor", "edited_at")
    readonly_fields = ("collection", "response", "editor", "edited_at", "previous_values", "new_values")

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
