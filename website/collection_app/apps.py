from django.apps import AppConfig


class CollectionAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "collection_app"

    def ready(self):
        from . import signals  # noqa: F401
