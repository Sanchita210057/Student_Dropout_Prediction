from django.contrib.auth import get_user_model
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Teacher


@receiver(post_save, sender=get_user_model())
def provision_bootstrap_administrator(sender, instance, created, **kwargs):
    """Give createsuperuser a normal website profile without changing its Django flags."""
    if created and instance.is_superuser:
        Teacher.objects.get_or_create(
            user=instance,
            defaults={"role": Teacher.Role.ADMIN},
        )
