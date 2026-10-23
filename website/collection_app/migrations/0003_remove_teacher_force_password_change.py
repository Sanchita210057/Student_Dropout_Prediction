from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("collection_app", "0002_teacher_force_password_change_teacher_role"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="teacher",
            name="force_password_change",
        ),
    ]
