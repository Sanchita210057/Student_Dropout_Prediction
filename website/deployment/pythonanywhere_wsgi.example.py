"""Example PythonAnywhere WSGI file. Set private values in the Web tab."""
import os
import sys

# Update the account name to match the PythonAnywhere home directory.
repository = "/home/<username>/Student_Dropout_Prediction"
django_project = os.path.join(repository, "website")
sys.path.insert(0, repository)
sys.path.insert(0, django_project)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "teacher_portal.settings")
os.environ.setdefault("DJANGO_DEBUG", "0")
# Set DJANGO_SECRET_KEY, host/origin, and private database variables in the
# PythonAnywhere WSGI config. Do not put their values in this example file.

from django.core.wsgi import get_wsgi_application  # noqa: E402

application = get_wsgi_application()
