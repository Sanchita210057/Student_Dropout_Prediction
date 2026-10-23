#!/usr/bin/env python
"""Management entry point for the separate Django teacher portal."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "teacher_portal.settings")

from django.core.management import execute_from_command_line  # noqa: E402

execute_from_command_line(sys.argv)
