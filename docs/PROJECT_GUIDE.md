# Project Guide

Run commands from the project root, `E:\cse-dropout-prediction`, unless noted. The raw training datasets remain outside this repository; training/source audits need authorized paths in `CSE_UCI_DATASET_PATH` and `CSE_BANGLADESH_DATASET_PATH`.

## Where to Find Everything

| Folder | What it contains |
|---|---|
| `website/` | Django application and user interface, plus `manage.py` and deployment files |
| `website/templates/` | HTML pages used by Django |
| `website/collection_app/` | Accounts, student collections, forms, permissions, views, and database migrations |
| `website/teacher_portal/` | Django project settings, URL configuration, and WSGI entry point |
| `website/deployment/` | PythonAnywhere guide and WSGI example |
| `src/` | Shared machine-learning schema, prediction code, and optional training code |
| `models/` | Saved trained ensemble and metadata; normal app use loads these without training |
| `model_reports/` | Model performance results, methodology, and training documentation |
| `tests/` | Repository and prediction-service automated tests |
| `docs/` | This project handover guide and `DEMO_GUIDE.md` |
| `data/` | Versioned questionnaire/data dictionary files (not raw training records) |

The local SQLite database is stored outside the project at `%USERPROFILE%\.student_dropout\db.sqlite3` by default. Set `DJANGO_SQLITE_PATH` to choose another location.

## Common changes

| To change | Edit this file |
|---|---|
| Replace the university logo | `website/collection_app/static/collection_app/images/university_logo.png`; retain the path, then run `python website/manage.py collectstatic --noinput` |
| Change page layout or shared styling | `website/templates/collection_app/base.html` for shared layout/styles; the page-specific files in `website/templates/collection_app/` for Dashboard, Assessment, Batch Analysis, and About |
| Change the sign-in page | `website/templates/registration/login.html` |
| Change student collection questionnaire fields | `src/bangladesh_collection.py` is the canonical 18-input schema; update `website/collection_app/forms.py` and relevant templates/tests in coordination, without adding the target as a feature |
| Change prediction input validation/inference | `src/prediction_services.py` and `src/combined_ensemble_v2.py`; preserve the saved artifact contract and test both single and batch flows |
| Change Django settings or database configuration | `website/teacher_portal/settings.py`; use environment variables and never commit secrets or student data |
| Change URLs or request handling | `website/teacher_portal/urls.py`, `website/collection_app/urls.py`, and `website/collection_app/views.py` |
| Change account permissions or collection behavior | `website/collection_app/auth_views.py`, `access.py`, `auth_services.py`, `models.py`, and `services.py`; schema changes require a new migration |
| Update PythonAnywhere setup | `website/deployment/pythonanywhere.md` and `website/deployment/pythonanywhere_wsgi.example.py` |

## Key files and model safeguards

- Active artifact: `models/combined_uci_bangladesh_18_v2/weighted_ensemble_dropout_status_18.joblib`
- Metadata: `models/combined_uci_bangladesh_18_v2/model_metadata.json`
- Evaluation: `model_reports/combined_uci_bangladesh_18_v2/`
- Shared input schema: `src/bangladesh_collection.py`
- Batch/individual inference: `src/prediction_services.py`
- Optional training scripts: `src/train_bangladesh_18.py`, `src/train_combined_ensemble_18.py` (not required to run the website)

The 18-input model classifies reported current Bangladesh dropout status, not verified future dropout. UCI and Bangladesh records train separate model components and have different outcome definitions; reported cross-validation results are retrospective estimates on a small Bangladesh cohort. Do not present them as prospective future-risk validation. Do not casually change the artifact, calibration, weights, threshold, feature order, or category mappings.
