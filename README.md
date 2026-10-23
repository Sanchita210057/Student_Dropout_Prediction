# Student Dropout Early Warning System

The final local web application is Django. It loads the saved 18-input Logistic Regression ensemble and does not train during normal use.

The target is **reported current Bangladesh dropout status**, not future dropout. Bangladesh labels Dropout as 1 and Running/Passed as 0; Running does not mean the student eventually graduated.

## Main features

The site provides:

- Dashboard
- Individual Assessment with 18 predictors and a PDF report
- Batch Analysis with CSV/Excel templates, validation, results, search, and downloads
- About Project with model methodology and five-fold evaluation results
- Teacher Workspace with authenticated, owner-scoped collections, record editing and audit history, data exports, and direct saved-model analysis
- Secure shareable student submission forms with Student ID, allowed Semester, and the same 18 predictors
- Institute-managed accounts with username/email sign-in, Guest access, administrator user management, and administrator-assigned password resets

The student status target is not an input. New collection submissions never retrain the model.

## Active model

Artifact: `models/combined_uci_bangladesh_18_v2/weighted_ensemble_dropout_status_18.joblib`  
SHA-256: `D9F4D5663002E3C17482E57EE5E50E97038CFEDB3B987046ECB0F652CD4DBA06`  
Threshold: `0.3040698561883773` (30.4%)  
Ensemble: 5% UCI academic component and 95% Bangladesh-local component, followed by logistic calibration.

Separate components use 3,630 resolved UCI records (10 aligned academic features) and 251 Bangladesh questionnaire responses (18 predictors); the datasets are not pooled. Nested five-fold cross-validation on the local cohort reported accuracy 96.4%, precision 89.1%, recall 94.2%, F1 91.6%, ROC-AUC 98.6%, PR-AUC 93.5%, Brier 0.032, and TN=193, FP=6, FN=3, TP=49. These retrospective estimates on a small local sample are not prospective validation. See [model architecture and evaluation](model_reports/combined_uci_bangladesh_18_v2/model_architecture_and_evaluation.md).

## Project structure

`website/` contains the Django application, `src/` the shared schema/inference and optional research scripts, `models/` the active artifact, `model_reports/` the methodology, `data/` the versioned questionnaire dictionary, `tests/` the automated tests, and `docs/` the handover guides. Raw training datasets are kept outside this tree. See [Project Guide](docs/PROJECT_GUIDE.md) for exact file locations and [Demo Guide](docs/DEMO_GUIDE.md) for a synthetic demonstration.

## Install and run locally

From the repository root, use a Python version compatible with the pinned scikit-learn runtime:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python website/manage.py migrate
python website/manage.py createsuperuser
python website/manage.py runserver 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/>. The welcome page offers institute sign-in or Continue as Guest. Create the initial administrator with `python website/manage.py createsuperuser`; the administrator role profile is provisioned automatically. The administrator signs in through the website and adds Teacher/Staff accounts using Admin Management. New accounts can sign in immediately. Teachers manage collections from Batch Analysis or `/workspace/`. Student links at `/collect/<secure-token>/` remain public.

There is no public account registration, email verification, SMTP, OTP, or email password recovery. An administrator assigns a temporary replacement password if a user loses access; communicate it through a private channel.

Local SQLite defaults to `~/.student_dropout/db.sqlite3` (on Windows, `%USERPROFILE%\.student_dropout\db.sqlite3`), outside the source tree. The populated local database was moved there without changing its records. Override the location with `DJANGO_SQLITE_PATH` or `DJANGO_LOCAL_DATA_DIR`. Use `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=0`, explicit `DJANGO_ALLOWED_HOSTS`, and `DJANGO_CSRF_TRUSTED_ORIGINS` for hosted use.

## Verification

```powershell
python website/manage.py check
python website/manage.py makemigrations --check --dry-run
python website/manage.py test collection_app
python -m pytest -q
python website/manage.py collectstatic --noinput
```

The Django production runtime is defined by `requirements.txt`. Optional MySQL support is in `website/requirements-mysql.txt`; development test tools are in `requirements-dev.txt`.

## Data, privacy, and training

Normal website use loads the saved model artifact and does not require either raw training dataset. The datasets are stored outside the source tree for authorized research and reproduction. Student collection records, local databases, environment secrets, and raw survey response workbooks must not be committed. Student IDs and answers are private personal data; configure restricted teacher accounts, HTTPS, retention rules, and encrypted database backups before collecting real submissions.

Training scripts remain separate and optional. They require authorized external copies; configure `CSE_UCI_DATASET_PATH` and `CSE_BANGLADESH_DATASET_PATH` to those files. Do not run training during deployment.

## PythonAnywhere

Use [website/deployment/pythonanywhere.md](website/deployment/pythonanywhere.md) for deployment, private environment settings, WSGI configuration, static files, database setup, backups, and verification.
