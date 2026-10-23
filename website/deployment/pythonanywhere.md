# PythonAnywhere deployment guide

This guide describes deploying the Django project from `Sanchita210057/Student_Dropout_Prediction` to PythonAnywhere. Use the reviewed `django-production` branch (or the final release branch selected by the project owner).

## 1. Create the PythonAnywhere app and checkout

Create a PythonAnywhere web app using a Python version supported by the account and compatible with the pinned scikit-learn runtime. Use the same interpreter for its virtualenv and the web app.

```bash
cd ~
git clone https://github.com/Sanchita210057/Student_Dropout_Prediction.git
cd ~/Student_Dropout_Prediction
git fetch origin
git switch django-production
python3.12 -m venv ~/.virtualenvs/dropout-django
source ~/.virtualenvs/dropout-django/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r website/requirements-mysql.txt
```

Replace `python3.12` with the selected PythonAnywhere interpreter path if that version is not offered. Before using another interpreter or changing the scikit-learn pin, validate that it can load the checked-in artifact. Do not run any training command on the server.

## 2. Create a private MySQL database

Create a database in the PythonAnywhere **Databases** tab. Keep the database credentials private. Record the database name, username, password, and host shown by PythonAnywhere. Do not create the database inside the Git checkout and do not upload a local SQLite database.

Set the following variables in the private WSGI configuration:

```text
DJANGO_DEBUG=0
DJANGO_SECRET_KEY=<generated-private-secret>
DJANGO_ALLOWED_HOSTS=<username>.pythonanywhere.com
DJANGO_CSRF_TRUSTED_ORIGINS=https://<username>.pythonanywhere.com
DJANGO_SECURE_SSL_REDIRECT=1
DJANGO_DB_ENGINE=mysql
DJANGO_DB_NAME=<database-name>
DJANGO_DB_USER=<database-user>
DJANGO_DB_PASSWORD=<database-password>
DJANGO_DB_HOST=<host-from-PythonAnywhere-Databases-tab>
DJANGO_DB_PORT=3306
DJANGO_CACHE_TABLE=django_cache_table
INSTITUTE_NAME=<institution-display-name>
```

Generate a new secret locally with `python -c "import secrets; print(secrets.token_urlsafe(50))"` and paste it directly into the private WSGI config. Never put an actual secret in `.env.example`, this guide, or Git. `DEBUG=False` enables secure session/CSRF cookies, HTTPS redirect, one-year HSTS, and database-backed rate-limit cache by default. Only enable HSTS after HTTPS works.

## 3. Configure WSGI and static files

In the Web tab, set the source/working directory to `/home/<username>/Student_Dropout_Prediction`, select `~/.virtualenvs/dropout-django`, and edit the WSGI file using [`pythonanywhere_wsgi.example.py`](pythonanywhere_wsgi.example.py). Replace `<username>`, set the environment variables above, then import `application` from `teacher_portal.wsgi`.

Map the static URL `/static/` to:

```text
/home/<username>/Student_Dropout_Prediction/website/staticfiles
```

## 4. Initialize schema and static files

Run in a Bash console with the deployment virtualenv activated:

```bash
cd ~/Student_Dropout_Prediction
source ~/.virtualenvs/dropout-django/bin/activate
python website/manage.py check --deploy
python website/manage.py migrate
python website/manage.py createcachetable django_cache_table
python website/manage.py collectstatic --noinput
python website/manage.py createsuperuser
```

The cache-table command is needed once. If it reports that the table already exists, verify the table and continue. Do not drop or recreate the application database during a normal code update.

The `createsuperuser` account receives the Institute Administrator role automatically and signs in at `/accounts/login/` like other users. Use Admin Management in the application to create Teacher/Staff accounts; no public registration exists. Each added account can sign in immediately with the assigned password. Only an administrator can reset a password, including their own; communicate replacement passwords through a private channel. Routine account management does not require Django Admin.

This demonstration intentionally has no SMTP setup, email verification, OTP, or email password recovery. An administrator assigns a replacement password if an account needs recovery. Collection links offer all configured semester choices; old collection records and response records remain preserved.

## 5. Verify before using real records

Reload the web app and test using synthetic records only:

1. Root login screen, Continue as Guest flow, and guest-only Dashboard, Individual Assessment, and About Project access.
2. 18-input assessment and PDF download.
3. Guest denial of batch routes, then authenticated Batch Analysis with CSV/Excel templates, uploads, validation, search, and exports.
4. Teacher sign-in, collection creation/closing, ownership restrictions, edit audit, and analysis.
5. Public student submission, duplicate rejection, closing date, and rate limiting.
6. The active artifact hash and threshold listed in the repository-root `README.md`.

Check the PythonAnywhere error log after the first request. Do not delete, replace, or upload a local database during deployment. Apply schema changes only through reviewed Django migrations.

## Backups and recovery

Use a private backup directory outside the Git checkout. For MySQL, run a database dump using the host/name/user from the PythonAnywhere Databases tab; let `mysqldump` prompt for the password and restrict the dump file permissions. Keep encrypted, dated backups outside the web-served tree and test restoration into a separate database before relying on them.

Example (replace placeholders and create the private backup directory first):

```bash
umask 077
mysqldump --single-transaction -h <mysql-host> -u <mysql-user> -p <database-name> > ~/private-backups/dropout-$(date +%F).sql
```

For recovery, restore a selected dump into a separate empty database first, verify its tables and records, then update the private database configuration during a planned maintenance window. Never restore over the live database without a verified backup and explicit approval. Also retain the Git revision currently deployed and the previous dependency lock/requirements state so code can be rolled back without reverting or overwriting database data.

## Privacy and limitations

The collection database contains Student IDs and student answers. Restrict access to authorized teachers and administrators, use HTTPS, rotate compromised credentials, document a retention/deletion schedule, and arrange institution-approved backups. Configure a regular scheduled `python website/manage.py clearsessions` task so expired session-stored assessment/batch report data is purged. Student submissions are not automatically used for retraining.

Bangladesh outcomes are reported current status, and predictor timing relative to discontinuation is not established. The site must not present the retrospective classification as validated future dropout prediction or use it as an automatic adverse decision.
