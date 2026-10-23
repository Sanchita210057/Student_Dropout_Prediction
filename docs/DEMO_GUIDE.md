# Supervisor Demonstration Guide

Use a synthetic demonstration database and invented student answers only. Do not expose real student IDs, account credentials, or private response workbooks.

## Before the demonstration

From the project root:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python website/manage.py migrate
python website/manage.py createsuperuser
python website/manage.py runserver 127.0.0.1:8000
```

For a fresh safe demonstration, set `DJANGO_SQLITE_PATH` to a new SQLite file outside the project before running migrations. Create a synthetic administrator for the demo and do not reuse production credentials. The initial superuser is provisioned as an Institute Administrator.

## Walkthrough

1. Open `http://127.0.0.1:8000/` and show the university-branded sign-in page.
2. Choose **Continue as Guest**. Explain that guests can see Dashboard, Individual Assessment, and About Project but cannot access private batch records.
3. On Dashboard, introduce the system. On Individual Assessment, enter a synthetic example and show the risk assessment, indicators, support suggestions, and PDF download.
4. Open About Project and explain the two separate model components, the fixed 5%/95% combination, and the retrospective evaluation results.
5. Sign in as the Institute Administrator. Open Admin Management and create a demonstration Teacher using a temporary synthetic password.
6. Log out and sign in as that Teacher with the assigned demonstration password. Use My Account to change it if useful for the walkthrough.
7. Open Batch Analysis / Teacher Workspace and create a collection with a title, department/class, allowed semesters, and optional closing date.
8. Copy the generated public collection URL into a private browser window. Submit an invented Student ID, semester, and 18 synthetic answers. Show the successful confirmation.
9. Return to the Teacher workspace. View the saved response, demonstrate the duplicate Student ID message, then edit the record and point out the audit history.
10. Export the collection as CSV and Excel. Run direct batch analysis and show the risk summary and downloadable results.
11. Close the collection and explain that subsequent student submissions are blocked. Point out that new submissions do not retrain the model.

## Explaining the model responsibly

The saved Logistic Regression ensemble has 18 student-facing inputs. Its academic component was trained on 3,630 resolved UCI records with ten aligned features; its local component used 251 Bangladesh questionnaire responses with 18 predictors. The cohorts are not pooled. The reported nested five-fold cross-validation results are retrospective estimates for reported current dropout-status patterns in a small local sample. They do not establish prospective accuracy for predicting future dropout. In particular, a student currently marked Running is not confirmed to have graduated.

Use only synthetic data for the demonstration. Before any real collection, obtain institutional approval, provide a clear student notice, restrict accounts, enable HTTPS, set retention rules, and protect database backups.
