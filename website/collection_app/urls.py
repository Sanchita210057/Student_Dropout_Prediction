from django.urls import path
from . import auth_views, views

urlpatterns = [
    path("dashboard/", views.dashboard, name="dashboard"),
    path("dashboard/", views.dashboard, name="home"),
    path("assessment/", views.individual_assessment, name="individual_assessment"),
    path("assessment/report.pdf", views.assessment_pdf_download, name="assessment_pdf_download"),
    path("batch/", views.batch_analysis, name="batch_analysis"),
    path("batch/template/<str:file_type>/", views.batch_template_download, name="batch_template_download"),
    path("batch/results/<str:file_type>/", views.batch_result_download, name="batch_result_download"),
    path("about/", views.about_project, name="about_project"),
    path("workspace/", views.workspace, name="workspace"),
    path("workspace/collections/create/", views.collection_create, name="collection_create"),
    path("workspace/collections/<str:collection_id>/", views.collection_detail, name="collection_detail"),
    path("workspace/collections/<str:collection_id>/status/", views.collection_status, name="collection_status"),
    path("workspace/collections/<str:collection_id>/edit/<uuid:record_id>/", views.response_edit, name="response_edit"),
    path("workspace/collections/<str:collection_id>/export.csv", views.export_collection_csv, name="export_collection_csv"),
    path("workspace/collections/<str:collection_id>/export.xlsx", views.export_collection_excel, name="export_collection_excel"),
    path("workspace/collections/<str:collection_id>/analyze/", views.collection_analysis, name="collection_analysis"),
    path("workspace/collections/<str:collection_id>/analysis.csv", views.analysis_download, {"file_type": "csv"}, name="analysis_csv"),
    path("workspace/collections/<str:collection_id>/analysis.xlsx", views.analysis_download, {"file_type": "xlsx"}, name="analysis_excel"),
    path("collect/<str:collection_id>/", views.student_collect, name="student_collect"),
]
