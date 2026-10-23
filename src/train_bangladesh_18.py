"""Train and evaluate the Bangladesh 18-input current-status classifier.

The target is reported current academic status (Dropout=1, Running/Passed=0).
It is not a validated future-dropout model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.bangladesh_collection import (
    CATEGORICAL_CHOICES,
    PREDICTOR_FIELDS,
    SCHEMA_VERSION,
    STATUS_FIELD,
    clean_google_forms_responses,
    load_bangladesh_training_data,
)
from src.training_data_paths import (
    BANGLADESH_DATA_ENV,
    bangladesh_dataset_path,
    require_training_source,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = bangladesh_dataset_path()
MODEL_DIR = ROOT / "models" / "bangladesh_18"
MODEL_PATH = MODEL_DIR / "dropout_current_status_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORT_DIR = ROOT / "model_reports" / "bangladesh_18"
RESULTS_PATH = REPORT_DIR / "training_results_18_input.json"
REPORT_PATH = REPORT_DIR / "model_evaluation_18_input.md"
TEST_SIZE = 0.25
SPLIT_RANDOM_STATE = 42
THRESHOLD_MAX_ALERT_RATE = 0.50

LEGACY_COMPARISON_FEATURES = tuple(PREDICTOR_FIELDS[:15])
CATEGORICAL_FIELDS = tuple(CATEGORICAL_CHOICES)
NUMERIC_FIELDS = tuple(field for field in PREDICTOR_FIELDS if field not in CATEGORICAL_FIELDS)


def make_pipeline(model_name: str, features: tuple[str, ...]) -> Pipeline:
    categorical = [field for field in features if field in CATEGORICAL_FIELDS]
    numeric = [field for field in features if field in NUMERIC_FIELDS]
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=3000, solver="liblinear", random_state=42
        )
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def parameter_grid(model_name: str) -> dict[str, list[Any]]:
    if model_name == "Logistic Regression":
        return {"classifier__C": [0.1, 1.0, 10.0]}
    if model_name == "Random Forest":
        return {
            "classifier__min_samples_leaf": [3, 7],
            "classifier__max_depth": [None, 5],
        }
    raise ValueError(f"Unsupported model: {model_name}")


def fit_search(
    model_name: str,
    features: tuple[str, ...],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int,
) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    search = GridSearchCV(
        make_pipeline(model_name, features),
        parameter_grid(model_name),
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X.loc[:, features], y)
    return search


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probability)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "flagged_count": int(prediction.sum()),
        "flagged_rate": float(prediction.mean()),
    }


def nested_training_oof(
    model_name: str,
    features: tuple[str, ...],
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Produce training-only probabilities with outer stratified folds.

    Preprocessing and hyperparameter search are refit inside every outer-train
    fold, keeping the corresponding outer validation fold untouched.
    """
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    probability = np.full(len(y), np.nan, dtype=float)
    folds = []
    for fold_index, (train_index, valid_index) in enumerate(outer.split(X, y), start=1):
        search = fit_search(
            model_name,
            features,
            X.iloc[train_index],
            y.iloc[train_index],
            folds=3,
        )
        fold_probability = search.predict_proba(X.iloc[valid_index].loc[:, features])[:, 1]
        probability[valid_index] = fold_probability
        folds.append(
            {
                "fold": fold_index,
                "average_precision_inner_cv": float(search.best_score_),
                "best_params": search.best_params_,
                "positive_support": int(y.iloc[valid_index].sum()),
                "validation_size": int(len(valid_index)),
            }
        )
    if np.isnan(probability).any():
        raise RuntimeError("Nested CV did not produce a prediction for every training row.")
    return probability, folds


def select_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    rows = []
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        rate = float(prediction.mean())
        if rate <= THRESHOLD_MAX_ALERT_RATE:
            rows.append(
                {
                    "threshold": float(threshold),
                    "f1": float(f1_score(y, prediction, zero_division=0)),
                    "precision": float(precision_score(y, prediction, zero_division=0)),
                    "recall": float(recall_score(y, prediction, zero_division=0)),
                    "flagged_rate": rate,
                }
            )
    if not rows:
        raise RuntimeError("No training-only threshold satisfies the alert-rate constraint.")
    chosen = max(rows, key=lambda row: (row["f1"], row["precision"], row["recall"]))
    return chosen["threshold"], rows


def evaluate_candidate(
    model_name: str,
    features: tuple[str, ...],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], GridSearchCV]:
    oof_probability, fold_details = nested_training_oof(model_name, features, X_train, y_train)
    threshold, threshold_search = select_threshold(y_train.to_numpy(), oof_probability)
    oof_metrics = binary_metrics(y_train.to_numpy(), oof_probability, threshold)
    full_train_search = fit_search(model_name, features, X_train, y_train, folds=5)
    test_probability = full_train_search.predict_proba(X_test.loc[:, features])[:, 1]
    test_metrics = binary_metrics(y_test.to_numpy(), test_probability, threshold)
    fold_05 = [binary_metrics(y_train.to_numpy(), oof_probability, 0.5)]
    result = {
        "model": model_name,
        "features": list(features),
        "feature_count": len(features),
        "nested_cv_oof_metrics_at_selected_threshold": oof_metrics,
        "nested_cv_oof_metrics_at_0_50": fold_05[0],
        "nested_cv_folds": fold_details,
        "threshold_selection": {
            "threshold": threshold,
            "objective": "maximize training-only OOF F1, subject to no more than 50% of training rows flagged",
            "candidate_count": len(threshold_search),
        },
        "best_training_cv_average_precision": float(full_train_search.best_score_),
        "best_training_params": full_train_search.best_params_,
        "heldout_metrics": test_metrics,
    }
    return result, full_train_search


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def train() -> dict[str, Any]:
    raw_path = require_training_source(RAW_PATH, BANGLADESH_DATA_ENV, "Bangladesh")
    X, y, original_status, audit = load_bangladesh_training_data(raw_path)
    raw_frame = pd.read_excel(raw_path, sheet_name="Form Responses 1")
    data_quality = {
        "source_rows": int(raw_frame.shape[0]),
        "source_columns": int(raw_frame.shape[1]),
        "source_missing_cells": int(raw_frame.isna().sum().sum()),
        "source_duplicate_rows": int(raw_frame.duplicated().sum()),
        "rows_removed": 0,
    }
    if tuple(X.columns) != tuple(PREDICTOR_FIELDS) or STATUS_FIELD in X.columns:
        raise RuntimeError("Prepared feature schema does not match the 18-input contract.")
    if set(y.dropna().unique()) != {0, 1}:
        raise RuntimeError("Both target classes must be present before training.")

    train_positions, test_positions = train_test_split(
        np.arange(len(X)),
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y,
    )
    X_train = X.iloc[train_positions].reset_index(drop=True)
    X_test = X.iloc[test_positions].reset_index(drop=True)
    y_train = y.iloc[train_positions].astype(int).reset_index(drop=True)
    y_test = y.iloc[test_positions].astype(int).reset_index(drop=True)

    model_results: dict[str, Any] = {"18_input": {}, "15_input_bangladesh_ablation": {}}
    searches: dict[str, GridSearchCV] = {}
    for variant, features in (
        ("18_input", tuple(PREDICTOR_FIELDS)),
        ("15_input_bangladesh_ablation", LEGACY_COMPARISON_FEATURES),
    ):
        candidate_results = {}
        for model_name in ("Logistic Regression", "Random Forest"):
            result, search = evaluate_candidate(
                model_name, features, X_train, y_train, X_test, y_test
            )
            candidate_results[model_name] = result
            searches[f"{variant}:{model_name}"] = search
        model_results[variant] = candidate_results

    selected_name = max(
        model_results["18_input"],
        key=lambda name: model_results["18_input"][name]["nested_cv_oof_metrics_at_selected_threshold"]["pr_auc_average_precision"],
    )
    selected_result = model_results["18_input"][selected_name]
    threshold = selected_result["threshold_selection"]["threshold"]

    # Training-only sensitivity check: repeat selected-model nested OOF with
    # zero GPA values left unchanged, preserving the same rows and split.
    unadjusted_X = clean_google_forms_responses(raw_frame).loc[:, PREDICTOR_FIELDS].copy()
    unadjusted_train = unadjusted_X.iloc[train_positions].reset_index(drop=True)
    unadjusted_oof, _ = nested_training_oof(selected_name, tuple(PREDICTOR_FIELDS), unadjusted_train, y_train)
    gpa_rule_sensitivity = {
        "method": "Nested stratified training-only out-of-fold evaluation with the same 18-input algorithm and training partition; raw zero GPAs retained rather than marked missing.",
        "selected_model": selected_name,
        "marked_unavailable_oof_pr_auc": selected_result["nested_cv_oof_metrics_at_selected_threshold"]["pr_auc_average_precision"],
        "zero_retained_oof_pr_auc": float(average_precision_score(y_train, unadjusted_oof)),
        "marked_unavailable_oof_metrics_at_threshold": selected_result["nested_cv_oof_metrics_at_selected_threshold"],
        "zero_retained_oof_metrics_at_same_threshold": binary_metrics(y_train.to_numpy(), unadjusted_oof, threshold),
        "interpretation": "Sensitivity to the unpublished-GPA assumption on training-only OOF predictions; not a separate model-selection or holdout-tuning step.",
    }

    # Refit the training-selected hyperparameters on every available local row.
    # Do not search/tune on the heldout partition after reporting its metrics.
    selected_training_search = searches[f"18_input:{selected_name}"]
    final_pipeline = make_pipeline(selected_name, tuple(PREDICTOR_FIELDS))
    final_pipeline.set_params(**selected_training_search.best_params_)
    final_pipeline.fit(X.loc[:, PREDICTOR_FIELDS], y)
    artifact = {
        "pipeline": final_pipeline,
        "threshold": threshold,
        "raw_input_features": list(PREDICTOR_FIELDS),
        "input_feature_count": len(PREDICTOR_FIELDS),
        "target_field": STATUS_FIELD,
        "target_is_predictor": False,
        "selected_model": selected_name,
        "selected_model_params": selected_training_search.best_params_,
        "heldout_metrics": selected_result["heldout_metrics"],
        "performance_scope": "Internal stratified heldout evaluation of reported current status only; not future dropout prediction.",
        "schema_version": SCHEMA_VERSION,
        "dataset_source": str(RAW_PATH.relative_to(ROOT)),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    status_counts = original_status.value_counts().to_dict()
    target_counts = y.value_counts().sort_index().to_dict()
    legacy_model_not_comparable = (
        "The prior UCI 15-input model uses Portuguese-coded inputs and a different outcome definition. "
        "Its benchmark is not directly comparable to this Bangladesh current-status target. "
        "A separate 15-feature Bangladesh ablation uses the same local cohort and split."
    )
    results = {
        "project": "Project A Bangladesh 18-input current dropout-status classifier",
        "dataset_source": str(RAW_PATH.relative_to(ROOT)),
        "source_sheet": "Form Responses 1",
        "schema_version": SCHEMA_VERSION,
        "raw_records": int(len(X)),
        "source_data_quality": data_quality,
        "input_feature_count": len(PREDICTOR_FIELDS),
        "input_features": list(PREDICTOR_FIELDS),
        "excluded_from_predictors": ["Timestamp", STATUS_FIELD],
        "status_original_labels": status_counts,
        "binary_target_counts": {str(int(key)): int(value) for key, value in target_counts.items()},
        "target_mapping": {
            "Dropout (Discontinued Studies)": 1,
            "Running (Currently Studying)": 0,
            "Passed (Graduated)": 0,
        },
        "data_preparation_audit": audit,
        "gpa_zero_handling": {
            "rule": "Semester GPA is unavailable when it is zero, some courses were passed, and passed courses are fewer than enrolled courses.",
            "affected_records": audit["semester_gpa_marked_unavailable_count"],
            "rows_removed": 0,
            "course_counts_changed": False,
            "affected_status_counts": audit["semester_gpa_marked_unavailable_status_counts"],
            "training_only_sensitivity": gpa_rule_sensitivity,
        },
        "validation": {
            "method": "One stratified 75/25 train/heldout split; nested stratified CV within training data; all imputers, encoders, scalers, model selection and threshold selection fitted from training folds only.",
            "random_state": SPLIT_RANDOM_STATE,
            "train_records": int(len(X_train)),
            "heldout_records": int(len(X_test)),
            "train_dropout_count": int(y_train.sum()),
            "heldout_dropout_count": int(y_test.sum()),
            "selection_metric": "Mean nested-CV PR-AUC (average precision); accuracy is not used for selection.",
            "class_imbalance_handling": "class_weight=balanced for Logistic Regression; balanced_subsample for Random Forest.",
        },
        "candidate_results": model_results,
        "gpa_zero_handling_sensitivity": gpa_rule_sensitivity,
        "selected_model": selected_name,
        "selected_model_params": selected_training_search.best_params_,
        "selected_threshold": threshold,
        "selected_18_input_heldout_metrics": selected_result["heldout_metrics"],
        "selected_15_input_bangladesh_ablation": max(
            model_results["15_input_bangladesh_ablation"],
            key=lambda name: model_results["15_input_bangladesh_ablation"][name]["nested_cv_oof_metrics_at_selected_threshold"]["pr_auc_average_precision"],
        ),
        "previous_uci_15_input_comparison": {
            "direct_comparison_meaningful": False,
            "reason": legacy_model_not_comparable,
        },
        "deployment_fit": "Final active model reuses hyperparameters selected only within the training partition, then refits those fixed settings on all 251 Bangladesh records; heldout metrics describe the pre-refit training-split model.",
        "limitations": [
            "Small local convenience sample with only 52 reported current-dropout cases.",
            "Current Academic Status is contemporaneous self-reported status; this is not future dropout prediction.",
            "Running is coded 0 by the required mapping but is not proof of graduation or future persistence.",
            "The source does not establish that every predictor was measured before discontinuation.",
            "Performance estimates have substantial uncertainty and are not externally validated.",
        ],
    }
    METADATA_PATH.write_text(json.dumps({
        "project_title": "Bangladesh University Student Dropout Current-Status Classification",
        "model_version": "bangladesh-18-input-v1",
        "schema_version": SCHEMA_VERSION,
        "dataset_source": results["dataset_source"],
        "selected_model": selected_name,
        "selected_model_params": selected_training_search.best_params_,
        "threshold": threshold,
        "raw_input_features": list(PREDICTOR_FIELDS),
        "input_feature_count": len(PREDICTOR_FIELDS),
        "target_field": STATUS_FIELD,
        "target_is_predictor": False,
        "target_mapping": results["target_mapping"],
        "heldout_metrics": selected_result["heldout_metrics"],
        "performance_scope": "Internal stratified heldout evaluation; current reported status only, not future dropout prediction.",
        "limitations": results["limitations"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    holdout = selected_result["heldout_metrics"]
    matrix = holdout["confusion_matrix"]
    report_lines = [
        "# Bangladesh 18-Input Model Evaluation",
        "",
        "## Data and label",
        "",
        f"- Source: `{results['dataset_source']}`; raw sheet: `{results['source_sheet']}`.",
        f"- Records: {len(X)}; predictors: {len(PREDICTOR_FIELDS)}; rows removed: 0.",
        f"- Raw workbook quality: {data_quality['source_rows']} rows × {data_quality['source_columns']} columns; {data_quality['source_missing_cells']} missing cells; {data_quality['source_duplicate_rows']} duplicate rows.",
        f"- Reported statuses: {status_counts}.",
        f"- Binary target: Dropout = 1; Running and Passed = 0. Running is not proof of eventual graduation or persistence.",
        f"- Zero GPA rule marked {audit['semester_gpa_marked_unavailable_count']} values missing, keeping all records and course counts unchanged.",
        f"- Affected current-status counts: {audit['semester_gpa_marked_unavailable_status_counts']}. Training-only nested-CV PR-AUC with the rule: {_fmt(gpa_rule_sensitivity['marked_unavailable_oof_pr_auc'])}; retaining the zeros: {_fmt(gpa_rule_sensitivity['zero_retained_oof_pr_auc'])}. This is a sensitivity check, not an external validation.",
        "- Timestamp and Current Academic Status were excluded from model inputs.",
        "",
        "## Validation and model selection",
        "",
        "A stratified 75/25 split was made once. Model/parameter selection used stratified nested CV within the training split. Imputation, one-hot encoding, scaling, class-weight handling and threshold selection occurred using training data only. Candidates were selected using mean nested-CV PR-AUC (average precision), not accuracy.",
        "",
        f"Selected 18-input algorithm: **{selected_name}** (training-partition CV parameters: `{selected_training_search.best_params_}`). Training-only threshold: **{threshold:.3f}**.",
        "",
        "## Heldout metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Accuracy | {_fmt(holdout['accuracy'])} |",
        f"| Balanced accuracy | {_fmt(holdout['balanced_accuracy'])} |",
        f"| Precision | {_fmt(holdout['precision'])} |",
        f"| Recall | {_fmt(holdout['recall'])} |",
        f"| F1 | {_fmt(holdout['f1'])} |",
        f"| ROC-AUC | {_fmt(holdout['roc_auc'])} |",
        f"| PR-AUC (average precision) | {_fmt(holdout['pr_auc_average_precision'])} |",
        "",
        f"Confusion matrix (TN={matrix['tn']}, FP={matrix['fp']}, FN={matrix['fn']}, TP={matrix['tp']}).",
        "",
        "## Same-cohort 15-feature ablation",
        "",
        "The 15-feature Bangladesh ablation uses the same split and local category meanings. It is not the prior UCI model.",
        "",
        "| Feature set | Model | Training-only threshold | CV OOF PR-AUC | Heldout accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN/FP/FN/TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, result_set in model_results.items():
        for name, value in result_set.items():
            cv_ap = value["nested_cv_oof_metrics_at_selected_threshold"]["pr_auc_average_precision"]
            test = value["heldout_metrics"]
            cm = test["confusion_matrix"]
            report_lines.append(
                f"| {variant} ({value['feature_count']} inputs) | {name} | {value['threshold_selection']['threshold']:.3f} | {_fmt(cv_ap)} | {_fmt(test['accuracy'])} | {_fmt(test['balanced_accuracy'])} | {_fmt(test['precision'])} | {_fmt(test['recall'])} | {_fmt(test['f1'])} | {_fmt(test['roc_auc'])} | {_fmt(test['pr_auc_average_precision'])} | {cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']} |"
            )
    report_lines.extend(
        [
            "",
            "The previous UCI 15-input model is not directly comparable because it uses Portuguese-coded inputs and a different outcome definition. Its artifact remains preserved as the legacy version.",
        "The extra Bangladesh-only attendance, household-income and employment fields are not available in the UCI records. UCI data were therefore not pooled into this training set and no synthetic values were assigned.",
        "Predictor timing is not established in the questionnaire. Semester GPA, course completion, attendance, tuition/debt, and employment may describe conditions concurrent with or after a student stopped studying. This creates a material retrospective leakage risk for any claimed early warning use; only a future longitudinal study can establish valid prediction timing.",
        f"The selected {selected_name} had higher nested-CV OOF PR-AUC ({_fmt(selected_result['nested_cv_oof_metrics_at_selected_threshold']['pr_auc_average_precision'])}) than the other 18-input candidate ({_fmt(model_results['18_input']['Random Forest' if selected_name == 'Logistic Regression' else 'Logistic Regression']['nested_cv_oof_metrics_at_selected_threshold']['pr_auc_average_precision'])}), so it was selected without reference to the heldout split. Random Forest scored higher on this one heldout split; that disagreement is reported but does not change the pre-specified training-only choice.",
            "",
            "## Limitations",
            "",
            "The sample is small (251 responses, including 52 Dropout labels). The label is reported current status, not a dated future outcome. Running is coded 0 only because the project requirement specifies it; it is not confirmed persistence. The source does not establish predictor timing relative to discontinuation. Treat the scores as an internal academic prototype, not a validated early-warning system.",
            "",
            "The deployed artifact was refit on all 251 records using hyperparameters selected on training data only; metrics above come from the pre-refit heldout split.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "selected_model": selected_name,
        "feature_count": len(PREDICTOR_FIELDS),
        "threshold": threshold,
        "heldout_metrics": holdout,
        "15_input_ablation": results["candidate_results"]["15_input_bangladesh_ablation"],
        "model_path": str(MODEL_PATH),
        "report_path": str(REPORT_PATH),
    }, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    train()
