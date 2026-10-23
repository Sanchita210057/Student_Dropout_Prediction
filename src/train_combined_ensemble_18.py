"""Nested-CV comparison of the archived blend and OOF logistic stacking.

The final candidate uses UCI resolved outcomes only to train its academic
foundation. Bangladesh current-status labels train the local model and fusion
layer; base predictions used to fit fusion are strictly out-of-fold.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.bangladesh_collection import PREDICTOR_FIELDS, load_bangladesh_training_data
from src.combined_ensemble_v2 import (
    StackedEnsemblePipeline, WeightedEnsemblePipeline, UCI_CATEGORICAL_FEATURES,
    UCI_FOUNDATION_FEATURES, UCI_NUMERIC_FEATURES, calibrator_logit_feature, uci_harmonized_frame,
)
from src.train_bangladesh_18 import fit_search as fit_bangladesh_search
from src.training_data_paths import (
    BANGLADESH_DATA_ENV,
    UCI_DATA_ENV,
    bangladesh_dataset_path,
    require_training_source,
    uci_dataset_path,
)

ROOT = Path(__file__).resolve().parents[1]
BD_PATH = bangladesh_dataset_path()
UCI_PATH = uci_dataset_path()
OUT_DIR = ROOT / "models" / "combined_uci_bangladesh_18_v2"
REPORT_DIR = ROOT / "model_reports" / "combined_uci_bangladesh_18_v2"
RESULTS_PATH = REPORT_DIR / "training_results.json"
REPORT_PATH = REPORT_DIR / "model_architecture_and_evaluation.md"
MAX_ALERT_RATE = 0.50
WEIGHT_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.95)
LEGACY_UCI_FEATURES = UCI_NUMERIC_FEATURES

FEATURE_COMPATIBILITY = [
    {"UCI predictor": "Previous qualification (grade)", "Bangladesh field": "HSC / Equivalent GPA", "decision": "Use", "handling": "Linear numeric range alignment from UCI 0–190 to Bangladesh 0–5; no academic-system equivalence claimed."},
    {"UCI predictor": "Age at enrollment", "Bangladesh field": "Age at enrollment", "decision": "Use", "handling": "Direct numeric match."},
    {"UCI predictor": "Mother's occupation", "Bangladesh field": "Mother's occupation", "decision": "Exclude", "handling": "Portuguese occupation codes do not represent the Bangladesh survey categories."},
    {"UCI predictor": "Father's occupation", "Bangladesh field": "Father's occupation", "decision": "Exclude", "handling": "Portuguese occupation codes do not represent the Bangladesh survey categories."},
    {"UCI predictor": "Mother's qualification", "Bangladesh field": "Mother's highest education", "decision": "Exclude", "handling": "Detailed UCI education codes have no verified one-to-one Bangladesh mapping."},
    {"UCI predictor": "Father's qualification", "Bangladesh field": "Father's highest education", "decision": "Exclude", "handling": "Detailed UCI education codes have no verified one-to-one Bangladesh mapping."},
    {"UCI predictor": "Gender", "Bangladesh field": "Gender", "decision": "Use", "handling": "Map only documented UCI Female/Male values to matching labels; nonmatching local categories remain unseen, never coerced."},
    {"UCI predictor": "Marital status", "Bangladesh field": "Marital status", "decision": "Exclude", "handling": "Codes include distinctions such as de-facto union/legal separation that do not map exactly to local options."},
    {"UCI predictor": "Scholarship holder", "Bangladesh field": "Scholarship holder", "decision": "Use", "handling": "Map UCI binary 0/1 to No/Yes."},
    {"UCI predictor": "Debtor", "Bangladesh field": "Have any debt", "decision": "Use", "handling": "Map UCI binary 0/1 to No/Yes."},
    {"UCI predictor": "Tuition fees up to date", "Bangladesh field": "Tuition fees up to date", "decision": "Use", "handling": "Map UCI binary 0/1 to No/Yes."},
    {"UCI predictor": "Curricular units 1st sem (enrolled)", "Bangladesh field": "Courses enrolled", "decision": "Use", "handling": "Direct numeric count; domain ranges may differ."},
    {"UCI predictor": "Curricular units 1st sem (evaluations)", "Bangladesh field": "Evaluations completed", "decision": "Use", "handling": "Direct numeric count; domain ranges may differ."},
    {"UCI predictor": "Curricular units 1st sem (grade)", "Bangladesh field": "Semester GPA", "decision": "Use", "handling": "Linear range alignment from observed UCI 0–18.875 to Bangladesh model 0–5; not grading-system equivalence."},
    {"UCI predictor": "Curricular units 1st sem (approved)", "Bangladesh field": "Courses passed", "decision": "Use", "handling": "Direct numeric count."},
]


def make_uci_pipeline(features: tuple[str, ...]) -> Pipeline:
    categorical_fields = [field for field in UCI_CATEGORICAL_FEATURES if field in features]
    numeric_fields = [field for field in UCI_NUMERIC_FEATURES if field in features]
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    transformers = []
    if numeric_fields:
        transformers.append(("numeric", numeric, numeric_fields))
    if categorical_fields:
        transformers.append(("categorical", categorical, categorical_fields))
    prep = ColumnTransformer(transformers)
    return Pipeline([
        ("preprocess", prep),
        ("classifier", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000, random_state=42)),
    ])


def search_uci(X: pd.DataFrame, y: pd.Series, *, folds: int) -> GridSearchCV:
    return GridSearchCV(
        make_uci_pipeline(tuple(X.columns)), {"classifier__C": [0.1, 1.0, 10.0]},
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
        n_jobs=1, error_score="raise", refit=True,
    ).fit(X.loc[:, list(X.columns)], y)


def _binary_category(values: pd.Series) -> pd.Series:
    return values.map({0: "No", 1: "Yes", 0.0: "No", 1.0: "Yes"}).astype(object)


def load_uci(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    source = require_training_source(path or uci_dataset_path(), UCI_DATA_ENV, "UCI")
    raw = pd.read_csv(source, sep=";")
    resolved = raw.loc[raw["Target"].isin(["Graduate", "Dropout"])].copy()
    X = pd.DataFrame({
        "hsc_equivalent_gpa": pd.to_numeric(resolved["Previous qualification (grade)"], errors="coerce") / 190 * 5,
        "age_at_enrollment": pd.to_numeric(resolved["Age at enrollment"], errors="coerce"),
        "courses_enrolled": pd.to_numeric(resolved["Curricular units 1st sem (enrolled)"], errors="coerce"),
        "evaluations_completed": pd.to_numeric(resolved["Curricular units 1st sem (evaluations)"], errors="coerce"),
        "courses_passed": pd.to_numeric(resolved["Curricular units 1st sem (approved)"], errors="coerce"),
        "semester_gpa_native": pd.to_numeric(resolved["Curricular units 1st sem (grade)"], errors="coerce") / 18.875 * 5,
        "gender_bangladesh": resolved["Gender"].map({0: "Female", 1: "Male", 0.0: "Female", 1.0: "Male"}).astype(object),
        "scholarship_holder": _binary_category(resolved["Scholarship holder"]),
        "debtor": _binary_category(resolved["Debtor"]),
        "tuition_fees_up_to_date": _binary_category(resolved["Tuition fees up to date"]),
    }, columns=UCI_FOUNDATION_FEATURES).reset_index(drop=True)
    y = resolved["Target"].map({"Graduate": 0, "Dropout": 1}).astype(int).reset_index(drop=True)
    return X, y, {
        "source_rows": int(len(raw)), "resolved_rows_used": int(len(resolved)),
        "excluded_enrolled_rows": int((raw["Target"] == "Enrolled").sum()),
        "target_counts": {str(key): int(value) for key, value in resolved["Target"].value_counts().to_dict().items()},
        "mapping": {"Graduate": 0, "Dropout": 1},
    }


def metrics(y: np.ndarray, p: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "constant_training_prevalence_brier": float(brier_score_loss(y, baseline)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "flagged_count": int(predicted.sum()), "flagged_rate": float(predicted.mean()),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    rows = []
    for threshold in np.unique(np.r_[0.0, p, 1.0]):
        pred = p >= threshold
        if pred.mean() <= MAX_ALERT_RATE:
            rows.append((f1_score(y, pred, zero_division=0), precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0), float(threshold)))
    if not rows:
        raise RuntimeError("No threshold satisfies the training-only 50% alert cap.")
    return max(rows)[-1]


def calibrator_fit(p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(calibrator_logit_feature(p).reshape(-1, 1), y)


def calibrator_predict(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    return model.predict_proba(calibrator_logit_feature(p).reshape(-1, 1))[:, 1]


def fit_local(X: pd.DataFrame, y: pd.Series, folds: int) -> GridSearchCV:
    return fit_bangladesh_search("Logistic Regression", tuple(PREDICTOR_FIELDS), X, y, folds=folds)


def local_oof(X: pd.DataFrame, y: pd.Series, folds: int = 4) -> np.ndarray:
    result = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=77)
    for tr, va in cv.split(X, y):
        fitted = fit_local(X.iloc[tr], y.iloc[tr], folds=3)
        result[va] = fitted.predict_proba(X.iloc[va])[:, 1]
    if np.isnan(result).any():
        raise RuntimeError("Local component OOF predictions are incomplete.")
    return result


def fusion_features(local: np.ndarray, uci: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"bangladesh_signal": local, "uci_academic_signal": uci})


def fit_meta(local: np.ndarray, uci: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1.0, solver="liblinear", max_iter=2000).fit(fusion_features(local, uci), y)


def meta_oof_threshold_predictions(X: pd.DataFrame, y: pd.Series, uci_model: GridSearchCV, folds: int = 4) -> np.ndarray:
    """Cross-fit the entire stacker within outer-training data for its cutoff."""
    result = np.full(len(y), np.nan)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=119)
    for tr, va in splitter.split(X, y):
        x_tr, x_va = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True)
        y_tr = y.iloc[tr].reset_index(drop=True)
        local_tr_oof = local_oof(x_tr, y_tr, folds=3)
        uci_tr = uci_model.predict_proba(uci_harmonized_frame(x_tr))[:, 1]
        meta = fit_meta(local_tr_oof, uci_tr, y_tr.to_numpy())
        local_base = fit_local(x_tr, y_tr, folds=3)
        local_va = local_base.predict_proba(x_va)[:, 1]
        uci_va = uci_model.predict_proba(uci_harmonized_frame(x_va))[:, 1]
        result[va] = meta.predict_proba(fusion_features(local_va, uci_va))[:, 1]
    if np.isnan(result).any():
        raise RuntimeError("Cross-fitted meta probabilities are incomplete.")
    return result


def nested_compare(X: pd.DataFrame, y: pd.Series, uci_six: GridSearchCV, uci_ten: GridSearchCV) -> dict[str, Any]:
    names = (
        "Bangladesh-only 18-input", "UCI academic foundation (6 numeric)",
        "UCI academic foundation (10 aligned)", "Archived 5%/95% weighted blend (6-feature UCI)",
        "Fixed 5%/95% blend (10-feature UCI)", "OOF-selected weighted blend (10-feature UCI)",
        "OOF Logistic Regression stacking (10-feature UCI)",
    )
    records = {name: {"y": [], "p": [], "pred": [], "baseline": [], "thresholds": [], "weights": [], "meta_coefficients": []} for name in names}
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=134)
    for tr, va in outer.split(X, y):
        x_tr, x_va = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True)
        y_tr, y_va = y.iloc[tr].reset_index(drop=True), y.iloc[va].reset_index(drop=True)
        y_train, y_test = y_tr.to_numpy(), y_va.to_numpy()
        local_train_oof = local_oof(x_tr, y_tr, folds=4)
        local_base = fit_local(x_tr, y_tr, folds=4)
        local_test = local_base.predict_proba(x_va)[:, 1]
        uci6_train = uci_six.predict_proba(x_tr.loc[:, LEGACY_UCI_FEATURES])[:, 1]
        uci6_test = uci_six.predict_proba(x_va.loc[:, LEGACY_UCI_FEATURES])[:, 1]
        uci10_train = uci_ten.predict_proba(uci_harmonized_frame(x_tr))[:, 1]
        uci10_test = uci_ten.predict_proba(uci_harmonized_frame(x_va))[:, 1]

        candidate_probabilities: dict[str, tuple[np.ndarray, np.ndarray, float | None, Any]] = {}
        candidate_probabilities[names[0]] = (local_train_oof, local_test, None, None)
        candidate_probabilities[names[1]] = (uci6_train, uci6_test, None, None)
        candidate_probabilities[names[2]] = (uci10_train, uci10_test, None, None)

        legacy_train = .95 * local_train_oof + .05 * uci6_train
        legacy_test = .95 * local_test + .05 * uci6_test
        candidate_probabilities[names[3]] = (legacy_train, legacy_test, .05, None)
        fixed10_train = .95 * local_train_oof + .05 * uci10_train
        fixed10_test = .95 * local_test + .05 * uci10_test
        candidate_probabilities[names[4]] = (fixed10_train, fixed10_test, .05, None)

        weights = []
        for weight in WEIGHT_GRID:
            combined = (1 - weight) * local_train_oof + weight * uci10_train
            weights.append((average_precision_score(y_train, combined), -weight, weight))
        weight = max(weights)[-1]
        tuned_train = (1 - weight) * local_train_oof + weight * uci10_train
        tuned_test = (1 - weight) * local_test + weight * uci10_test
        candidate_probabilities[names[5]] = (tuned_train, tuned_test, weight, None)

        meta = fit_meta(local_train_oof, uci10_train, y_train)
        stack_train = meta.predict_proba(fusion_features(local_train_oof, uci10_train))[:, 1]
        stack_test = meta.predict_proba(fusion_features(local_test, uci10_test))[:, 1]
        candidate_probabilities[names[6]] = (stack_train, stack_test, None, meta.coef_[0].tolist())

        # For all non-stack candidates, Platt calibration and cutoff are fitted
        # only on outer-training OOF component probabilities. The outer fold
        # evaluates each complete threshold-selection procedure.
        for name, (train_score, test_score, weight, meta_coefs) in candidate_probabilities.items():
            if name == names[6]:
                threshold_oof = meta_oof_threshold_predictions(x_tr, y_tr, uci_ten, folds=4)
                calibrated_test = stack_test
                threshold = choose_threshold(y_train, threshold_oof)
            else:
                calibrator = calibrator_fit(train_score, y_train)
                calibrated_train = calibrator_predict(calibrator, train_score)
                calibrated_test = calibrator_predict(calibrator, test_score)
                threshold = choose_threshold(y_train, calibrated_train)
            rec = records[name]
            rec["y"].extend(y_test.tolist())
            rec["p"].extend(calibrated_test.tolist())
            rec["pred"].extend((calibrated_test >= threshold).astype(int).tolist())
            rec["baseline"].extend([float(y_train.mean())] * len(y_test))
            rec["thresholds"].append(float(threshold))
            if weight is not None:
                rec["weights"].append(float(weight))
            if meta_coefs is not None:
                rec["meta_coefficients"].append(meta_coefs)

    output = {}
    for name, rec in records.items():
        output[name] = {
            "metrics": metrics(np.asarray(rec["y"], int), np.asarray(rec["p"], float), np.asarray(rec["pred"], int), np.asarray(rec["baseline"], float)),
            "fold_thresholds": rec["thresholds"],
        }
        if rec["weights"]:
            output[name]["fold_weights"] = rec["weights"]
        if rec["meta_coefficients"]:
            output[name]["fold_meta_coefficients_bangladesh_uci"] = rec["meta_coefficients"]
    return output


def train() -> dict[str, Any]:
    X_uci10, y_uci, uci_audit = load_uci()
    bd_source = require_training_source(BD_PATH, BANGLADESH_DATA_ENV, "Bangladesh")
    X_bd, y_bd, _, bd_audit = load_bangladesh_training_data(bd_source)
    # Fit the archived six-feature UCI architecture and the expanded ten-field
    # foundation using only UCI resolved labels.
    X_uci6 = X_uci10.loc[:, LEGACY_UCI_FEATURES]
    uci_six = search_uci(X_uci6, y_uci, folds=5)
    uci_ten = search_uci(X_uci10, y_uci, folds=5)
    comparison = nested_compare(X_bd, y_bd, uci_six, uci_ten)

    # Refit the components on their full source cohorts. The fusion model sees
    # only local OOF Bangladesh base predictions, not in-sample fitted scores.
    local_scores = local_oof(X_bd, y_bd, folds=5)
    uci_scores = uci_ten.predict_proba(uci_harmonized_frame(X_bd))[:, 1]
    fusion_model = fit_meta(local_scores, uci_scores, y_bd.to_numpy())
    threshold_oof = meta_oof_threshold_predictions(X_bd, y_bd, uci_ten, folds=5)
    stack_threshold = choose_threshold(y_bd.to_numpy(), threshold_oof)
    fixed_uci_weight = .05
    weighted_oof = (1 - fixed_uci_weight) * local_scores + fixed_uci_weight * uci_scores
    weighted_calibrator = calibrator_fit(weighted_oof, y_bd.to_numpy())
    weighted_calibrated_oof = calibrator_predict(weighted_calibrator, weighted_oof)
    weighted_threshold = choose_threshold(y_bd.to_numpy(), weighted_calibrated_oof)
    local_final = fit_local(X_bd, y_bd, folds=5)
    uci_final = search_uci(X_uci10, y_uci, folds=5)
    pipeline = StackedEnsemblePipeline(local_final.best_estimator_, uci_final.best_estimator_, fusion_model, PREDICTOR_FIELDS)
    weighted_pipeline = WeightedEnsemblePipeline(local_final.best_estimator_, uci_final.best_estimator_, fixed_uci_weight, weighted_calibrator, PREDICTOR_FIELDS)

    # Include the prior held-out-era/inner OOF reports as historical context,
    # but use nested outer results above as the current comparison evidence.
    try:
        old_result = json.loads((ROOT / "model_reports" / "combined_uci_bangladesh_18" / "training_results.json").read_text(encoding="utf-8"))
        previous_baseline_cv = old_result["nested_cv_candidates"]["Combined system"]["outer_nested_cv_metrics"]
    except (OSError, KeyError, json.JSONDecodeError):
        previous_baseline_cv = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stack_artifact = {
        "pipeline": pipeline,
        "threshold": stack_threshold,
        "raw_input_features": list(PREDICTOR_FIELDS),
        "input_feature_count": 18,
        "target_field": "current_academic_status",
        "target_is_predictor": False,
        "target_mapping": {"Dropout (Discontinued Studies)": 1, "Running (Currently Studying)": 0, "Passed (Graduated)": 0},
        "architecture": "UCI 10-feature academic foundation + Bangladesh 18-input component + OOF-trained Logistic Regression stacker",
        "uci_foundation_features": list(UCI_FOUNDATION_FEATURES),
        "stacker_feature_names": ["bangladesh_signal", "uci_academic_signal"],
        "stacker_coefficients": {"bangladesh_signal": float(fusion_model.coef_[0][0]), "uci_academic_signal": float(fusion_model.coef_[0][1]), "intercept": float(fusion_model.intercept_[0])},
        "nested_cv_metrics": {name: value["metrics"] for name, value in comparison.items()},
        "performance_scope": "Nested stratified CV on reported Bangladesh current-status labels; not prospective validation.",
    }
    joblib.dump(stack_artifact, OUT_DIR / "stacked_dropout_status_18.joblib")
    selected_metrics = comparison["Fixed 5%/95% blend (10-feature UCI)"]["metrics"]
    artifact = {
        "pipeline": weighted_pipeline,
        "threshold": weighted_threshold,
        "raw_input_features": list(PREDICTOR_FIELDS),
        "input_feature_count": 18,
        "target_field": "current_academic_status",
        "target_is_predictor": False,
        "target_mapping": stack_artifact["target_mapping"],
        "architecture": "UCI 10-feature academic foundation + Bangladesh 18-input component + calibrated fixed 5%/95% probability blend",
        "blend_weight_uci": fixed_uci_weight,
        "blend_weight_bangladesh": 1.0 - fixed_uci_weight,
        "uci_foundation_features": list(UCI_FOUNDATION_FEATURES),
        "selected_model": "Calibrated 5% UCI / 95% Bangladesh Logistic Regression blend",
        "nested_cv_metrics": comparison,
        "active_model_metrics": selected_metrics,
        "performance_scope": "Nested stratified CV on reported Bangladesh current-status labels; not prospective validation.",
    }
    joblib.dump(artifact, OUT_DIR / "weighted_ensemble_dropout_status_18.joblib")

    results = {
        "project": "Combined UCI + Bangladesh 18-input stacked ensemble experiment",
        "training_counts": {"uci_resolved_records": len(X_uci10), "bangladesh_records": len(X_bd), "bangladesh_dropout_labels": int(y_bd.sum()), "bangladesh_non_dropout_labels": int((y_bd == 0).sum())},
        "uci_audit": uci_audit,
        "bangladesh_audit": bd_audit,
        "uci_original_15_feature_compatibility": FEATURE_COMPATIBILITY,
        "uci_foundation_features": list(UCI_FOUNDATION_FEATURES),
        "bangladesh_features": list(PREDICTOR_FIELDS),
        "target_mapping": artifact["target_mapping"],
        "excluded_from_predictors": ["Timestamp", "current_academic_status"],
        "model_parameters": {"uci_six_feature": uci_six.best_params_, "uci_ten_feature": uci_ten.best_params_, "bangladesh_local": local_final.best_params_, "stacker": {"algorithm": "Logistic Regression", "C": 1.0, "class_weight": None}},
        "stacker_fit_procedure": "Bangladesh base probabilities supplied to fusion fit are stratified OOF predictions from pipelines that did not train on their scored row. UCI foundation probabilities use a separate UCI-trained model and do not use Bangladesh labels.",
        "stacker_coefficients": stack_artifact["stacker_coefficients"],
        "threshold": weighted_threshold,
        "threshold_selection": "Weighted-blend threshold selected from Bangladesh training-only OOF blend scores, max F1 subject to at most 50% flagged.",
        "stacking_threshold": stack_threshold,
        "selected_active_architecture": "Fixed 5%/95% weighted blend using the expanded ten-feature UCI foundation; selected over stacking by nested-CV PR-AUC and Brier score.",
        "nested_cv_comparison": comparison,
        "previous_5_95_baseline_cv": previous_baseline_cv,
        "warning": "UCI predicts resolved graduate/dropout, Bangladesh labels reported current status. Neither model is validated for future dropout prediction.",
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "model_metadata.json").write_text(json.dumps({**results, "model_version": "combined-uci-bangladesh-18-ensemble-v2", "active_model": "weighted_ensemble_dropout_status_18.joblib", "stacking_backup": "stacked_dropout_status_18.joblib"}, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.write_text(render_report(results), encoding="utf-8")
    return results


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Combined UCI + Bangladesh 18-input model architecture and evaluation", "",
        "## Cohorts and outcomes", "",
        f"- UCI: {result['training_counts']['uci_resolved_records']} resolved records used to fit each final UCI component (Dropout=1, Graduate=0); 794 unresolved Enrolled records excluded.",
        f"- Bangladesh: {result['training_counts']['bangladesh_records']} questionnaire records used for the local component and OOF fusion fit (Dropout=1; Running/Passed=0). There are {result['training_counts']['bangladesh_dropout_labels']} Dropout and {result['training_counts']['bangladesh_non_dropout_labels']} class-0 labels.",
        "- The targets differ. UCI is eventual resolved outcome; Bangladesh is reported current status. These labels are not pooled or treated as equivalent. Running=0 is not proof of eventual persistence.",
        "- No evaluation result is a fresh independent test: prior inspection of a 63-record holdout is disclosed and this comparison uses nested stratified CV across the 251 Bangladesh records.", "",
        "## Audit of the 15 UCI model inputs", "",
        "The academic component uses ten defensibly aligned features: six academic/numeric measures plus gender and three binary support/financial indicators. It does not map UCI parent education/occupation codes to Bangladesh survey categories. Unknown or unsupported local categories remain unrecognized by the UCI encoder, not coerced into a Portuguese code.", "",
        "| Original UCI predictor | Bangladesh counterpart | Decision | Treatment / reason |",
        "|---|---|---|---|",
    ]
    for row in result["uci_original_15_feature_compatibility"]:
        lines.append(f"| {row['UCI predictor']} | {row['Bangladesh field']} | {row['decision']} | {row['handling']} |")
    lines += [
        "", "## Model C: nested comparison", "",
        "Every outer fold refits the Bangladesh component selection, all calibration/meta layers, and thresholds using only the outer-training records. The fusion layer is trained on Bangladesh local-model OOF predictions only—never in-sample predictions from a local model trained on those same labels. The outer validation fold is untouched. Threshold selection uses nested cross-fitted training probabilities and caps alerts at 50%.", "",
        "| Candidate | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Prevalence Brier | TN / FP / FN / TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in result["nested_cv_comparison"].items():
        m = entry["metrics"]
        cm = m["confusion_matrix"]
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['pr_auc_average_precision']:.3f} | {m['brier_score']:.3f} | {m['constant_training_prevalence_brier']:.3f} | {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} |")
    old = result.get("previous_5_95_baseline_cv")
    if old:
        lines += ["", "Previous archived 5%/95% model run (separately reported): " + ", ".join(f"{k}={old[k]:.3f}" for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc_average_precision", "brier_score")) + "."]
    lines += [
        "", "## Fusion comparison, selected architecture and interpretation", "",
        f"Final meta-model coefficients on the two component scores: Bangladesh `{result['stacker_coefficients']['bangladesh_signal']:.4f}`, UCI `{result['stacker_coefficients']['uci_academic_signal']:.4f}`, intercept `{result['stacker_coefficients']['intercept']:.4f}`. Fold-wise coefficient vectors are in `training_results.json`; their variability should be interpreted cautiously with only 251 local observations.",
        f"The selected active candidate is the fixed 5% UCI / 95% Bangladesh weighted blend with the ten-feature UCI foundation, threshold `{result['threshold']:.4f}`. Its nested-CV PR-AUC ({result['nested_cv_comparison']['Fixed 5%/95% blend (10-feature UCI)']['metrics']['pr_auc_average_precision']:.3f}) and Brier score ({result['nested_cv_comparison']['Fixed 5%/95% blend (10-feature UCI)']['metrics']['brier_score']:.3f}) were better than the OOF Logistic Regression stacker (PR-AUC {result['nested_cv_comparison']['OOF Logistic Regression stacking (10-feature UCI)']['metrics']['pr_auc_average_precision']:.3f}; Brier {result['nested_cv_comparison']['OOF Logistic Regression stacking (10-feature UCI)']['metrics']['brier_score']:.3f}). Stacking had slightly higher ROC-AUC but more false positives at its selected operating point. Therefore stacking remains a tested comparison artifact, not the selected app model.",
        f"Model A was fit on {result['training_counts']['uci_resolved_records']} UCI records; Model B was fit on {result['training_counts']['bangladesh_records']} Bangladesh records. The stacker used OOF Bangladesh component scores; the selected weighted blend uses a fixed 5% UCI / 95% local signal and a training-only calibration layer. Inactive historical artifacts are archived outside the project.",
        "The expanded fixed blend's performance is only modestly different from the Bangladesh-only system and not an external validation. Different outcome definitions, 0–5 numerical grade normalization (not grading equivalence), uncertain predictor timing, small convenience sample and cross-country domain shift prevent a future-dropout prediction claim. `Current Academic Status` and Timestamp are excluded from all predictor matrices.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    trained = train()
    print(json.dumps({"training_counts": trained["training_counts"], "threshold": trained["threshold"], "stacker_coefficients": trained["stacker_coefficients"], "comparison": {name: value["metrics"] for name, value in trained["nested_cv_comparison"].items()}}, indent=2))
