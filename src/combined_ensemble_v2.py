"""Inference and source-category handling for the UCI + Bangladesh ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.bangladesh_collection import PARENT_OCCUPATIONS


UCI_NUMERIC_FEATURES = (
    "hsc_equivalent_gpa", "age_at_enrollment", "courses_enrolled",
    "evaluations_completed", "courses_passed", "semester_gpa_native",
)
UCI_CATEGORICAL_FEATURES = (
    "gender_bangladesh", "scholarship_holder", "debtor", "tuition_fees_up_to_date",
)
UCI_FOUNDATION_FEATURES = (*UCI_NUMERIC_FEATURES, *UCI_CATEGORICAL_FEATURES)

# UCI occupation labels are retained as source-qualified choices in the shared
# assessment form. These are Portuguese job-role classifications, not local
# employment-sector/status categories. Code 99 is an unspecified value, so it
# is represented by the existing local unknown option rather than a job choice.
UCI_OCCUPATIONS = {
    0: "Student", 1: "Legislative/executive representatives, directors and managers",
    2: "Intellectual and scientific specialists", 3: "Intermediate-level technicians and professions",
    4: "Administrative staff", 5: "Personal services, security and sales workers",
    6: "Skilled agriculture, fisheries and forestry workers", 7: "Skilled industry, construction and craft workers",
    8: "Machine operators and assembly workers", 9: "Unskilled workers", 10: "Armed Forces",
    90: "Other situation", 101: "Armed Forces officers", 102: "Armed Forces sergeants",
    103: "Other Armed Forces personnel", 112: "Administrative and commercial services directors",
    114: "Hotel, catering, trade and other service directors", 121: "Physical sciences, mathematics and engineering specialists",
    122: "Health professionals", 123: "Teachers", 124: "Finance, administration and commercial specialists",
    125: "Information and communication technology specialists", 131: "Intermediate science and engineering technicians",
    132: "Intermediate health technicians", 134: "Intermediate legal, social, cultural and sports services",
    135: "Information and communication technology technicians", 141: "Office, secretarial and data-processing workers",
    143: "Data, accounting, statistics and financial operators", 144: "Other administrative support staff",
    151: "Personal service workers", 152: "Sellers", 153: "Personal care workers",
    154: "Protection and security personnel", 161: "Market-oriented farmers and skilled agricultural workers",
    163: "Subsistence farmers, fishers, hunters and gatherers", 171: "Skilled construction workers (except electricians)",
    172: "Skilled metallurgy and metalworking workers", 173: "Printing, precision, jewellery and craft workers",
    174: "Skilled electrical and electronics workers", 175: "Food, wood, clothing and other craft workers",
    181: "Fixed-plant and machine operators", 182: "Assembly workers", 183: "Vehicle drivers and mobile-equipment operators",
    191: "Cleaning workers", 192: "Unskilled agriculture, fisheries and forestry workers",
    193: "Unskilled extractive, construction, manufacturing and transport workers",
    194: "Meal preparation assistants", 195: "Street vendors and service providers",
}
UCI_OCCUPATION_PREFIX = "UCI source-only (not used by prediction) — "
UCI_BROAD_OCCUPATION_CODES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90)
UCI_OCCUPATION_BY_LABEL = {
    f"{UCI_OCCUPATION_PREFIX}{UCI_OCCUPATIONS[code]}": code
    for code in UCI_BROAD_OCCUPATION_CODES
}
UCI_DETAILED_OCCUPATION_TO_BROAD = {
    **{code: code for code in UCI_BROAD_OCCUPATION_CODES if code != 90},
    90: 90,
    101: 10, 102: 10, 103: 10,
    112: 1, 114: 1,
    121: 2, 122: 2, 123: 2, 124: 2, 125: 2,
    131: 3, 132: 3, 134: 3, 135: 3,
    141: 4, 143: 4, 144: 4,
    151: 5, 152: 5, 153: 5, 154: 5,
    161: 6, 163: 6,
    171: 7, 172: 7, 173: 7, 174: 7, 175: 7,
    181: 8, 182: 8, 183: 8,
    191: 9, 192: 9, 193: 9, 194: 9, 195: 9,
}
COMBINED_PARENT_OCCUPATION_CHOICES = (
    *PARENT_OCCUPATIONS,
    *(label for label in UCI_OCCUPATION_BY_LABEL if UCI_OCCUPATION_BY_LABEL[label] != 99),
)

UCI_MARITAL_TO_BANGLADESH = {
    1: "Single / Never married", 2: "Married", 3: "Widowed",
    4: "Divorced", 6: "Separated",
}


def parent_occupation_to_bangladesh(value):
    """Keep local survey labels; treat UCI-only and unrecognized choices as missing."""
    if value in PARENT_OCCUPATIONS:
        return value
    return pd.NA


def parent_occupation_to_uci(value):
    """Map a visibly UCI-qualified broad occupation choice to its UCI group code."""
    return UCI_OCCUPATION_BY_LABEL.get(value)


def uci_occupation_code_to_broad(code):
    """Collapse detailed source codes to their UCI-defined broad occupation group."""
    try:
        return UCI_DETAILED_OCCUPATION_TO_BROAD.get(int(float(code)))
    except (TypeError, ValueError):
        return None


def marital_status_to_bangladesh(code):
    """Map only unambiguous UCI marital labels; leave de-facto union unsupported."""
    try:
        return UCI_MARITAL_TO_BANGLADESH.get(int(float(code)), pd.NA)
    except (TypeError, ValueError):
        return pd.NA


def bangladesh_component_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Mask source-specific UCI occupation choices before local-model inference."""
    local = frame.loc[:, list(frame.columns)].copy()
    for field in ("mother_occupation_bangladesh", "father_occupation_bangladesh"):
        local[field] = local[field].map(parent_occupation_to_bangladesh).astype(object)
        local[field] = local[field].where(local[field].notna(), np.nan)
    return local


def calibrator_logit_feature(probability):
    """Shared training/inference transform for calibrated blended probabilities."""
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def uci_harmonized_frame(bangladesh_frame: pd.DataFrame) -> pd.DataFrame:
    """Project the 18 Bangladesh inputs onto ten defensibly shared UCI fields.

    Numeric grade ranges are normalized for model compatibility only. A
    Unsupported local categories are made explicitly missing for the UCI
    component rather than being coerced into a Portuguese value.
    """
    data = {field: pd.to_numeric(bangladesh_frame[field], errors="coerce") for field in UCI_NUMERIC_FEATURES}
    for field in UCI_CATEGORICAL_FEATURES:
        values = bangladesh_frame[field].astype(object)
        if field == "gender_bangladesh":
            values = values.where(values.isin(["Female", "Male"]), np.nan)
        else:
            values = values.where(values.isin(["Yes", "No"]), np.nan)
        data[field] = values
    return pd.DataFrame(data, index=bangladesh_frame.index).loc[:, UCI_FOUNDATION_FEATURES]


class StackedEnsemblePipeline:
    """Run both component models and fuse their OOF-trained score signals."""

    def __init__(self, bangladesh_pipeline, uci_pipeline, fusion_model, features):
        self.bangladesh_pipeline = bangladesh_pipeline
        self.uci_pipeline = uci_pipeline
        self.fusion_model = fusion_model
        self.feature_names_in_ = np.asarray(list(features), dtype=object)
        self.classes_ = np.asarray([0, 1])

    def component_probabilities(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Stacked ensemble requires the exact 18-input Bangladesh schema.")
        local = self.bangladesh_pipeline.predict_proba(bangladesh_component_frame(frame.loc[:, self.feature_names_in_]))[:, 1]
        uci = self.uci_pipeline.predict_proba(uci_harmonized_frame(frame))[:, 1]
        return local, uci

    def fusion_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        local, uci = self.component_probabilities(frame)
        return pd.DataFrame({"bangladesh_signal": local, "uci_academic_signal": uci}, index=frame.index)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Stacked ensemble requires the exact 18-input Bangladesh schema.")
        probability = self.fusion_model.predict_proba(self.fusion_features(frame))[:, 1]
        return np.column_stack([1.0 - probability, probability])


class WeightedEnsemblePipeline:
    """Calibrated fixed-weight blend using both fitted component models."""

    def __init__(self, bangladesh_pipeline, uci_pipeline, blend_weight, calibrator, features):
        self.bangladesh_pipeline = bangladesh_pipeline
        self.uci_pipeline = uci_pipeline
        self.blend_weight = float(blend_weight)
        self.blend_weight_uci = self.blend_weight
        self.calibrator = calibrator
        self.feature_names_in_ = np.asarray(list(features), dtype=object)
        self.classes_ = np.asarray([0, 1])

    def component_probabilities(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Weighted ensemble requires the exact 18-input Bangladesh schema.")
        local = self.bangladesh_pipeline.predict_proba(bangladesh_component_frame(frame.loc[:, self.feature_names_in_]))[:, 1]
        uci = self.uci_pipeline.predict_proba(uci_harmonized_frame(frame))[:, 1]
        return local, uci

    def blended_probability(self, frame: pd.DataFrame) -> np.ndarray:
        local, uci = self.component_probabilities(frame)
        return (1.0 - self.blend_weight) * local + self.blend_weight * uci

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Weighted ensemble requires the exact 18-input Bangladesh schema.")
        calibration_input = calibrator_logit_feature(self.blended_probability(frame))
        calibrated = self.calibrator.predict_proba(calibration_input.reshape(-1, 1))[:, 1]
        return np.column_stack([1.0 - calibrated, calibrated])
