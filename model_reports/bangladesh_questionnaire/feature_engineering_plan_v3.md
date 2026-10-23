# Bangladesh Questionnaire Feature and Target Schema (v3.0.0)

## Scope and separation

The form has exactly 19 answerable questions: the existing 18 Bangladesh predictors in their agreed order, followed by the required `Current Academic Status` dropdown. It has no consent, ID, university name, date, email, or other research-metadata question. Any required consent and governance process must be handled separately.

This survey schema does not alter or score with the production UCI 15-input Random Forest. Bangladesh occupation and education values remain local categories and are not converted to UCI codes. The current-status field is never included in a model feature matrix.

## Predictor fields and feature sets

The first 18 questions and their order are defined in `data/bangladesh_questionnaire/v3/data_dictionary_v3.json`. The importer exports all 18 features plus the status target as the 19th column. `future_comparison_feature_sets()` explicitly selects predictor columns only, so `current_academic_status` cannot leak into the input matrix.

The helper feature sets are preparation groups only: `shared_academic` has five broad academic features, `shared_core` has 13 concepts, and `bangladesh_expanded` has all 18 Bangladesh predictor fields. Similar names do not establish measurement equivalence across countries or universities. No GPA scale conversion is performed.

## Requested target coding and limitation

`Current Academic Status` is required and accepts:

- `Running (Currently Studying)` → 0
- `Passed (Graduated)` → 0
- `Dropout (Discontinued Studies)` → 1

This implements the requested binary target convention. It describes status observed when the form is submitted. In particular, `Running = 0` does not mean the student is a confirmed graduate or will never discontinue. A model trained against this label would be retrospective current-status classification unless a separate future outcome and time horizon were defined and verified. Do not describe this mapping as evidence of future dropout prediction.

## Import and missing-data handling

The importer accepts either Google Forms question titles or canonical column names, ignores incidental Google metadata such as Timestamp, and emits exactly 19 columns in question order. Missing predictor headers produce warnings and blank columns. The target header and target value are required and missing values or unsupported categories are errors. Optional predictor blanks and `Unknown / Prefer not to say` / `Undisclosed` answers stay missing. No imputation occurs. Numeric ranges and course-count consistency are validated.

The income brackets (Poor below ৳25,000; Below Average ৳25,000–39,999; Average ৳40,000–69,999; Higher Middle Class ৳70,000–149,999; Rich ৳150,000+; Undisclosed) are provisional survey categories, not official poverty definitions.

## Form update and response safety

`updateExistingBangladeshResearchForm()` opens the existing configured form. It checks the existing response count before any mutation. If responses exist, it stops and leaves the form unchanged; archive those responses before requesting another migration approach. If none exist, it turns off email collection, replaces old items with exactly 19 questions, verifies no extra sections, and leaves response collection off. It does not create a form or spreadsheet.

No Bangladesh model is trained by this change. Before any training, review the target's status timing, establish consent and governance, define a defensible prediction point and horizon, obtain verified outcomes, and evaluate with appropriate grouped/temporal splits.
