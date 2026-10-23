# Combined UCI + Bangladesh 18-input model architecture and evaluation

## Cohorts and outcomes

- UCI: 3630 resolved records used to fit each final UCI component (Dropout=1, Graduate=0); 794 unresolved Enrolled records excluded.
- Bangladesh: 251 questionnaire records used for the local component and OOF fusion fit (Dropout=1; Running/Passed=0). There are 52 Dropout and 199 class-0 labels.
- The targets differ. UCI is eventual resolved outcome; Bangladesh is reported current status. These labels are not pooled or treated as equivalent. Running=0 is not proof of eventual persistence.
- No evaluation result is a fresh independent test: prior inspection of a 63-record holdout is disclosed and this comparison uses nested stratified CV across the 251 Bangladesh records.

## Audit of the 15 UCI model inputs

The academic component uses ten defensibly aligned features: six academic/numeric measures plus gender and three binary support/financial indicators. It does not use parent occupation, parent education or marital status. During live prediction, its harmonizer retains only exact Female/Male gender and Yes/No binary support labels; unsupported/unknown local values are made missing rather than coerced into a Portuguese code.

The production assessment keeps 18 inputs and uses exactly the original Bangladesh parent-occupation choices, including `Not Alive` and `Housewife`; UCI-specific occupation options are not shown. The UCI academic component uses ten harmonized features and does not learn parental occupation categories. No Portuguese-to-Bangladesh occupation crosswalk is used. `Housewife` exists in the mother's fitted category vocabulary but not the father's, where it is treated as unseen by one-hot encoding. The Bangladesh research schema and stored historical responses are unchanged.

Other categorical review: UCI parent education contains Portugal-specific completed and incomplete year/technical/degree-cycle distinctions and has no defensible direct crosswalk to the local education brackets; no mapping is applied. UCI marital status has limited exact semantic overlap: codes 1/2/3/4/6 correspond respectively to local Single / Never married, Married, Widowed, Divorced and Separated. Code 5 (de-facto union) is unsupported. This documented crosswalk is not an active model feature mapping because the UCI foundation excludes marital status. Female/Male and Yes/No are mapped exactly; other or unknown local labels become missing for Model A. Definitions: [UCI dataset codebook](https://archive.ics.uci.edu/dataset/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess).

| Original UCI predictor | Bangladesh counterpart | Decision | Treatment / reason |
|---|---|---|---|
| Previous qualification (grade) | HSC / Equivalent GPA | Use | Linear numeric range alignment from UCI 0–190 to Bangladesh 0–5; no academic-system equivalence claimed. |
| Age at enrollment | Age at enrollment | Use | Direct numeric match. |
| Mother's occupation | Mother's occupation | Exclude from UCI | UCI job-role categories cannot be mapped to Bangladesh status/sector without invented equivalence. The local model receives the original Bangladesh category. |
| Father's occupation | Father's occupation | Exclude from UCI | UCI job-role categories cannot be mapped to Bangladesh status/sector without invented equivalence. The local model receives the original Bangladesh category. |
| Mother's qualification | Mother's highest education | Exclude | Detailed UCI education codes have no verified one-to-one Bangladesh mapping. |
| Father's qualification | Father's highest education | Exclude | Detailed UCI education codes have no verified one-to-one Bangladesh mapping. |
| Gender | Gender | Use | Map only documented UCI Female/Male values to matching labels; nonmatching local categories remain unseen, never coerced. |
| Marital status | Marital status | Exclude | Limited exact semantic crosswalk is documented, but the UCI component schema excludes it; de-facto union remains unsupported. |
| Scholarship holder | Scholarship holder | Use | Map UCI binary 0/1 to No/Yes. |
| Debtor | Have any debt | Use | Map UCI binary 0/1 to No/Yes. |
| Tuition fees up to date | Tuition fees up to date | Use | Map UCI binary 0/1 to No/Yes. |
| Curricular units 1st sem (enrolled) | Courses enrolled | Use | Direct numeric count; domain ranges may differ. |
| Curricular units 1st sem (evaluations) | Evaluations completed | Use | Direct numeric count; domain ranges may differ. |
| Curricular units 1st sem (grade) | Semester GPA | Use | Linear range alignment from observed UCI 0–18.875 to Bangladesh model 0–5; not grading-system equivalence. |
| Curricular units 1st sem (approved) | Courses passed | Use | Direct numeric count. |

## Model C: nested comparison

Every outer fold refits the Bangladesh component selection, all calibration/meta layers, and thresholds using only the outer-training records. The fusion layer is trained on Bangladesh local-model OOF predictions only—never in-sample predictions from a local model trained on those same labels. The outer validation fold is untouched. Threshold selection uses nested cross-fitted training probabilities and caps alerts at 50%.

| Candidate | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Prevalence Brier | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bangladesh-only 18-input | 0.972 | 0.961 | 0.925 | 0.942 | 0.933 | 0.983 | 0.933 | 0.032 | 0.164 | 195 / 4 / 3 / 49 |
| UCI academic foundation (6 numeric) | 0.904 | 0.847 | 0.780 | 0.750 | 0.765 | 0.915 | 0.750 | 0.100 | 0.164 | 188 / 11 / 13 / 39 |
| UCI academic foundation (10 aligned) | 0.908 | 0.829 | 0.837 | 0.692 | 0.758 | 0.939 | 0.810 | 0.073 | 0.164 | 192 / 7 / 16 / 36 |
| Archived 5%/95% weighted blend (6-feature UCI) | 0.972 | 0.961 | 0.925 | 0.942 | 0.933 | 0.985 | 0.911 | 0.032 | 0.164 | 195 / 4 / 3 / 49 |
| Fixed 5%/95% blend (10-feature UCI) | 0.964 | 0.956 | 0.891 | 0.942 | 0.916 | 0.986 | 0.935 | 0.032 | 0.164 | 193 / 6 / 3 / 49 |
| OOF-selected weighted blend (10-feature UCI) | 0.968 | 0.959 | 0.907 | 0.942 | 0.925 | 0.986 | 0.933 | 0.033 | 0.164 | 194 / 5 / 3 / 49 |
| OOF Logistic Regression stacking (10-feature UCI) | 0.952 | 0.949 | 0.845 | 0.942 | 0.891 | 0.987 | 0.919 | 0.037 | 0.164 | 190 / 9 / 3 / 49 |

Previous archived 5%/95% model run (separately reported): accuracy=0.972, balanced_accuracy=0.961, precision=0.925, recall=0.942, f1=0.933, roc_auc=0.978, pr_auc_average_precision=0.885, brier_score=0.045.

## Fusion comparison, selected architecture and interpretation

Final meta-model coefficients on the two component scores: Bangladesh `3.6121`, UCI `1.6659`, intercept `-3.2976`. Fold-wise coefficient vectors are in `training_results.json`; their variability should be interpreted cautiously with only 251 local observations.
The selected active candidate is the fixed 5% UCI / 95% Bangladesh weighted blend with the ten-feature UCI foundation, threshold `0.3040698561883773` (30.4% when displayed). The saved calibrator was fitted on `logit(clip(weighted_blend))`; runtime inference now applies the identical clipping bounds (`1e-6`, `1-1e-6`) and logit formula before calling it. The previous runtime path supplied the clipped raw blend and was inconsistent with training. This is an inference correction only; no model was retrained and the active artifact bytes remain unchanged. Its nested-CV PR-AUC (0.935) and Brier score (0.032) were better than the OOF Logistic Regression stacker (PR-AUC 0.919; Brier 0.037). Stacking had slightly higher ROC-AUC but more false positives at its selected operating point. Therefore stacking is a documented comparison, not the selected app model; its inactive artifact is archived outside the project.

No new occupation-expanded UCI candidate was trained. The source categories do not admit a meaningful mapping for use on Bangladeshi survey responses; adding raw UCI job roles to Model A would only provide values for the source's own records, while local responses would remain missing. That does not create a validated common occupation predictor. The app excludes these unused UCI-specific occupation choices rather than presenting them as harmonized features.
Model A was fit on 3630 UCI records; Model B was fit on 251 Bangladesh records. The stacker used OOF Bangladesh component scores; the selected weighted blend uses a fixed 5% UCI / 95% local signal and a training-only calibration layer. The original six-feature blend artifact is archived outside the project; its evaluation summary is retained above for historical comparison.
The expanded fixed blend's performance is only modestly different from the Bangladesh-only system and not an external validation. Different outcome definitions, 0–5 numerical grade normalization (not grading equivalence), uncertain predictor timing, small convenience sample and cross-country domain shift prevent a future-dropout prediction claim. `Current Academic Status` and Timestamp are excluded from all predictor matrices.

## Calibration and blend-weight verification (2026-10-22)

The active artifact was checked directly with PowerShell `Get-FileHash -Algorithm SHA256`; its SHA-256 is `D9F4D5663002E3C17482E57EE5E50E97038CFEDB3B987046ECB0F652CD4DBA06`, matching the previously published hash. The differing hash mentioned during review is not present in this report or the artifact's metadata, and could not be reproduced against the active artifact. A hash identifies exact file bytes, so any different value likely referred to a different file/version or was transcribed incorrectly. No model file was changed.

The regression suite now checks that runtime calibration applies the same `clip(p, 1e-6, 1-1e-6)` followed by `log(p / (1-p))` transform as the training calibration function. A component-score example (`UCI=0.179`, `Bangladesh=0.076`) produces a raw 5%/95% blend of `0.08115`; the active fitted calibrator maps that to approximately `0.06064`. This illustrates why the final score need not equal the raw blend, and confirms the corrected inference path uses the fitted transformation rather than forcing a raw blend result.

### Leakage-safe Bangladesh validation sensitivity

For this re-check, each outer stratified fold withheld Bangladesh outcomes from local-model fitting, calibration, and threshold selection. The local component and calibration/threshold were fit using only the outer training partition; the UCI foundation was independently trained on its 3,630 source records. All variants use the same five-fold stratified partitions and 251 Bangladesh responses (52 reported dropouts, 199 non-dropouts). No exact duplicate predictor rows were found. Since the questionnaire has no stable participant identifier, repeated submissions by the same person cannot be ruled out. The threshold is selected within training-only predictions for each outer fold. Consequently, these pooled cross-validation metrics are comparison evidence, not an untouched external test estimate.

| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Confusion matrix (TN, FP, FN, TP) | Flagged |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Bangladesh-only local model | 0.9721 | 0.9245 | 0.9423 | 0.9333 | 0.9826 | 0.9333 | 0.03238 | (195, 4, 3, 49) | 53 (21.1%) |
| Corrected 5% UCI / 95% Bangladesh blend | 0.9641 | 0.8909 | 0.9423 | 0.9159 | 0.9863 | 0.9349 | 0.03235 | (193, 6, 3, 49) | 55 (21.9%) |
| 30% UCI / 70% Bangladesh blend | 0.9602 | 0.8750 | 0.9423 | 0.9074 | 0.9876 | 0.9218 | 0.03554 | (192, 7, 3, 49) | 56 (22.3%) |
| 50% UCI / 50% Bangladesh blend | 0.9363 | 0.8333 | 0.8654 | 0.8491 | 0.9850 | 0.9071 | 0.04194 | (190, 9, 7, 45) | 54 (21.5%) |
| OOF Logistic Regression stacker | 0.9522 | 0.8448 | 0.9423 | 0.8909 | 0.9866 | 0.9191 | 0.03679 | (190, 9, 3, 49) | 58 (23.1%) |

The Brier score of a constant Bangladesh-prevalence prediction is `0.16430`; every candidate improves on that uncalibrated prevalence baseline, with the 5% blend marginally lowest at `0.03235` and Bangladesh-only at `0.03238`. The active blend's ranking improvement over Bangladesh-only is small (+0.0037 ROC-AUC and +0.0015 PR-AUC), while the local-only model has better precision, F1, accuracy, and four rather than six false positives at its training-selected operating point. Raising the UCI weight to 30% or 50% does not improve the PR-AUC/Brier trade-off; 50/50 also misses four additional dropout cases. The stacker recovers the same 49 of 52 cases as the 5% blend but flags three more students and has worse PR-AUC and Brier score.

The weighted blend's pre-calibration component coefficients are explicitly 5% UCI and 95% Bangladesh (30/70 and 50/50 candidates use those respective shares); the final calibrated probability is a nonlinear transform, so those shares are not additive shares of the displayed final score. The stacker's fitted coefficients are Bangladesh `3.6121`, UCI `1.6659`, intercept `-3.2976`; they are logistic score coefficients, not percentages. This evaluation does not support increasing the UCI contribution merely because its source cohort is larger. For a project that requires a hybrid architecture, retain 5%/95% as the most defensible tested blend: it offers a very small ranking/calibration gain over local-only and outperforms other hybrid candidates on PR-AUC and Brier. Keep Bangladesh-only as an important comparator; no evidence establishes the blend as a reliable future-dropout predictor.

### Occupation dropdown effect

The assessment dropdowns now present exactly the Bangladesh parent-occupation categories, preserving the original response labels including `Not Alive` and `Housewife`. UCI source-specific occupation groups are excluded from the form because neither trained component uses them. The UCI academic component uses ten harmonized academic/demographic features and does not learn parental occupations. The Bangladesh mother's occupation encoder recognizes `Business owner`, `Government employee`, `Housewife`, `Not Alive`, `Other`, `Private-sector employee`, `Retired`, and `Self-employed / Daily-wage worker`; the father's encoder recognizes the same list except `Housewife`. `Unknown / Prefer not to say` is interpreted as missing and imputed, not as a learned category. Therefore father `Housewife` and explicit unknown choices have no category-specific learned effect, although they remain valid questionnaire/form responses. Historical Bangladesh response labels remain unchanged. UCI's numeric occupational taxonomy is source-specific and does not support defensible sector/survival mappings to Bangladesh categories; see the [official UCI dataset documentation](https://archive.ics.uci.edu/dataset/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess).
