# CarePredict-AI: 6-slide outline

**Team:** Rashmita, Akshata, Ananya, Deepika, Amrutha

## Slide 1 - Problem
- About 1 in 9 diabetic hospital encounters ends in a 30-day readmission (11.3%).
- Hospitals need to decide who gets follow-up calls, early appointments and extra discharge planning, with limited staff.
- CarePredict-AI ranks encounters by readmission risk to support (not replace) clinicians.
- Team names on the slide: Rashmita, Akshata, Ananya, Deepika, Amrutha.

## Slide 2 - Data and preparation
- UCI Diabetes 130-US Hospitals (1999-2008): 101,766 encounters, 71,518 patients.
- Excluded 1,652 expired encounters (dispositions 11, 19, 20) -> 100,114 encounters.
- Target: readmitted <30 days = 1 (11.34%), else 0.
- Missing values kept as explicit categories ("Unknown", "Not measured"); weight dropped (96.9% missing); ICD-9 codes grouped into 17 diagnosis categories.
- Patient-level split (no patient in train and test): 63,949 train / 16,090 validation / 20,075 test.

## Slide 3 - Approach
- Three models: Logistic Regression, Random Forest, XGBoost; class imbalance handled with class weights.
- Preprocessing fit on training data only; test set used once.
- Race and gender are NOT inputs (audit only); validation ablation showed no accuracy loss without them.
- Risk tiers from validation ranks: High = top 20%, Medium = next 30%, Low = rest.

## Slide 4 - Results (test set, High-risk cut-off)
| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.773 | 0.219 | 0.379 | 0.659 |
| Random Forest | 0.778 | 0.223 | 0.377 | 0.662 |
| **XGBoost (selected on validation)** | **0.781** | **0.231** | **0.388** | **0.667** |
- Random flagging of 20% would give precision 0.115; the model doubles that (2.0x).
- Observed readmission rate by tier (validation): Low 6.2%, Medium 12.7%, High 22.6%.
- Accuracy alone is misleading: predicting "no readmission" for everyone is 88.5% accurate and catches nobody.

## Slide 5 - Fairness and impact
- Audited by age, gender, race (recall, precision, FPR, FNR, selection rate, ROC-AUC).
- Gender: recall gap 0.002; race (groups with enough data): recall gap 0.041, selection-rate gap 0.008.
- Age shows the largest gap: recall gap 0.127 (higher in 30-40, lower in 60-70); groups under 500 patients not interpreted.
- Impact: focus follow-up on the top-risk 20%, who are readmitted at about twice the average rate.

## Slide 6 - Demo and future scope
- Live demo: FastAPI + HTML/JS app -> risk score, category, guidance, main factors, disclaimer.
- Limitations: decision support only, 1999-2008 data, discharge disposition assumed known, relative (not clinical) cut-offs.
- Future: tuning, engineered utilisation features, local validation, calibration monitoring, clinician feedback.
