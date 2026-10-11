# Limitations (say these out loud in the demo)

1. **Decision support only.** Not a diagnosis; clinicians decide.
2. **Moderate accuracy of ranking.** Test ROC-AUC about 0.67 (typical for this public dataset). The model ranks risk better than chance but is far from perfect: at the High-risk cut-off, about 23% of flagged encounters are readmitted within 30 days (precision) and about 39% of all readmissions are caught (recall).
3. **Discharge disposition is a feature.** The tool assumes disposition is known at discharge planning. If a prediction were needed earlier in the stay, this feature could leak information (D4, R8).
4. **Old data (1999-2008), 130 US hospitals.** Practice has changed; needs local validation before any real use.
5. **Risk tiers are relative.** High/Medium/Low are rank-based on validation data (top 20% / next 30% / rest), not clinically validated thresholds (D8/R5).
6. **Fairness.** Race and gender are not inputs, but proxies (payer_code, medical_specialty, diagnoses) can still carry information. Recall differs noticeably by age (see reports/fairness_disparity_summary.csv). Groups with n < 500 or fewer than 30 positives (e.g. ages 0-30 and 90+, Asian, Hispanic, Other, Unknown race) are shown but not interpreted.
7. **Hospice discharges (codes 13, 14, 21) are retained** (R2); these patients are unlikely to be readmitted.
8. **Unused signals.** No hyper-parameter search (deadline); engineered features F1, F2, F4 not implemented (R4).
9. **Explanations show association, not causation.**
