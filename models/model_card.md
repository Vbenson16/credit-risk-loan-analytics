# Model Card: Loan Default Model (CRDA-RISK-2024)

## 1. Model details
- **Model:** XGBoost classifier, predicts the probability that a disbursed loan defaults.
- **Parameters:** learning_rate 0.01, max_depth 3, n_estimators 200, subsample 1.0, colsample_bytree 1.0, scale_pos_weight 3, random_state 42.
- **Features:** 54 columns (bureau, application, demographic and transaction features, one-hot encoded). No scaling is used for XGBoost.
- **Decision threshold:** 0.45.
- **Saved file:** `models/default_model.pkl` (model, column list and threshold).

## 2. Intended use
Decision support for underwriters and risk managers: a default probability, a 0-100 risk score and a Low / Medium / High band. It advises and does not replace the underwriter (PRD section 1).
Not for automatic approval or rejection, pricing, or credit-line decisions.

## 3. Data
- 5,000 customers, 5,000 credit records, 7,500 loan applications, 40,079 transactions.
- **Training population:** the 3,403 disbursed loans (default rate 23.4%), because the outcome is only known for them (PRD 13.1).
- **Split:** stratified 80/20, random_state 42: 2,722 training rows and 681 test rows (159 defaulters).
- **Excluded:** names, email and phone (personal data); `approval_status`, `approved_amount`, `decision_date` and `disbursed_flag` (known only after the decision).
- **Missing values:** none reached the model, so no imputation was needed.

## 4. Performance
**Test set, XGBoost at threshold 0.45**

| Metric | Result | PRD target |
|---|---|---|
| ROC-AUC | 0.699 | >= 0.80 (not met) |
| Recall | 0.686 | >= 0.70 (not met, 3 defaulters short) |
| Precision | 0.349 | none |
| Accuracy | 0.628 | none |
| F1 | 0.463 | none |

Confusion matrix: [[319, 203], [50, 109]].

**Model comparison**

| Model | CV ROC-AUC | CV threshold (recall about 0.73) | Test ROC-AUC (0.5 cut-off) |
|---|---|---|---|
| XGBoost | 0.7300 | 0.45 | 0.699 |
| Logistic Regression | 0.7357 | 0.44 | 0.703 |
| Random Forest | 0.7222 | 0.43 | 0.683 |
| Decision Tree | 0.6782 | 0.48 | 0.662 |

## 5. Why XGBoost
Logistic Regression and XGBoost tie on CV ROC-AUC (0.7357 vs 0.7300, a gap inside the noise). XGBoost gave higher precision at the same recall, and the PRD asks for XGBoost / Random Forest importances and SHAP (13.6). Logistic Regression is the runner-up and the simplest to explain.

## 6. Threshold
Chosen from 5-fold cross-validation on the training set only, as the highest threshold giving recall of at least 0.73, giving 0.45. The test set was then evaluated once at that threshold. (An earlier threshold of 0.4 had been picked by looking at test results; it was replaced by this method.)

## 7. Explainability
Top drivers (XGBoost importance and SHAP): debt-to-income ratio (0.105), credit score (0.076), applicant monthly income (0.059), Self-Employed (0.038) and total debit (0.037). See `models/feature_importance.csv` and `models/shap_summary.png`.

| Driver | SQL 7.10 | SHAP |
|---|---|---|
| Credit score < 600 | 40.1% default | #2, low score raises risk. Matches |
| Debt-to-income > 0.5 | 28.5% default | #1, high DTI raises risk. Matches |
| Utilization > 0.8 | 33.3% (36 loans) | #13, same direction, small group |
| Prior default | 33.1% (142 loans) | Not in the top 20. Only 142 disbursed loans have one, too few for the model to rely on |
| Employment type | not in 7.10 | Self-Employed raises risk, Salaried lowers it |

Interest rate acts as a stand-in for loan type, so a higher rate does not mean a safer loan.

## 8. Risk scoring (PRD 13.7)
- risk_score = round(P(default) x 100). Bands: 0-33 Low, 34-66 Medium, 67-100 High.
- 3,904 customers scored (the other 1,096 have no application). Each customer gets the highest probability across their applications.

| Band | Customers | Share | Actual default rate* |
|---|---|---|---|
| Low | 529 | 13.6% | 9.0% |
| Medium | 1,423 | 36.4% | 29.3% |
| High | 1,952 | 50.0% | 47.3% |

*Customers with a disbursed loan only (2,309). These include loans the model was trained on, so they are slightly optimistic. The default rate rises from Low to High, as it should.
High holds 50% of customers because scale_pos_weight=3 inflates probabilities. The cut-offs can be tuned to the business risk appetite (13.7).

## 9. Experiments that were tested and rejected
- **SMOTE:** test ROC-AUC 0.671 vs 0.699 for the champion.
- **Five extra features** (EMI burden, total late payments, debit share of income, low-score flag, prior-default flag): CV ROC-AUC 0.7335 vs 0.7346, no gain.
- **Split grouped by customer:** results were similar (XGBoost test ROC-AUC 0.711, CV 0.7334), so the PRD's stratified split was kept.

## 10. Limitations
- ROC-AUC 0.80 and recall 0.70 were not reached. Four algorithms plateau at 0.66 to 0.70, so the limit looks like the data.
- The test set was viewed more than once during development (PRD 13.4 asks for once).
- 55.9% of disbursed loans belong to customers with 2 or more loans, so train and test share customers.
- Transactions cover only the last 6 months and the bureau snapshot has one date, so some may be dated after the loan.
- 418 applications were dated before `customer_since`; tenure was set to 0.
- The model was trained on disbursed loans only and applied to all applications to score every customer.
- Precision is 0.349: about 2 in 3 flagged loans would not default.
- No fair-lending or bias review (a future enhancement in the PRD).