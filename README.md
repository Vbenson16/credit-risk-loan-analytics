## Credora Finance
This project builds a model that predicts whether a disbursed loan will default, based on applicant features such as debt-to-income ratio, credit score and monthly income. It turns the predicted probability into a risk score from 0 to 100, with Low, Medium and High risk bands, and saves a scored customer table for the dashboard.

## Folder structure
- `datasets/`: raw and cleaned CSVs (not committed, they contain personal data)
- `sql/`: schema and the 10 analytical queries
- `src/`: `data_load.py`, `features.py`, `train.py`, `risk_score.py`
- `models/`: `default_model.pkl`, `feature_importance.csv`, `shap_summary.png`, `model_card.md`
- `notebooks/`: the earlier pre-analysis, cleaning and modelling scripts, run from the project folder
- `outputs/`: `scored_customers.csv`
- `dashboard/`: Tableau dashboard (`credit_risk_dashboard.twbx`) and screenshots
- `documentation/`: PRD and `data_dictionary.md`
- `main.py`: runs the whole pipeline

## Setup
Python 3.11 or later.

    pip install -r requirements.txt

## Run
From the project folder:

1. Put the 4 raw CSVs in `datasets/` with these names: `customers (1).csv`, `credit_history.csv`, `loan_applications.csv`, `transactions.csv`.
2. Uncomment the 4 `to_csv` lines in `notebooks/Data Cleaning.py` and run it once to create the cleaned files.
3. Run the pipeline:

       python main.py

Takes about 4 minutes. Results are identical on every run (fixed random seeds).

## Database (SQL layer)
Before running `sql/credit_risk_platform.sql` in PostgreSQL 14+, replace `<path_to_project>` in section 6 with your project folder. The file creates the tables, loads the **raw** CSVs from `datasets/` (not the cleaned ones), and then runs the 10 analytical queries.

## Results (held-out test set)
Model: XGBoost, threshold 0.45.

| Metric | Result | PRD target |
|---|---|---|
| ROC-AUC | 0.699 | >= 0.80 (not met) |
| Recall | 0.686 | >= 0.70 (not met) |
| Precision | 0.349 | none set |
| Accuracy | 0.628 | none set |
| F1 | 0.463 | none set |

Four models were compared (Logistic Regression, Decision Tree, Random Forest,
XGBoost); all plateau at ROC-AUC 0.66 to 0.70. See `models/model_card.md`.

## Outputs
- `outputs/scored_customers.csv`: one row per customer (3,904 customers) with
  default probability, risk score, risk category and a 0/1 prediction.
  The other 1,096 customers never applied for a loan, so they have no score.
- `models/`: the saved model (with its column list and threshold), the ranked
  feature-importance table and the SHAP plot.
- `dashboard/credit_risk_dashboard.twbx`: Tableau dashboard with the 8 PRD components and filters (loan type, state, risk category, month). Open with Tableau Desktop or Tableau Public.

## Documentation
- Model card: `models/model_card.md`
- Data dictionary: `documentation/data_dictionary.md`