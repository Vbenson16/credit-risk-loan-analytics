import os
import numpy as np
import pandas as pd


def get_risk_category(score):
    if score <= 33:
        return "Low Risk"
    elif score <= 66:
        return "Medium Risk"
    else:
        return "High Risk"


def score_customers(mrg_df, xgb_model, x_train, x_test, xgb_proba, customer_ids_all, y, threshold):

    score_df = mrg_df.copy()
    score_ids = score_df["customer_id"].copy()

    score_df = pd.get_dummies(score_df, columns=["employment_type", "residence_type", "loan_type", "education"], dtype=int)

# columns the model needs but this table lacks (should print [])
    print("Missing columns:", [c for c in x_train.columns if c not in score_df.columns])

    x_score = score_df.reindex(columns=x_train.columns, fill_value=0)
    score_proba = xgb_model.predict_proba(x_score)[:, 1]

# check: test rows must match xgb_proba
    check = pd.Series(score_proba, index=score_df.index).loc[x_test.index]
    print("Matches xgb_proba:", np.allclose(check.values, xgb_proba))

# one row per customer, highest probability
    app_level = pd.DataFrame({"customer_id": score_ids.values, "default_probability": score_proba})
    scored_customers = app_level.groupby("customer_id", as_index=False)["default_probability"].max()

    print(scored_customers.shape)
    print(scored_customers.head())
    print(scored_customers.isnull().sum())

    scored_customers["risk_score"] = np.round(scored_customers["default_probability"] * 100).astype(int)
    scored_customers["risk_category"] = scored_customers["risk_score"].apply(get_risk_category)
    scored_customers["predicted_default"] = (scored_customers["default_probability"] >= threshold).astype(int)

# band sizes (count and %)
    print(scored_customers["risk_category"].value_counts())
    print((scored_customers["risk_category"].value_counts(normalize=True) * 100).round(1))

# actual default rate per band (customers with a disbursed loan only)
    outcome = pd.DataFrame({"customer_id": customer_ids_all.values, "loan_default": y.values})
    outcome = outcome.groupby("customer_id", as_index=False)["loan_default"].max()
    chk = scored_customers.merge(outcome, on="customer_id", how="inner")
    print(chk.groupby("risk_category")["loan_default"].agg(["count", "mean"]).round(3))

    os.makedirs("outputs", exist_ok=True)
    scored_customers.to_csv("outputs/scored_customers.csv", index=False)

    return scored_customers