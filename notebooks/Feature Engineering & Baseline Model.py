import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import shap
import os
#from imblearn.over_sampling import SMOTE


#### Data Loading ####
cus_df_cln= pd.read_csv("datasets/cleaned_customers.csv")
crd_his_df_cln= pd.read_csv("datasets/cleaned_credit_history.csv")
lon_ap_df_cln= pd.read_csv("datasets/cleaned_loan_applications.csv")
trans_df_cln= pd.read_csv("datasets/cleaned_transactions.csv")


#### Transaction Aggregation ####
#combine 2 datasets
cus_trans_agg= pd.merge(cus_df_cln, trans_df_cln, on="customer_id", how="left")

summary_df= cus_trans_agg.groupby("customer_id").agg( 
    avg_bal= pd.NamedAgg(column="balance_after", aggfunc="mean"),
    trns_count= pd.NamedAgg(column="transaction_type", aggfunc="count"))


total_credit=cus_trans_agg[cus_trans_agg["transaction_type"] == 1].groupby("customer_id")["amount"].sum()
total_debit=cus_trans_agg[cus_trans_agg["transaction_type"] == 0].groupby("customer_id")["amount"].sum()

summary_df["total_credit"]= total_credit
summary_df["total_debit"]= total_debit

summary_df["total_credit"]= summary_df["total_credit"].fillna(0)
summary_df["total_debit"]= summary_df["total_debit"].fillna(0)

summary_df["net_cash_flow"]= summary_df["total_credit"] - summary_df["total_debit"]
# Step 1: count the problem cases
inf_cnt = ((summary_df["total_debit"] == 0) & (summary_df["total_credit"] > 0)).sum()
both_zero = ((summary_df["total_debit"] == 0) & (summary_df["total_credit"] == 0)).sum()
#print("Outflow 0, inflow > 0:", inf_cnt)
#   print("Both 0:", both_zero)

# Step 2: calculate the ratio and fix the problem cases
ratio = summary_df["total_credit"] / summary_df["total_debit"]
ratio = ratio.replace([np.inf, -np.inf], np.nan)

cap = ratio.quantile(0.99)   # 99th percentile of the normal ratios
ratio[(summary_df["total_debit"] == 0) & (summary_df["total_credit"] > 0)] = cap
ratio = ratio.fillna(0)      # both zero = no activity

summary_df["inflow_outflow_ratio"] = ratio

# Step 3: check
#print(summary_df["inflow_outflow_ratio"].isnull().sum())
#print(np.isinf(summary_df["inflow_outflow_ratio"]).sum())


emi_outflow= cus_trans_agg[cus_trans_agg["category"] == "EMI"].groupby("customer_id")["amount"].sum()
summary_df["emi_outflow"]= emi_outflow
summary_df["emi_outflow"]= summary_df["emi_outflow"].fillna(0)


# checking for duplicates in the summary_df
#print(summary_df["trns_count"].sum())
#print(cus_df_cln['customer_id'].duplicated().sum())
#print(crd_his_df_cln['customer_id'].duplicated().sum())
#print(summary_df['customer_id'].duplicated().sum())
#print(np.isinf(summary_df["inflow_outflow_ratio"]).sum())

# merging the datasets
mrg_df = lon_ap_df_cln.merge(cus_df_cln, on='customer_id', how='left')
mrg_df = mrg_df.merge(crd_his_df_cln, on='customer_id', how='left')
mrg_df = mrg_df.merge(summary_df, on='customer_id', how='left')
#print(mrg_df.shape)


# Creating new features 
debt_to_lmt_ratio= mrg_df["total_outstanding_debt"]/mrg_df["total_credit_limit"]
ln_to_incm_ratio= mrg_df["requested_amount"]/mrg_df["annual_income"]
mrg_df["debt_to_lmt_ratio"]= debt_to_lmt_ratio
mrg_df["ln_to_incm_ratio"]= ln_to_incm_ratio

# change data type of customer_since from str to datetime
mrg_df["customer_since"]= pd.to_datetime(mrg_df["customer_since"], errors= "coerce")
mrg_df["application_date"]= pd.to_datetime(mrg_df["application_date"], errors= "coerce") 

tenure = (mrg_df["application_date"] - mrg_df["customer_since"]).dt.days / 365
mrg_df["tenure"]= tenure

# the fix
mrg_df["tenure"] = mrg_df["tenure"].clip(lower=0)

## 418 apps dated before customer_since; set to 0.
# filtering the merged dataset to include only rows where disbursed_flag is 1
filtered_mrg_df = mrg_df[(mrg_df["disbursed_flag"]==1)] 


#### One Hot encoding ####
# One-hot encoding for categorical variables
filtered_mrg_df = pd.get_dummies(filtered_mrg_df, columns=["employment_type", "residence_type", "loan_type", "education"], dtype=int)


# leakage control: dropping columns that may cause data leakage
filtered_mrg_df= filtered_mrg_df.drop(columns=["approval_status", "approved_amount", "decision_date", "disbursed_flag", "email", "phone"]) # email and phone are not relevant for model trainingand are personal data.

# Define the specific columns to standardize
std_clmns= ["requested_amount", "loan_term_months", "interest_rate", "applicant_monthly_income","existing_emi",
            "debt_to_income_ratio", "age", "dependents", "annual_income",
            "years_at_residence", "credit_score", "num_open_accounts", "num_credit_inquiries_6m",
            "credit_utilization_ratio", "num_late_payments_30d", "num_late_payments_90d",
            "num_defaults_prior", "total_credit_limit", "total_outstanding_debt",
            "oldest_account_age_months", "avg_bal", "trns_count", "total_credit", "total_debit",
            "net_cash_flow", "inflow_outflow_ratio", "emi_outflow", "debt_to_lmt_ratio", "ln_to_incm_ratio", "tenure"]


# 80/20 stratified train/test on loan default
y= filtered_mrg_df["loan_default"]

# Check and identify string columns in the filtered_mrg_df
#print((filtered_mrg_df.dtypes=="str").value_counts())
#print(filtered_mrg_df.select_dtypes(include=["str"]).columns.tolist())
#print(filtered_mrg_df['num_defaults_prior'].value_counts())


# Save customer_id before it gets dropped
customer_ids_all = filtered_mrg_df["customer_id"]


# dropping columns with string and datetime types
filtered_mrg_df = filtered_mrg_df.drop(columns=filtered_mrg_df.select_dtypes(include=["str", "datetime64[ns]"]).columns.tolist())
filtered_mrg_df= filtered_mrg_df.drop(columns=["loan_default"])
x= filtered_mrg_df


# Split the data into training and testing sets with stratification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# Initialize the StandardScaler
scaler = StandardScaler()
x_train_sc = x_train.copy()
x_test_sc = x_test.copy()

# Fit on training data only, transform both
x_train_sc[std_clmns] = scaler.fit_transform(x_train[std_clmns])
x_test_sc[std_clmns] = scaler.transform(x_test[std_clmns])


# Check the distribution of the target variable in both training and testing sets
#print(y_train.value_counts(normalize=True))
#print(y_test.value_counts(normalize=True))

# logistic regression model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(x_train_sc, y_train)

# Model Prediction
xtest_pred = model.predict(x_test_sc)
xtest_pred_prob = model.predict_proba(x_test_sc)[:, 1]

# computing the 6 mertics for logistic regression model
cm = confusion_matrix(y_test, xtest_pred)
acc = accuracy_score(y_test, xtest_pred)
pre = precision_score(y_test, xtest_pred)
recall = recall_score(y_test, xtest_pred)
f1 = f1_score(y_test, xtest_pred)
roc_auc = roc_auc_score(y_test, xtest_pred_prob)


print("Logistic Regression Model Metrics:")
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {acc}")
print(f"Precision: {pre}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")

# decision Tree model
dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=5)
dt_model.fit(x_train, y_train)

# check
#print(dt_model.predict(x_test)[:10])

# dt model predictions
dt_xtest_pred = dt_model.predict(x_test)
dt_xtest_pred_prob = dt_model.predict_proba(x_test)[:, 1]

# computing the 6 metrics for decision tree model
dt_cm = confusion_matrix(y_test, dt_xtest_pred)
dt_acc = accuracy_score(y_test, dt_xtest_pred)
dt_pre = precision_score(y_test, dt_xtest_pred)
dt_recall = recall_score(y_test, dt_xtest_pred)
dt_f1 = f1_score(y_test, dt_xtest_pred)
dt_roc_auc = roc_auc_score(y_test, dt_xtest_pred_prob)


print("\nDecision Tree Model Metrics:")
print(f"Confusion Matrix:\n{dt_cm}")
print(f"Accuracy: {dt_acc}")
print(f"Precision: {dt_pre}")
print(f"Recall: {dt_recall}")
print(f"F1 Score: {dt_f1}")
print(f"ROC AUC: {dt_roc_auc}")


# Making Random Forest model
rf_model = RandomForestClassifier(class_weight= "balanced", random_state=42)

# defining hyperparameters for Random Forest
rf_param= {
    'n_estimators': [300, 400, 500],
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [4, 6, 8]
}

# set up grid search for Random Forest
rf_grid= GridSearchCV(estimator=rf_model, param_grid=rf_param, cv=5, scoring='roc_auc', n_jobs=-1)


# fit the Random Forest model
rf_grid.fit(x_train, y_train)


# Making XGBoost model
xgb_model = XGBClassifier(scale_pos_weight=3, random_state=42)


# defining hyperparameters for XGBoost
xgb_param= {
    'n_estimators': [200, 300, 400],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.005, 0.01, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# set up grid search for XGBoost
xgb_grid= GridSearchCV(estimator=xgb_model, param_grid=xgb_param, cv=5, scoring='roc_auc', n_jobs=-1)

# fit the XGBoost model
xgb_grid.fit(x_train, y_train)


# Checking the performance of Random Forest and XGBoost models
# Random Forest
rf_best = rf_grid.best_estimator_
rf_preds = rf_best.predict(x_test)
rf_proba = rf_best.predict_proba(x_test)[:, 1]

print("Random Forest Model Metrics:")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, rf_preds)}")
print(f"Accuracy: {accuracy_score(y_test, rf_preds)}")
print(f"Precision: {precision_score(y_test, rf_preds)}")
print(f"Recall: {recall_score(y_test, rf_preds)}")
print(f"F1 Score: {f1_score(y_test, rf_preds)}")
print(f"ROC AUC: {roc_auc_score(y_test, rf_proba)}")

# XGBoost
xgb_best = xgb_grid.best_estimator_
xgb_preds = xgb_best.predict(x_test)
xgb_proba = xgb_best.predict_proba(x_test)[:, 1]

print("\nXGBoost Model Metrics:")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, xgb_preds)}")
print(f"Accuracy: {accuracy_score(y_test, xgb_preds)}")
print(f"Precision: {precision_score(y_test, xgb_preds)}")
print(f"Recall: {recall_score(y_test, xgb_preds)}")
print(f"F1 Score: {f1_score(y_test, xgb_preds)}")
print(f"ROC AUC: {roc_auc_score(y_test, xgb_proba)}")

print("Random Forest Best Params:", rf_grid.best_params_)
print("XGBoost Best Params:", xgb_grid.best_params_)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# each model with the data it was trained on
models = {
    "Logistic Regression": (model, x_train_sc),
    "Decision Tree": (dt_model, x_train),
    "Random Forest": (rf_best, x_train),
    "XGBoost": (xgb_best, x_train),
}

target_recall = 0.73
rows = []

for name, (mdl, X) in models.items():
    # cross-validated probabilities on the TRAINING set only
    proba = cross_val_predict(mdl, X, y_train, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y_train, proba)

    # highest threshold that still gives recall >= 0.73
    best_thr = 0.05
    for thr in np.arange(0.90, 0.04, -0.01):
        pred = (proba >= thr).astype(int)
        if recall_score(y_train, pred) >= target_recall:
            best_thr = round(thr, 2)
            break

    pred = (proba >= best_thr).astype(int)
    rows.append({
        "Model": name,
        "CV ROC-AUC": round(auc, 4),
        "Threshold": best_thr,
        "Recall": round(recall_score(y_train, pred), 3),
        "Precision": round(precision_score(y_train, pred), 3),
    })

cv_table = pd.DataFrame(rows).set_index("Model")
print(cv_table.sort_values("Precision", ascending=False))



final_thr = 0.45   # chosen from the training-set cross-validation

final_pred = (xgb_proba >= final_thr).astype(int)

print("XGBoost final test result @ threshold", final_thr)
print(confusion_matrix(y_test, final_pred))
print("Accuracy:", accuracy_score(y_test, final_pred))
print("Precision:", precision_score(y_test, final_pred))
print("Recall:", recall_score(y_test, final_pred))
print("F1:", f1_score(y_test, final_pred))
print("ROC AUC:", roc_auc_score(y_test, xgb_proba))






# Pulling feature importances from the best Random Forest model
#rf_feature_importances = pd.DataFrame({'feature': x_train.columns, 'importance': rf_best.feature_importances_})
#rf_feature_importances = rf_feature_importances.sort_values(by='importance', ascending=False)
#print("\nTop Feature Importances from Random Forest:")
#print(rf_feature_importances.head())

# Pulling feature importances from the best XGBoost model
xgb_feature_importances = pd.DataFrame({'feature': x_train.columns, 'importance': xgb_best.feature_importances_})
xgb_feature_importances = xgb_feature_importances.sort_values(by='importance', ascending=False)
print("\nTop Feature Importances from XGBoost:")
print(xgb_feature_importances.head())

os.makedirs("models", exist_ok=True)
#######xgb_feature_importances.to_csv("models/feature_importance.csv", index=False)

# bottom feature importances random forest and xgboost
#print("\nBottom Feature Importances from Random Forest:")
#print(rf_feature_importances.tail(10))

#print("\nBottom Feature Importances from XGBoost:")
#print(xgb_feature_importances.tail(10))


##print(y_train_smt.value_counts())

#Train XGBoost on SMOTE-resampled data
##xgb_param_smt= {'n_estimators': 200,
##    'max_depth': 3,
##    'learning_rate': 0.01,
##    'subsample': 1.0,
##    'colsample_bytree': 1.0}


##xgb_model_smt = XGBClassifier(**xgb_param_smt, random_state=42)
##xgb_model_smt.fit(x_train_smt, y_train_smt)

# 3. Predict on ORIGINAL untouched test set
##xgb_proba_smt = xgb_model_smt.predict_proba(x_test)[:, 1]
##xgb_pred_smt = (xgb_proba_smt >= 0.4).astype(int)  # same threshold as champion

# 4. Evaluate
##roc_auc_smt = roc_auc_score(y_test, xgb_proba_smt)
##recall_smt = recall_score(y_test, xgb_pred_smt)

##print("SMOTE Model ROC-AUC:", roc_auc_smt)
##print("SMOTE Model Recall:", recall_smt)

# 5. Compare against current champion (already known)
##print("Champion ROC-AUC:", roc_auc_score(y_test, xgb_proba))
##print("Champion Recall @", final_thr, ":", recall_score(y_test, final_pred))

# SMOTE experiment (comparison only)
##smt = SMOTE(random_state=42)
##x_train_smt, y_train_smt = smt.fit_resample(x_train_sc, y_train)
##print(y_train_smt.value_counts())

# same best parameters as the champion, trained on the SMOTE data
###xgb_model_smt = XGBClassifier(**xgb_grid.best_params_, random_state=42)
###xgb_model_smt.fit(x_train_smt, y_train_smt)

# test on the original, untouched test set
###xgb_proba_smt = xgb_model_smt.predict_proba(x_test_sc)[:, 1]
###xgb_pred_smt = (xgb_proba_smt >= final_thr).astype(int)

###print("SMOTE ROC-AUC:", roc_auc_score(y_test, xgb_proba_smt))
###print("SMOTE Recall:", recall_score(y_test, xgb_pred_smt))
###print("SMOTE Precision:", precision_score(y_test, xgb_pred_smt))

###print("Champion ROC-AUC:", roc_auc_score(y_test, xgb_proba))
###print("Champion Recall:", recall_score(y_test, final_pred))
###print("Champion Precision:", precision_score(y_test, final_pred))

# SHAP Analysis for XGBoost
explner = shap.TreeExplainer(xgb_best)
shap_values = explner.shap_values(x_test)
print("SHAP Summary Plot for XGBoost:")
shap.summary_plot(shap_values, x_test, show=False)
plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()

def get_risk_category(score):
    if score <= 33:
        return "Low Risk"
    elif score <= 66:
        return "Medium Risk"
    else:
        return "High Risk"
    

#print(conv_score[:10])
#print(risk_category[:10])       



# Now derive risk_score and risk_category from the aggregated probability
##scored_customers["risk_score"] = np.round(scored_customers["default_probability"] * 100).astype(int)
##scored_customers["risk_category"] = [get_risk_category(s) for s in scored_customers["risk_score"]]


# ---- C1: score all applications ---
score_df = mrg_df.copy()
score_ids = score_df["customer_id"].copy()

score_df = pd.get_dummies(score_df, columns=["employment_type", "residence_type", "loan_type", "education"], dtype=int)

# columns the model needs but this table lacks (should print [])
print("Missing columns:", [c for c in x_train.columns if c not in score_df.columns])

x_score = score_df.reindex(columns=x_train.columns, fill_value=0)
score_proba = xgb_best.predict_proba(x_score)[:, 1]

# check: test rows must match xgb_proba
check = pd.Series(score_proba, index=score_df.index).loc[x_test.index]
print("Matches xgb_proba:", np.allclose(check.values, xgb_proba))

# one row per customer, highest probability
app_level = pd.DataFrame({"customer_id": score_ids.values, "default_probability": score_proba})
scored_customers = app_level.groupby("customer_id", as_index=False)["default_probability"].max()

print(scored_customers.shape)
print(scored_customers.head())
print(scored_customers.isnull().sum())

# ---- C2: risk score, bands and checks ----
scored_customers["risk_score"] = np.round(scored_customers["default_probability"] * 100).astype(int)
scored_customers["risk_category"] = scored_customers["risk_score"].apply(get_risk_category)
scored_customers["predicted_default"] = (scored_customers["default_probability"] >= final_thr).astype(int)

# band sizes (count and %)
print(scored_customers["risk_category"].value_counts())
print((scored_customers["risk_category"].value_counts(normalize=True) * 100).round(1))

# actual default rate per band (customers with a disbursed loan only)
outcome = pd.DataFrame({"customer_id": customer_ids_all.values, "loan_default": y.values})
outcome = outcome.groupby("customer_id", as_index=False)["loan_default"].max()
chk = scored_customers.merge(outcome, on="customer_id", how="inner")
print(chk.groupby("risk_category")["loan_default"].agg(["count", "mean"]).round(3))

# save model, check it, save scored table 
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump({"model": xgb_best,
             "columns": list(x_train.columns),
             "threshold": final_thr},
            "models/default_model.pkl")

saved = joblib.load("models/default_model.pkl")
saved_proba = saved["model"].predict_proba(x_test[saved["columns"]])[:, 1]
print("Saved model matches xgb_proba:", np.allclose(saved_proba, xgb_proba))
print("Saved threshold:", saved["threshold"])


#scored_customers.to_csv("outputs/scored_customers.csv", index=False)
#print("Saving to:", os.getcwd())