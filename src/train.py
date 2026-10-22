import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import os
import joblib
import shap
import matplotlib.pyplot as plt


# Best settings found by GridSearchCV in notebooks/Feature Engineering & Baseline Model.py (fixed here to keep the run under 5 minutes)

RF_PARAMS = {"max_depth": 8, "min_samples_leaf": 6, "min_samples_split": 2, "n_estimators": 500}
XGB_PARAMS = {"colsample_bytree": 1.0, "learning_rate": 0.01, "max_depth": 3, "n_estimators": 200, "subsample": 1.0}
FINAL_THR = 0.45

def train_models(x_train, x_train_sc, y_train):

    # logistic regression model
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(x_train_sc, y_train)

    # decision Tree model
    dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=5)
    dt_model.fit(x_train, y_train)

    rf_model = RandomForestClassifier(class_weight= "balanced", random_state=42, **RF_PARAMS)
    rf_model.fit(x_train, y_train)

    xgb_model = XGBClassifier(scale_pos_weight=3, random_state=42, **XGB_PARAMS)
    xgb_model.fit(x_train, y_train)

    return {"Logistic Regression": model, "Decision Tree": dt_model, "Random Forest": rf_model, "XGBoost": xgb_model}


def cv_threshold_table(models, x_train, x_train_sc, y_train, target_recall=0.73):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# each model with the data it was trained on
    data = {"Logistic Regression": x_train_sc, "Decision Tree": x_train,
                "Random Forest": x_train, "XGBoost": x_train}


    rows = []

    for name, mdl in models.items():
        X = data[name]
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
    return cv_table



def final_test(xgb_model, x_test, y_test, threshold=FINAL_THR):
    xgb_proba = xgb_model.predict_proba(x_test)[:, 1]

    final_pred = (xgb_proba >= threshold).astype(int)
    print("XGBoost final test result @ threshold", threshold)
    print(confusion_matrix(y_test, final_pred))
    print("Accuracy:", accuracy_score(y_test, final_pred))
    print("Precision:", precision_score(y_test, final_pred))
    print("Recall:", recall_score(y_test, final_pred))
    print("F1:", f1_score(y_test, final_pred))
    print("ROC AUC:", roc_auc_score(y_test, xgb_proba))
    
    return xgb_proba


def save_artifacts(xgb_model, x_train, x_test, xgb_proba, threshold=FINAL_THR):
    xgb_feature_importances = pd.DataFrame({'feature': x_train.columns, 'importance': xgb_model.feature_importances_})
    xgb_feature_importances = xgb_feature_importances.sort_values(by='importance', ascending=False)
    print("\nTop Feature Importances from XGBoost:")
    print(xgb_feature_importances.head())

    os.makedirs("models", exist_ok=True)
    xgb_feature_importances.to_csv("models/feature_importance.csv", index=False) 

    explner = shap.TreeExplainer(xgb_model)
    shap_values = explner.shap_values(x_test)
    print("SHAP Summary Plot for XGBoost:")
    shap.summary_plot(shap_values, x_test, show=False)
    plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump({"model": xgb_model,
             "columns": list(x_train.columns),
             "threshold": threshold},
            "models/default_model.pkl")

    saved = joblib.load("models/default_model.pkl")
    saved_proba = saved["model"].predict_proba(x_test[saved["columns"]])[:, 1]
    print("Saved model matches xgb_proba:", np.allclose(saved_proba, xgb_proba))
    print("Saved threshold:", saved["threshold"])