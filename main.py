from src.data_load import load_data
from src.features import build_merged, build_training_data
from src.train import (train_models, cv_threshold_table, final_test,save_artifacts, FINAL_THR)
from src.risk_score import score_customers


def main():
    # 1. load and build the data
    cus, crd, lon, trans = load_data()
    mrg_df = build_merged(cus, crd, lon, trans)
    (x_train, x_test, y_train, y_test,
     x_train_sc, x_test_sc, customer_ids_all, y) = build_training_data(mrg_df)

    # 2. train the four models
    models = train_models(x_train, x_train_sc, y_train)

    # 3. CV table (training data only)
    cv_table = cv_threshold_table(models, x_train, x_train_sc, y_train)
    print(cv_table.sort_values("Precision", ascending=False))

    # 4. final test, once, at the fixed threshold
    xgb_proba = final_test(models["XGBoost"], x_test, y_test)

    # 5. save importance table, SHAP plot and model
    save_artifacts(models["XGBoost"], x_train, x_test, xgb_proba)

    # 6. score all customers and save the table
    score_customers(mrg_df, models["XGBoost"], x_train, x_test, xgb_proba,
                    customer_ids_all, y, FINAL_THR)


if __name__ == "__main__":
    main()