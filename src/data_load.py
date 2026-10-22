import pandas as pd

def load_data():
    cus_df_cln = pd.read_csv("datasets/cleaned_customers.csv")
    crd_his_df_cln = pd.read_csv("datasets/cleaned_credit_history.csv")
    lon_ap_df_cln = pd.read_csv("datasets/cleaned_loan_applications.csv")
    trans_df_cln = pd.read_csv("datasets/cleaned_transactions.csv")
    return cus_df_cln, crd_his_df_cln, lon_ap_df_cln, trans_df_cln