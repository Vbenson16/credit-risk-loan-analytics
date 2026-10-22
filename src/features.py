import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define the specific columns to standardize
std_clmns= ["requested_amount", "loan_term_months", "interest_rate", "applicant_monthly_income","existing_emi",
            "debt_to_income_ratio", "age", "dependents", "annual_income",
            "years_at_residence", "credit_score", "num_open_accounts", "num_credit_inquiries_6m",
            "credit_utilization_ratio", "num_late_payments_30d", "num_late_payments_90d",
            "num_defaults_prior", "total_credit_limit", "total_outstanding_debt",
            "oldest_account_age_months", "avg_bal", "trns_count", "total_credit", "total_debit",
            "net_cash_flow", "inflow_outflow_ratio", "emi_outflow", "debt_to_lmt_ratio", "ln_to_incm_ratio", "tenure"]




def build_merged(cus_df_cln, crd_his_df_cln, lon_ap_df_cln, trans_df_cln):

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

# calculate the ratio and fix the problem cases
        ratio = summary_df["total_credit"] / summary_df["total_debit"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        cap = ratio.quantile(0.99)   # 99th percentile of the normal ratios
        ratio[(summary_df["total_debit"] == 0) & (summary_df["total_credit"] > 0)] = cap
        ratio = ratio.fillna(0)      # both zero = no activity

        summary_df["inflow_outflow_ratio"] = ratio


        emi_outflow= cus_trans_agg[cus_trans_agg["category"] == "EMI"].groupby("customer_id")["amount"].sum()
        summary_df["emi_outflow"]= emi_outflow
        summary_df["emi_outflow"]= summary_df["emi_outflow"].fillna(0)


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
        return mrg_df

## 418 apps dated before customer_since; set to 0.

def build_training_data(mrg_df):
        # filtering the merged dataset to include only rows where disbursed_flag is 1
    filtered_mrg_df = mrg_df[(mrg_df["disbursed_flag"]==1)] 

#### One Hot encoding ####
# One-hot encoding for categorical variables
    filtered_mrg_df = pd.get_dummies(filtered_mrg_df, columns=["employment_type", "residence_type", "loan_type", "education"], dtype=int)


# leakage control: dropping columns that may cause data leakage
    filtered_mrg_df= filtered_mrg_df.drop(columns=["approval_status", "approved_amount", "decision_date", "disbursed_flag", "email", "phone"])



# 80/20 stratified train/test on loan default
    y= filtered_mrg_df["loan_default"]


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

    return x_train, x_test, y_train, y_test, x_train_sc, x_test_sc, customer_ids_all, y