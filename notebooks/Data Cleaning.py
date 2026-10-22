import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

cus_df= pd.read_csv("datasets/customers (1).csv")
crd_his_df= pd.read_csv("datasets/credit_history.csv")
lon_ap_df= pd.read_csv("datasets/loan_applications.csv")
trans_df= pd.read_csv("datasets/transactions.csv")

#### Customer Dataset ####

## Gender Mapping
cus_df["gender"]= cus_df["gender"].map({"Male": 1, "Female": 0})
#print(cus_df["gender"].head(5))
#print(cus_df.info())


## Date of Birth Conversion
cus_df["date_of_birth"]= pd.to_datetime(cus_df["date_of_birth"], errors='coerce')


## marital_status Mapping
cus_col= ["marital_status", "employment_type","residence_type"]
cus_df[cus_col]= cus_df[cus_col].astype("category")


## Phone datatype Conversion
cus_df["phone"]= cus_df["phone"].astype(str)
cus_df["phone"] = cus_df["phone"].str.replace("-", "", regex=False)

## KYC Status Mapping
cus_df["kyc_status"]= cus_df["kyc_status"].map({"Pending": 0, "Verified": 1})


## Customer Since Conversion
cus_df["customer_since"]= pd.to_datetime(cus_df["customer_since"], errors='coerce')
#print(cus_df["customer_since"].head(5))
#print(cus_df.info())

## Check for whitespace in string columns
clmns= ["customer_id", "first_name", "last_name", "occupation", "city", "state", "email", "phone"]
wte_space = cus_df[clmns].apply(lambda x: x.str.contains(" ").any())
#print(wte_space)

# Remove leading and trailing spaces
cus_df[["occupation", "state"]] = cus_df[["occupation", "state"]].apply(lambda x: x.str.strip())
wte_space_strp = cus_df[["occupation", "state"]].apply(lambda x: (x != x.str.strip()).any())
#print(wte_space_strp)

## Logical consistency between DOB and Age
#cus_df["calculated_age"] = (pd.Timestamp("2026-08-25") - cus_df["date_of_birth"]).dt.days // 365
# Check how many rows have difference more than 1 year
#mismatch = (cus_df["age"] - cus_df["calculated_age"]).abs() > 1
#print("Number of mismatched ages:", mismatch.sum())

#print(cus_df[['age', 'dependents', 'annual_income', 'years_at_residence']].describe())
#print((cus_df['years_at_residence'] > cus_df['age']).sum())
## Export clean data to new CSV file

#### Credit History Dataset ####

#print("Full-row duplicates:", crd_his_df.duplicated().sum())

## as_of_date datatype conversion

crd_his_df["as_of_date"]= pd.to_datetime(crd_his_df["as_of_date"], errors= "coerce")

## Logical consistency check

#crd_his_df["utlz_calc"]= crd_his_df["total_outstanding_debt"]/crd_his_df["total_credit_limit"]
#print((crd_his_df["credit_utilization_ratio"] - crd_his_df["utlz_calc"]).abs().max())


#print(crd_his_df.info())
## Export clean credit history dataset


#### Transaction Dataset ####

# replacing Transaction Type DataType
trans_df["transaction_type"] = trans_df["transaction_type"].map({"Debit": 0, "Credit":1})

# Replacing Channel Datatype
trans_df["channel"] = trans_df["channel"].astype('category')

# Replacing Transaction date dataype to datetime
trans_df["transaction_date"] = pd.to_datetime(trans_df["transaction_date"], errors= "coerce")

# Replacing Category dataype to Category
trans_df['category'] = trans_df['category'].astype('category')
#print(trans_df['transaction_date'].isna().sum())

#print(trans_df['transaction_date'].min())
#print(trans_df['transaction_date'].max())

#### Credit History ####

# Convert application_date datatype to datetime
lon_ap_df["application_date"]= pd.to_datetime(lon_ap_df["application_date"], errors= "coerce")

# Convert decision_date datatype to datetime
lon_ap_df["decision_date"]= pd.to_datetime(lon_ap_df["decision_date"], errors= "coerce")


# Check and stripping trailing and leading whitespace from loan_type, loan_purpose, approval_status columns
lon_ap_df[["loan_type", "loan_purpose", "approval_status"]] = lon_ap_df[["loan_type", "loan_purpose", "approval_status"]].apply(lambda x: x.str.strip())
wte_space_strp_lp = lon_ap_df[["loan_type", "loan_purpose", "approval_status"]].apply(lambda x: (x != x.str.strip()).any())

# Converting the the datatype of loan_type, loan_purpose, approval_status
col= ["loan_type", "loan_purpose", "approval_status"]
lon_ap_df[col] = lon_ap_df[col].astype("category")
#print(lon_ap_df.info())


# Logical checks as per the PRD if decision dates are greater than application dates
lgcheck = lon_ap_df["decision_date"] >= lon_ap_df["application_date"]
#print(lgcheck.value_counts())


print(cus_df.isnull().sum())
print(crd_his_df.isnull().sum())
print(lon_ap_df.isnull().sum())
print(trans_df.isnull().sum())


print(cus_df.duplicated().sum())
print(crd_his_df.duplicated().sum())
print(lon_ap_df.duplicated().sum())
print(trans_df.duplicated().sum())

#print(cus_df["gender"].isnull().sum())


####### Commenting out the export of cleaned datasets to avoid overwriting existing files. Uncomment the following lines to save the cleaned datasets for further analysis and modeling.

# Extracting clean datasets for further analysis and modeling
#cus_df.to_csv("datasets/cleaned_customers.csv", index=False)
#crd_his_df.to_csv("datasets/cleaned_credit_history.csv", index=False)
#lon_ap_df.to_csv("datasets/cleaned_loan_applications.csv", index=False)
#trans_df.to_csv("datasets/cleaned_transactions.csv", index=False)


