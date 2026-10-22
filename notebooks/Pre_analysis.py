import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cus_df= pd.read_csv("datasets/customers (1).csv")
crd_his_df= pd.read_csv("datasets/credit_history.csv")
lon_ap_df= pd.read_csv("datasets/loan_applications.csv")
trans_df= pd.read_csv("datasets/transactions.csv")

# Checks
#print(cus_df.info())
#print(crd_his_df.info())
#print(lon_ap_df.info())
#print(trans_df.info())

# Find columns where all values are null

## For Lon Application
#empty_cols = lon_ap_df[['decision_date', 'loan_default']].isnull().sum()
#print(empty_cols)

## For Credit History
#mis_val_crd= crd_his_df[["credit_id", "customer_id", "as_of_date", "credit_score", "num_open_accounts", "num_credit_inquiries_6m", "credit_utilization_ratio", "num_late_payments_30d",
#                        "num_late_payments_90d", "num_defaults_prior", "bankruptcies", "total_credit_limit",
#                        "total_outstanding_debt", "oldest_account_age_months"]].isnull().sum()
#print(mis_val_crd)

# Check for duplicates
dup_cus= cus_df['customer_id'].duplicated().sum()
print(dup_cus)

dup_cre= crd_his_df['credit_id'].duplicated().sum()
print(dup_cre)

dup_lon= lon_ap_df['application_id'].duplicated().sum()
print(dup_lon)

dup_tra= trans_df['transaction_id'].duplicated().sum()
print(dup_tra)


#### EDA Analysis ###

## Duplicate Values

#print(cus_df.info())
#print(crd_his_df.info())
#print(lon_ap_df.info())
#print(trans_df.info())

## Summary Stats

#print(cus_df.describe())
#print(crd_his_df.describe())
#print(lon_ap_df.describe())
#print(trans_df.describe())


######## Charts are commented out to avoid cluttering the notebook with multiple plots. uncomment them to regenerate the plots.

### Univariate Analysis ###

### For Categorical Variables in Customer Dataframe

## Age Distribution histogram


#age_dis = cus_df["age"].plot(kind="hist", color=["skyblue", "lightpink"], edgecolor="black", linestyle='-')

#age_dis.bar_label(age_dis.containers[0], padding=3)

#plt.title("Age Distribution")
#plt.xlabel("Age")
#plt.ylabel("Frequency")
#plt.xticks(rotation=0)  # Keeps text labels horizontal

# 4. Display the plot
#plt.show()


## Gender Distribution Bar Chart

# Count gender values and plot as a bar chart
#gen_dis = cus_df["gender"].value_counts().plot(kind="bar", color=["skyblue", "lightpink"], edgecolor="black", width=0.6)

#gen_dis.bar_label(gen_dis.containers[0], padding=3)

# 3. Customize the chart labels
#plt.title("Gender Distribution")
#plt.xlabel("Gender")
#plt.ylabel("Count")
#plt.xticks(rotation=0)  # Keeps text labels horizontal

# 4. Display the plot
#plt.show()

## Annual Income Distribution Histogram 

#anl_incm_hist = cus_df["annual_income"].plot(kind="hist", color=["skyblue"], edgecolor="black", linestyle='-')

#anl_incm_hist.bar_label(anl_incm_hist.containers[0], padding=3)


#plt.title("Annual Income Distribution")
#plt.xlabel("Annual Income in Millions")
#plt.ylabel("Frequency")
#plt.ticklabel_format(style='plain', axis='x')  # Keeps text labels horizontal and prevents scientific notation
#plt.xticks(rotation=30)  # Keeps text labels horizontal

# 4. Display the plot
#plt.show()


### For Categorical Variables in Credit History Dataframe

# Credit Score Distribution histogram
 
#crd_his_df_hist= crd_his_df["credit_score"].plot(kind="hist", color= ["Gold"], linestyle= "-")

#crd_his_df_hist.bar_label(crd_his_df_hist.containers[0], padding=3)

#plt.title("Credit Score Distribution")
#plt.xlabel("Credit Score")
#plt.ylabel("Frequency")
#plt.xticks(rotation=0)  # Keeps text labels horizontal

#plt.show()

# Credit Limit Distribution histogram

#crd_his_df_hist= crd_his_df["total_credit_limit"].plot(kind="hist", color= ["Gold"], linestyle= "-")

#crd_his_df_hist.bar_label(crd_his_df_hist.containers[0], padding=3)

#plt.title("Credit Limit Distribution")
#plt.xlabel("Credit Limit")
#plt.ylabel("Frequency")
#plt.ticklabel_format(style='plain', axis='x')  # Keeps text labels horizontal and prevents scientific notation
#plt.xticks(rotation=40)  # Keeps text labels horizontal

#plt.show()

# Total outstanding debt Distribution histogram

#crd_his_df_hist= crd_his_df["total_outstanding_debt"].plot(kind="hist", color= ["Gold"], linestyle= "-")

#crd_his_df_hist.bar_label(crd_his_df_hist.containers[0], padding=3)

#plt.title("Total Outstanding Debt Distribution")
#plt.xlabel("Total Outstanding Debt")
#plt.ylabel("Frequency")
#plt.ticklabel_format(style='plain', axis='x')  

#plt.xticks(rotation=40)  # Keeps text labels horizontal

#plt.show()

### For Categorical Variables in Loan Applications Dataset

# Requested amount Distribution histogram

#ln_ap_df_hist= lon_ap_df["requested_amount"].dropna().astype(int).plot(kind="hist", color="gold", edgecolor="black")

#ln_ap_df_hist.bar_label(ln_ap_df_hist.containers[0], padding=3)

#plt.title("Requested Amount Distribution")
#plt.xlabel("Requested Amount")
#plt.ylabel("Frequency")  
#plt.ticklabel_format(style='plain', axis='x')  # Keeps text labels horizontal and prevents scientific notation
#plt.xticks(rotation=40)  # Keeps text labels horizontal

#plt.show()

# Interest Rate Distribution Histogram

#lon_ap_df_hist= lon_ap_df["interest_rate"].plot(kind="hist", color="gold", edgecolor="black")

#lon_ap_df_hist.bar_label(lon_ap_df_hist.containers[0], padding=3)

#plt.title("Interest Rate Distribution")
#plt.xlabel("Interest Rate")
#plt.ylabel("Frequency")
#plt.ticklabel_format(style='plain', axis='x')  # Keeps text labels horizontal and
#plt.xticks(rotation=0)  # Keeps text labels horizontal
#plt.show()  


# Applicant Monthly Income Distribution Histogram

#lon_ap_df_hist= lon_ap_df["applicant_monthly_income"].plot(kind="hist", color="gold", edgecolor="black")

#lon_ap_df_hist.bar_label(lon_ap_df_hist.containers[0], padding=3)

#plt.title("Applicant Monthly Income Distribution")
#plt.xlabel("Applicant Monthly Income")
#plt.ylabel("Frequency")
#plt.ticklabel_format(style='plain', axis='x')  # Keeps text labels horizontal and
#plt.xticks(rotation=40)  # Keeps text labels horizontal

#plt.show()

# Approved Amount Distribution Histogram

#lon_ap_df_hist= lon_ap_df["approved_amount"].plot(kind="hist", color="gold", edgecolor="black")
#lon_ap_df_hist.bar_label(lon_ap_df_hist.containers[0], padding=3)

#plt.title("Approved Amount Distribution")
#plt.xlabel("Approved Amount")
#plt.ylabel("Frequency")
#plt.ticklabel_format(style='plain', axis='x')  # Keeps text labels horizontal and prevents scientific notation
#plt.xticks(rotation=40)  # Keeps text labels horizontal

#plt.show()


# Loan Default Distribution Histogram

#lon_ap_df_bar = lon_ap_df["loan_default"].value_counts().plot(kind="bar", color=["skyblue", "lightpink"], edgecolor="black", width=0.6)

#lon_ap_df_bar.bar_label(lon_ap_df_bar.containers[0], padding=3)

#plt.title("Loan Default Distribution")
#plt.xlabel("Loan Default")
#plt.ylabel("Frequency")
#plt.xticks(rotation=0)  # Keeps text labels horizontal
#plt.show()


# transaction amount Distribution Histogram

#trans_df_hist= trans_df["amount"].plot(kind="hist", color="gold", edgecolor="black")

#trans_df_hist.bar_label(trans_df_hist.containers[0], padding=3)

#plt.title("Transaction Amount Distribution")
#plt.xlabel("Transaction Amount")
#plt.ylabel("Frequency")
#plt.ticklabel_format(style='plain', axis='x')  # Keeps text labels horizontal and
#plt.xticks(rotation=40)  # Keeps text labels horizontal

#plt.show()

# balance Distribution Histogram

#trans_df_hist= trans_df["balance_after"].plot(kind="hist", color="gold", edgecolor="black")

#trans_df_hist.bar_label(trans_df_hist.containers[0], padding=3)

#plt.title("Balance Distribution")
#plt.xlabel("Balance")
#plt.ylabel("Frequency")
#plt.ticklabel_format(style='plain', axis='x')  # Keeps text labels horizontal and
#plt.xticks(rotation=40)  # Keeps text labels horizontal

#plt.show()


### Bivariate Analysis ###

# Credit Score × Loan Default Box Plot

cre_lon_df= pd.merge(crd_his_df, lon_ap_df, on="customer_id", how="left")

#lon_cre_df_bx= cre_lon_df.boxplot(column="credit_score", by="loan_default", grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'), medianprops=dict(color='red'))
#print(cre_lon_df.info())

#plt.xlabel("Loan Default")
#plt.ylabel("Credit Score")
#plt.title("Credit Score v Loan Default")
#plt.suptitle('')
#plt.xticks(rotation=0)  # Keeps text labels horizontal
#plt.show()


# Credit Score × Loan Approval


#cre_lon_df_sct= sns.scatterplot(data=cre_lon_df, x="approved_amount", y="credit_score", color="gold")

#plt.xlabel("Approved Amount")
#plt.ylabel("credit_score")
#plt.title("Approved Amount v Credit Score")
#plt.ticklabel_format(style='plain', axis='x')
#plt.xticks(rotation=40)  # Keeps text labels horizontal

#plt.show()

# Interest Rate × Loan Default Box Plot

#lon_df_bx= lon_ap_df.boxplot(column="interest_rate", by="loan_default", grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'), medianprops=dict(color='red'))

#plt.xlabel("Loan Default")
#plt.ylabel("Interest Rate")
#plt.title("Interest Rate v Loan Default")
#plt.suptitle('')
#plt.xticks(rotation=0)  # Keeps text labels horizontal
#plt.show()  


#Annual Income × Loan Default
#Boxplot

cus_lon_df= pd.merge(cus_df, lon_ap_df, on="customer_id", how="left")

#cus_lon_df_bx= cus_lon_df.boxplot(column="annual_income", by="loan_default", grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'), medianprops=dict(color='red')) 

#plt.xlabel("Loan Default")
#plt.ylabel("Annual Income")
#plt.title("Annual Income v Loan Default")
#plt.suptitle('')
#plt.xticks(rotation=0)  # Keeps text labels horizontal
#plt.show()


# Outstanding Debt × Loan Default Box Plots

#cre_lon_df_osbx= cre_lon_df.boxplot(column="total_outstanding_debt", by="loan_default", grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'), medianprops=dict(color='red'))

#plt.xlabel("Loan Default")
#plt.ylabel("Total Outstanding Debt")
#plt.title("Total Outstanding Debt v Loan Default")
#plt.suptitle('')
#plt.xticks(rotation=0)  # Keeps text labels horizontal
#plt.show()


# Loan Purpose × Loan Default Bar chart — default rate

def_rte= lon_ap_df.groupby("loan_purpose")["loan_default"].mean().sort_values(ascending=False)

#lon_ap_df_bar= def_rte.plot(kind="bar", color=["skyblue"], edgecolor="black", width=0.6)

#plt.title("Loan Purpose v Loan Default Rate")
#plt.xlabel("Loan Purpose")
#plt.ylabel("Loan Default Rate")
#plt.ticklabel_format(style='plain', axis='y')  # Keeps text labels horizontal and prevents scientific notation
#plt.legend()
#plt.xticks(rotation=90)  # Keeps text labels horizontal
#plt.show()


# Loan Term × Loan Default Bar chart — default rate

#def_trm_rte= lon_ap_df.groupby("loan_term_months")["loan_default"].mean().sort_values(ascending=False)

#lon_ap_df_trm_bar= def_trm_rte.plot(kind="bar", color=["skyblue"], edgecolor="black", width=0.6)

#plt.title("Loan Term v Loan Default Rate")
#plt.xlabel("Loan Term")
#plt.ylabel("Loan Default Rate")
#plt.ticklabel_format(style='plain', axis='y')  # Keeps text labels horizontal and prevents scientific notation
#plt.legend()
#plt.xticks(rotation=0)  # Keeps text labels horizontal
#plt.show()


# Requested Amount × Approved Amount Scatter Plot

#lon_ap_df_sct= sns.scatterplot(data=lon_ap_df, x="requested_amount", y="approved_amount", color="gold")

#plt.xlabel("Requested Amount")
#plt.ylabel("Approved Amount")
#plt.title("Requested Amount v Approved Amount")
#plt.ticklabel_format(style='plain', axis='x')
#plt.xticks(rotation=40)  # Keeps text labels horizontal
#plt.show()

### Multivariate Analysis ###

# Credit Score × Annual Income × Loan Default: Scatter plot, differentiated by default status

#lon_cre_cus_df= pd.merge(cre_lon_df, cus_df, on="customer_id", how="left")

#lon_ap_df_sct= sns.scatterplot(data=lon_cre_cus_df, x="credit_score", y="annual_income", hue="loan_default", palette=["grey", "Gold"])

#plt.xlabel("Credit Score")
#plt.ylabel("Annual Income")
#plt.title("Credit Score v Annual Income v Loan Default")
#plt.ticklabel_format(style='plain', axis='y')  # Keeps text labels horizontal and
#plt.xticks(rotation=40)  # Keeps text labels horizontal
#plt.show()

# Credit Score × Outstanding Debt × Loan Default Scatter plot, differentiated by default status

#cre_out_lon_df= pd.merge(cre_lon_df, cus_df, on="customer_id", how="left")

#cre_out_lon_df_sct= sns.scatterplot(data=cre_out_lon_df, x="credit_score", y="total_outstanding_debt", hue="loan_default", palette=["grey", "Gold"])

#plt.xlabel("Credit Score")
#plt.ylabel("Total Outstanding Debt")
#plt.title("Credit Score v Total Outstanding Debt v Loan Default")
#plt.ticklabel_format(style='plain', axis='y')  # Keeps text labels horizontal and
#plt.xticks(rotation=40)  # Keeps text labels horizontal
#plt.show()


### Multivariate Analysis ###

# All important numerical credit/loan variables Correlation heatmap

#Credit Score
#Annual Income
#Applicant Monthly Income
#Total Credit Limit
#Total Outstanding Debt
#Requested Amount
#Approved Amount
#Interest Rate
#Transaction Amount
#Balance After

# 2. Merge them on a shared unique ID key
# (Skip this step if you simply want to concatenate columns side-by-side without an ID)
#mas_df = cus_df.merge(crd_his_df, on="customer_id").merge(lon_ap_df, on="customer_id").merge(trans_df, on="customer_id")

#selected_variables = [
#    'credit_score', 'total_credit_limit',
#    'total_outstanding_debt', 'requested_amount',
#    'approved_amount', 'interest_rate',
#    'amount', 'balance_after'
#]

# Filter the merged dataframe
#final_df = mas_df[selected_variables]

# 3. Keep only numeric columns
#numeric_df = mas_df.select_dtypes(include=[np.number])

# 4. Compute the Pearson correlation matrix
#corr_matrix = final_df.corr(method='pearson')

# 5. Optional: Mask the upper triangle (removes redundant duplicate mirrors)
#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 6. Plot the heatmap
#plt.figure(figsize=(16, 12))  # Large canvas for many variables
#sns.heatmap(
#    corr_matrix, 
#    #mask=mask, 
#    annot=True,          # Displays correlation numbers inside cells
#    fmt=".2f",           # Limits decimals to 2 places
#    cmap="RdBu",         # Red-Blue diverging palette (Red=Neg, Blue=Pos)
#    vmin=-1, vmax=1,     # Keeps color scale anchored strictly to correlation limits
#    square=True          # Forces grid cells to be perfect squares
#)
#plt.ticklabel_format(style='plain', axis='x')  # Keeps text labels horizontal and prevents scientific notation
#plt.xticks(rotation=45)  # Rotates x-axis labels for better readability
#plt.title("Master Correlation Heatmap Across 4 Datasets", fontsize=16)
#plt.show()