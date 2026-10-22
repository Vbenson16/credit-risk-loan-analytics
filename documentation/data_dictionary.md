# Data Dictionary: Credora Finance (CRDA-RISK-2024)

All four datasets join on `customer_id`. Files are in `datasets/` (raw and `cleaned_*.csv`).

## 1. customers (5,000 rows, PK `customer_id`)
| Column | Type | Description |
|---|---|---|
| customer_id | text | Unique customer id (e.g. CUST100001) |
| first_name, last_name | text | Customer name (personal data, not modelled) |
| gender | int | Gender, coded: Male = 1, Female = 0 |
| date_of_birth | date | Date of birth |
| age | int | Age in years (18-74) |
| marital_status | text | Single / Married / Divorced / Widowed |
| dependents | int | Number of dependents (0-6) |
| education | text | Highest education level |
| employment_type | text | Salaried / Self-Employed / Business Owner / Retired / Unemployed |
| occupation | text | Job title |
| annual_income | numeric | Annual income (INR) |
| city, state | text | Location (20 cities, 13 states) |
| residence_type | text | Owned / Rented / Mortgaged / Family Owned |
| years_at_residence | int | Years at current residence |
| email, phone | text | Contact details (personal data, not modelled) |
| kyc_status | int | KYC status, coded: Verified = 1, Pending = 0 |
| customer_since | date | Start of the relationship |

## 2. credit_history (5,000 rows, PK `credit_id`, one row per customer)
| Column | Type | Description |
|---|---|---|
| credit_id | text | Unique bureau record id |
| customer_id | text | Links to customers |
| as_of_date | date | Date of the bureau snapshot |
| credit_score | int | Bureau score, 300-900 |
| num_open_accounts | int | Open credit accounts |
| num_credit_inquiries_6m | int | Hard inquiries in the last 6 months |
| credit_utilization_ratio | numeric | Balance-to-limit ratio (0-1) |
| num_late_payments_30d | int | Payments 30+ days late |
| num_late_payments_90d | int | Payments 90+ days late |
| num_defaults_prior | int | Prior recorded defaults |
| bankruptcies | int | Bankruptcies on record |
| total_credit_limit | numeric | Total sanctioned limit (INR) |
| total_outstanding_debt | numeric | Current outstanding balance (INR) |
| oldest_account_age_months | int | Age of the oldest account (months) |

## 3. loan_applications (7,500 rows, PK `application_id`)
| Column | Type | Description |
|---|---|---|
| application_id | text | Unique application id |
| customer_id | text | Links to customers |
| application_date | date | Date submitted |
| loan_type | text | Personal / Home / Auto / Education / Business / Gold |
| loan_purpose | text | Stated purpose |
| requested_amount | numeric | Amount requested (INR) |
| loan_term_months | int | Requested tenure (months) |
| interest_rate | numeric | Offered annual rate (%) |
| applicant_monthly_income | numeric | Declared monthly income (INR) |
| existing_emi | numeric | Existing monthly EMI (INR) |
| debt_to_income_ratio | numeric | (existing + new EMI) / monthly income |
| collateral_flag | int | 1 if collateral offered |
| approval_status | text | Approved / Rejected / Under Review (excluded: leakage) |
| approved_amount | numeric | Sanctioned amount (excluded: leakage) |
| decision_date | date | Decision date (excluded: leakage) |
| disbursed_flag | int | 1 if disbursed (used only to select training rows) |
| loan_default | int | **Target.** 1 = defaulted, 0 = repaid, empty if not disbursed |

## 4. transactions (40,079 rows, PK `transaction_id`, last 6 months)
| Column | Type | Description |
|---|---|---|
| transaction_id | text | Unique transaction id |
| customer_id | text | Links to customers |
| transaction_date | date | Transaction date |
| transaction_type | int | 1 = credit, 0 = debit (cleaned file) |
| category | text | Salary, Shopping, Utilities, EMI, Dining, Travel, ATM, Transfer, Investment, Insurance, Refund |
| channel | text | UPI / Debit Card / Credit Card / Net Banking / ATM / Branch |
| amount | numeric | Amount (INR) |
| balance_after | numeric | Balance after the transaction (INR) |

## 5. Engineered features (src/features.py)
| Column | Description |
|---|---|
| avg_bal | Average `balance_after` per customer |
| trns_count | Number of transactions per customer |
| total_credit, total_debit | Sum of credit / debit transactions per customer |
| net_cash_flow | total_credit minus total_debit |
| inflow_outflow_ratio | total_credit / total_debit; capped at the 99th percentile when debit is 0; 0 if no activity |
| emi_outflow | Sum of transactions in the EMI category |
| debt_to_lmt_ratio | total_outstanding_debt / total_credit_limit |
| ln_to_incm_ratio | requested_amount / annual_income |
| tenure | Years from customer_since to application_date (minimum 0) |

## 6. Output: outputs/scored_customers.csv (3,904 customers)
| Column | Description |
|---|---|
| customer_id | Customer id |
| default_probability | Highest P(default) across the customer's applications (0-1) |
| risk_score | round(default_probability x 100), 0-100 |
| risk_category | Low (0-33), Medium (34-66), High (67-100) |
| predicted_default | 1 if default_probability >= 0.45, else 0 |

## Excluded from modelling
- `email`, `phone` and names: personal data (PRD section 8).
- `approval_status`, `approved_amount`, `decision_date`, `disbursed_flag`: known only after the decision (leakage, PRD section 13.3).
- Text and date columns are dropped after one-hot encoding of `employment_type`, `residence_type`, `loan_type` and `education`.