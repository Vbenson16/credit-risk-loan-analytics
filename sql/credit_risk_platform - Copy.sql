-- =====================================================================
-- Credora Finance | Credit Risk & Loan Approval Analytics Platform
-- Relational schema, seed data, indexes, constraints and analytical queries
-- Target engine: PostgreSQL 14+  (MySQL 8+ compatible with minor edits)
-- Owner: Data & Analytics Team, Credora Finance
-- =====================================================================

-- ---------------------------------------------------------------------
-- 0. Clean start (safe re-run)
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS transactions      CASCADE;
DROP TABLE IF EXISTS loan_applications CASCADE;
DROP TABLE IF EXISTS credit_history    CASCADE;
DROP TABLE IF EXISTS customers         CASCADE;

-- ---------------------------------------------------------------------
-- 1. DDL : CUSTOMERS  (parent table)
-- ---------------------------------------------------------------------
CREATE TABLE customers (
    customer_id         VARCHAR(12)   PRIMARY KEY,
    first_name          VARCHAR(50)   NOT NULL,
    last_name           VARCHAR(50)   NOT NULL,
    gender              VARCHAR(10)   CHECK (gender IN ('Male','Female','Other')),
    date_of_birth       DATE          NOT NULL,
    age                 SMALLINT      CHECK (age BETWEEN 18 AND 100),
    marital_status      VARCHAR(15),
    dependents          SMALLINT      DEFAULT 0 CHECK (dependents >= 0),
    education           VARCHAR(30),
    employment_type     VARCHAR(20),
    occupation          VARCHAR(60),
    annual_income       NUMERIC(12,2) CHECK (annual_income >= 0),
    city                VARCHAR(50),
    state               VARCHAR(50),
    residence_type      VARCHAR(20),
    years_at_residence  SMALLINT      CHECK (years_at_residence >= 0),
    email               VARCHAR(120),
    phone               VARCHAR(20),
    kyc_status          VARCHAR(15)   DEFAULT 'Pending'
                                      CHECK (kyc_status IN ('Verified','Pending','Rejected')),
    customer_since      DATE          NOT NULL
);

-- ---------------------------------------------------------------------
-- 2. DDL : CREDIT_HISTORY  (1:1 with customer, bureau snapshot)
-- ---------------------------------------------------------------------
CREATE TABLE credit_history (
    credit_id                 VARCHAR(12)   PRIMARY KEY,
    customer_id               VARCHAR(12)   NOT NULL UNIQUE,
    as_of_date                DATE          NOT NULL,
    credit_score              SMALLINT      CHECK (credit_score BETWEEN 300 AND 900),
    num_open_accounts         SMALLINT      CHECK (num_open_accounts >= 0),
    num_credit_inquiries_6m   SMALLINT      CHECK (num_credit_inquiries_6m >= 0),
    credit_utilization_ratio  NUMERIC(4,3)  CHECK (credit_utilization_ratio BETWEEN 0 AND 1),
    num_late_payments_30d     SMALLINT      DEFAULT 0,
    num_late_payments_90d     SMALLINT      DEFAULT 0,
    num_defaults_prior        SMALLINT      DEFAULT 0,
    bankruptcies              SMALLINT      DEFAULT 0,
    total_credit_limit        NUMERIC(12,2) CHECK (total_credit_limit >= 0),
    total_outstanding_debt    NUMERIC(12,2) CHECK (total_outstanding_debt >= 0),
    oldest_account_age_months SMALLINT      CHECK (oldest_account_age_months >= 0),
    CONSTRAINT fk_credit_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------
-- 3. DDL : LOAN_APPLICATIONS  (many:1 with customer, holds ML target)
-- ---------------------------------------------------------------------
CREATE TABLE loan_applications (
    application_id           VARCHAR(12)   PRIMARY KEY,
    customer_id              VARCHAR(12)   NOT NULL,
    application_date         DATE          NOT NULL,
    loan_type                VARCHAR(20)   NOT NULL,
    loan_purpose             VARCHAR(50),
    requested_amount         NUMERIC(12,2) CHECK (requested_amount > 0),
    loan_term_months         SMALLINT      CHECK (loan_term_months > 0),
    interest_rate            NUMERIC(5,2)  CHECK (interest_rate >= 0),
    applicant_monthly_income NUMERIC(12,2) CHECK (applicant_monthly_income >= 0),
    existing_emi             NUMERIC(12,2) DEFAULT 0,
    debt_to_income_ratio     NUMERIC(5,3)  CHECK (debt_to_income_ratio >= 0),
    collateral_flag          SMALLINT      DEFAULT 0 CHECK (collateral_flag IN (0,1)),
    approval_status          VARCHAR(15)   NOT NULL
                                           CHECK (approval_status IN ('Approved','Rejected','Under Review')),
    approved_amount          NUMERIC(12,2) DEFAULT 0,
    decision_date            DATE,
    disbursed_flag           SMALLINT      DEFAULT 0 CHECK (disbursed_flag IN (0,1)),
    loan_default             SMALLINT      CHECK (loan_default IN (0,1)),   -- NULL for non-disbursed
    CONSTRAINT fk_loan_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    CONSTRAINT chk_decision_after_application
        CHECK (decision_date IS NULL OR decision_date >= application_date)
);

-- ---------------------------------------------------------------------
-- 4. DDL : TRANSACTIONS  (many:1 with customer, behavioural signal)
-- ---------------------------------------------------------------------
CREATE TABLE transactions (
    transaction_id     VARCHAR(14)   PRIMARY KEY,
    customer_id        VARCHAR(12)   NOT NULL,
    transaction_date   DATE          NOT NULL,
    transaction_type   VARCHAR(10)   CHECK (transaction_type IN ('Credit','Debit')),
    category           VARCHAR(30),
    channel            VARCHAR(20),
    amount             NUMERIC(12,2) CHECK (amount >= 0),
    balance_after      NUMERIC(12,2),
    CONSTRAINT fk_txn_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------
-- 5. INDEXES  (query-path optimisation for analytics & joins)
-- ---------------------------------------------------------------------
CREATE INDEX idx_credit_customer      ON credit_history (customer_id);
CREATE INDEX idx_credit_score         ON credit_history (credit_score);
CREATE INDEX idx_loan_customer        ON loan_applications (customer_id);
CREATE INDEX idx_loan_status          ON loan_applications (approval_status);
CREATE INDEX idx_loan_appdate         ON loan_applications (application_date);
CREATE INDEX idx_loan_default         ON loan_applications (loan_default);
CREATE INDEX idx_txn_customer         ON transactions (customer_id);
CREATE INDEX idx_txn_date             ON transactions (transaction_date);
CREATE INDEX idx_txn_category         ON transactions (category);

-- =====================================================================
-- 6. DATA LOAD  (bulk load of the 4 raw CSVs)
--    Replace <path_to_project> with the full path to your project folder before running.
--    The sample seed rows were replaced by the full CSV load (they held personal data and clashed with the bulk load).
-- =====================================================================
COPY customers
FROM '<path_to_project>/datasets/customers (1).csv'
DELIMITER ','
CSV HEADER;

COPY credit_history
FROM '<path_to_project>/datasets/credit_history.csv'
WITH (FORMAT csv, HEADER true);

COPY transactions
FROM '<path_to_project>/datasets/transactions.csv'
WITH (FORMAT csv, HEADER true);

COPY loan_applications
FROM '<path_to_project>/datasets/loan_applications.csv'
WITH (FORMAT csv, HEADER true);


-- =====================================================================
-- 7. ANALYTICAL QUERIES
-- =====================================================================

-- 7.1  Credit score analysis: distribution across standard risk bands
SELECT
    CASE
        WHEN credit_score >= 750 THEN 'Excellent (750-900)'
        WHEN credit_score >= 700 THEN 'Good (700-749)'
        WHEN credit_score >= 650 THEN 'Fair (650-699)'
        WHEN credit_score >= 580 THEN 'Poor (580-649)'
        ELSE 'Very Poor (300-579)'
    END                                   AS credit_band,
    COUNT(*)                              AS customers,
    ROUND(AVG(credit_score),0)            AS avg_score,
    ROUND(AVG(credit_utilization_ratio),3) AS avg_utilization,
    ROUND(AVG(num_late_payments_30d),2)   AS avg_late_30d
FROM credit_history
GROUP BY 1
ORDER BY MIN(credit_score) DESC;

-- 7.2  Loan approval trends: monthly approval rate over time
SELECT
    DATE_TRUNC('month', application_date)::date       AS month,
    COUNT(*)                                          AS applications,
    SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
    ROUND(100.0 * SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*),0), 1)                    AS approval_rate_pct
FROM loan_applications
GROUP BY 1
ORDER BY 1;

-- 7.3  Customer segmentation: income band x employment, with approval behaviour
SELECT
    c.employment_type,
    CASE
        WHEN c.annual_income >= 1500000 THEN 'High (15L+)'
        WHEN c.annual_income >=  600000 THEN 'Mid (6-15L)'
        ELSE 'Entry (<6L)'
    END                                              AS income_band,
    COUNT(DISTINCT c.customer_id)                    AS customers,
    ROUND(AVG(ch.credit_score),0)                    AS avg_credit_score,
    ROUND(100.0*AVG(CASE WHEN la.approval_status='Approved' THEN 1.0 ELSE 0 END),1) AS approval_rate_pct
FROM customers c
JOIN credit_history ch    ON ch.customer_id = c.customer_id
LEFT JOIN loan_applications la ON la.customer_id = c.customer_id
GROUP BY 1,2
ORDER BY 1,2;

-- 7.4  High-risk customer identification (rule-based watchlist)
SELECT
    c.customer_id, c.first_name, c.last_name, c.city,
    ch.credit_score, ch.credit_utilization_ratio,
    ch.num_late_payments_90d, ch.num_defaults_prior
FROM customers c
JOIN credit_history ch ON ch.customer_id = c.customer_id
WHERE ch.credit_score < 600
   OR ch.credit_utilization_ratio > 0.80
   OR ch.num_defaults_prior >= 1
   OR ch.num_late_payments_90d >= 2
ORDER BY ch.credit_score ASC
LIMIT 100;

-- 7.5  Default rate analysis by loan type (disbursed loans only)
SELECT
    loan_type,
    COUNT(*)                                    AS disbursed_loans,
    SUM(loan_default)                           AS defaults,
    ROUND(100.0*AVG(loan_default),2)            AS default_rate_pct,
    ROUND(AVG(interest_rate),2)                 AS avg_interest_rate
FROM loan_applications
WHERE disbursed_flag = 1 AND loan_default IS NOT NULL
GROUP BY loan_type
ORDER BY default_rate_pct DESC;

-- 7.6  Loan amount distribution by loan type
SELECT
    loan_type,
    COUNT(*)                                        AS applications,
    ROUND(MIN(requested_amount),0)                  AS min_amount,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY requested_amount),0) AS median_amount,
    ROUND(AVG(requested_amount),0)                  AS avg_amount,
    ROUND(MAX(requested_amount),0)                  AS max_amount
FROM loan_applications
GROUP BY loan_type
ORDER BY avg_amount DESC;

-- 7.7  Monthly approval statistics with disbursed value
SELECT
    TO_CHAR(application_date,'YYYY-MM')             AS month,
    COUNT(*)                                        AS applications,
    SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
    SUM(CASE WHEN approval_status='Rejected' THEN 1 ELSE 0 END) AS rejected,
    ROUND(SUM(approved_amount)/1000000.0,2)         AS disbursed_amount_mn
FROM loan_applications
GROUP BY 1
ORDER BY 1;

-- 7.8  Risk categorisation blending bureau score and utilisation
SELECT
    risk_category, COUNT(*) AS customers,
    ROUND(AVG(credit_score),0) AS avg_score
FROM (
    SELECT customer_id, credit_score,
        CASE
            WHEN credit_score >= 700 AND credit_utilization_ratio < 0.4 THEN 'Low Risk'
            WHEN credit_score >= 620                                     THEN 'Medium Risk'
            ELSE 'High Risk'
        END AS risk_category
    FROM credit_history
) t
GROUP BY risk_category
ORDER BY MIN(credit_score) DESC;

-- 7.9  Transaction behaviour analysis: 6-month cash-flow profile per customer
SELECT
    t.customer_id,
    COUNT(*)                                                   AS txn_count,
    SUM(CASE WHEN transaction_type='Credit' THEN amount ELSE 0 END) AS total_inflow,
    SUM(CASE WHEN transaction_type='Debit'  THEN amount ELSE 0 END) AS total_outflow,
    SUM(CASE WHEN category='EMI' THEN amount ELSE 0 END)       AS emi_outflow,
    ROUND(AVG(balance_after),0)                                AS avg_balance
FROM transactions t
GROUP BY t.customer_id
ORDER BY total_outflow DESC
LIMIT 50;

-- 7.10  Top predictors of loan default: default rate sliced by driver
--        (feeds feature-selection / EDA before modelling)
SELECT 'credit_score<600' AS driver,
       ROUND(100.0*AVG(la.loan_default),2) AS default_rate_pct, COUNT(*) AS n
FROM loan_applications la JOIN credit_history ch USING (customer_id)
WHERE la.loan_default IS NOT NULL AND ch.credit_score < 600
UNION ALL
SELECT 'dti>0.5',
       ROUND(100.0*AVG(loan_default),2), COUNT(*)
FROM loan_applications WHERE loan_default IS NOT NULL AND debt_to_income_ratio > 0.5
UNION ALL
SELECT 'prior_default>=1',
       ROUND(100.0*AVG(la.loan_default),2), COUNT(*)
FROM loan_applications la JOIN credit_history ch USING (customer_id)
WHERE la.loan_default IS NOT NULL AND ch.num_defaults_prior >= 1
UNION ALL
SELECT 'utilization>0.8',
       ROUND(100.0*AVG(la.loan_default),2), COUNT(*)
FROM loan_applications la JOIN credit_history ch USING (customer_id)
WHERE la.loan_default IS NOT NULL AND ch.credit_utilization_ratio > 0.8
ORDER BY default_rate_pct DESC;

-- =====================================================================
-- End of script
-- =====================================================================
