# Credit Risk Modelling: End-to-End Project Explanation

## 1) Project Objective
The goal of this project was to build a **multi-class credit risk classification model** that predicts whether a customer belongs to one of four risk categories:
- **P1**: low risk / good customer
- **P2**: moderate risk
- **P3**: high risk
- **P4**: very high risk

In short, this is a supervised classification problem designed to support better lending decisions.

---

## 2) Data Sources
We worked with two datasets:
1. **Internal bank product dataset (DF1)**
2. **External bureau dataset from CIBIL (DF2)**

Both datasets included customer-level attributes such as credit score, demographics, education, loan characteristics, and repayment-related signals.

---

## 3) Data Cleaning Strategy
A major part of the project was handling missing and invalid values in a structured, rule-based way.

### DF1 (Internal Data)
- Around **40 rows** had invalid values (`-999`) in the last two columns.
- Since this count was very small, those rows were removed.

### DF2 (CIBIL Data)
- Missingness was much higher (in some columns up to ~70%).
- We applied a threshold policy based on 51,000 total rows:
  - If a column had **more than 10,000 missing values**, we dropped the whole column.
  - If a column had **fewer than 10,000 missing values**, we dropped only the affected rows.

This avoided heavy imputation on sensitive risk variables (for example, delinquency indicators).

---

## 4) Data Integration
- DF1 and DF2 were merged using **`Prospect ID`** as the key.
- We used an **inner join** to ensure only valid records present in both sources were retained.
- This also reduced null-related issues in downstream modelling.

---

## 5) Feature Engineering and Feature Selection

### 5.1 Categorical Feature Validation (Chi-Square Test)
For categorical variables (e.g., marital status, education, gender), we tested association with the target class (P1–P4):
- **H0 (null):** no association
- **H1 (alternative):** association exists

Decision rule:
- If **p-value < 0.05**, reject H0 and keep the feature.

Result:
- Categorical features showed meaningful association and were retained.

### 5.2 Numerical Feature Validation
We used a two-step process:

#### A) Multicollinearity Check (VIF)
- Computed **Variance Inflation Factor (VIF)** to detect correlated predictors.
- Threshold used: **VIF > 6** indicated problematic multicollinearity.
- Features were removed sequentially (not all at once) to preserve informative variables.

#### B) ANOVA for Target Relevance
- Applied ANOVA to numerical variables against target classes.
- Used **p-value < 0.05** as significance criterion.
- Numerical features reduced from **39 to 37** after this step.

---

## 6) Encoding Strategy
- **Ordinal categorical variables** (e.g., education level) were label encoded.
- **Nominal categorical variables** (e.g., gender, product type) were one-hot encoded.

This ensured compatibility with tree-based models while preserving category semantics.

---

## 7) Model Training and Baseline Evaluation
After preprocessing and feature selection:
- Data was split into **train** and **test** sets.
- Three baseline classifiers were trained:
  1. Decision Tree
  2. Random Forest
  3. XGBoost

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score

Result:
- **XGBoost** performed best, with approximately **78% accuracy** in one run.

Generalization check:
- Underfitting/overfitting was monitored by comparing training vs testing accuracy.

---

## 8) Hyperparameter Tuning (XGBoost)
Since XGBoost was strongest, we tuned it further.

- Built a hyperparameter grid (e.g., `max_depth`, `learning_rate`, and related settings).
- Generated all parameter combinations.
- Trained a model for each combination.
- Recorded training and testing accuracy for each run.

Outcome:
- Identified a parameter set with stronger test performance while controlling overfitting.

---

## 9) Deployment Approach
The tuned XGBoost model was serialized as a **pickle (`.pkl`) file** and deployed in two forms:

1. **Flask + HTML Web App**
   - User enters 8 key features.
   - Flask loads model and returns predicted risk class (P1–P4).

2. **Standalone Executable (`.exe`)**
   - Used for high-feature-input workflows (40+ features).
   - Allows prediction without requiring users to run Python manually.

---

## 10) Project Outcome
The project delivered a production-oriented credit risk pipeline that:
- cleans and integrates multi-source credit data,
- selects statistically relevant features,
- trains and tunes a robust classifier,
- and serves predictions through both web and executable interfaces.

---

## 11) Drift Monitoring Notes
Important drift types tracked conceptually for model maintenance:

- **Covariate Drift:** input feature distribution changes over time.
  - Example: customer profile mix shifts.

- **Concept Drift (most critical):** relationship between features and risk class changes.
  - Example: income previously correlated with low risk, but no longer does.

- **Label Drift:** class distribution changes.
  - Example: proportion of high-risk customers increases.

These drift concepts are central for post-deployment monitoring and retraining strategy.
