# Customer Churn Prediction (Project 02)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Portfolio Project 02**: An end-to-end Machine Learning classification project predicting customer churn using the Telco Customer Churn dataset. Designed specifically for a fresher Data Science portfolio with clean Python code, beginner-friendly explanations, and real-world business context.

---

## 1. Project Introduction

In the telecommunications industry, customer churn (when a subscriber cancels their service) is a major threat to profitability. Retaining an existing customer is estimated to be **5 to 25 times cheaper** than acquiring a new one. 

When telecommunication providers can anticipate which customers are at risk of leaving, the retention team can intervene early with tailored incentives, customized discounts, or proactive support before the customer switches to a competitor.

This project delivers a complete, beginner-friendly Machine Learning workflow that explores customer demographics, analyzes service subscriptions, handles real-world data quality issues, trains classification models, tunes hyperparameters, and evaluates performance using business-relevant metrics.

---

## 2. Project Objective

The primary objectives of this project are:
1. **Analyze Customer Behavior**: Discover the primary drivers and risk factors behind customer churn through Exploratory Data Analysis (EDA) and visualizations.
2. **Build Classification Models**: Train baseline and tuned versions of **Logistic Regression** and **Decision Tree Classifier**.
3. **Apply Resampling & Tuning**: Implement 5-Fold Cross-Validation, `GridSearchCV`, and `RandomizedSearchCV` to optimize model parameters and prevent overfitting.
4. **Evaluate Beyond Accuracy**: Demonstrate why Accuracy alone is insufficient on imbalanced datasets by evaluating Confusion Matrices, Precision, Recall, and F1-Scores.
5. **Deliver Actionable Business Recommendations**: Formulate practical retention strategies based on empirical model findings.

---

## 3. Dataset Description

* **Filename**: `Telco-Customer-Churn.csv`
* **Source**: IBM Telco Customer Churn Dataset
* **Observations**: 7,043 customers
* **Features**: 21 original columns (demographics, services, account details, and charges)
* **Target Variable**: `Churn` (`Yes` = Customer left, `No` = Customer stayed)

### Feature Breakdown:
* **Customer Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Account Information**: `tenure` (months with company), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Subscribed Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
* **Customer Identifier**: `customerID` (removed during data cleaning as it carries no predictive signal)

---

## 4. Technologies Used

* **Programming Language**: Python
* **Data Manipulation**: Pandas, NumPy
* **Data Visualization**: Matplotlib, Seaborn
* **Machine Learning**: Scikit-Learn
* **Development Environment**: Jupyter Notebook, IPykernel

---

## 5. Machine Learning Algorithms

This project focuses on two foundational classification algorithms:

### 1. Logistic Regression
* **Algorithm Type**: Linear Classification Model.
* **Mechanism**: Computes a linear weighted combination of features ($z = WX + b$) and passes it through the **Sigmoid (Logistic) Function** $\sigma(z) = \frac{1}{1 + e^{-z}}$, producing a probability between 0 and 1.
* **Decision Rule**: Classifies customers as churn if probability $\ge 0.50$.
* **Role**: Serves as our primary linear benchmark model.

### 2. Decision Tree Classifier
* **Algorithm Type**: Non-linear, Tree-based Model.
* **Mechanism**: Recursively partitions data using flowchart-like decision rules that minimize node impurity (Gini Impurity).
* **Role**: Demonstrates tree-based splitting, visual interpretability, the risk of overfitting when unconstrained, and how hyperparameter tuning (`max_depth`, `min_samples_split`) regularizes the model.

---

## 6. Project Workflow

The project follows a systematic 22-step Data Science lifecycle:

```
[1. Import Libraries] ➔ [2. Load Dataset] ➔ [3. Dataset Preview] ➔ [4. Dataset Info]
        │
[5. Missing Values Check] ➔ [6. Duplicate Check] ➔ [7. Basic Data Cleaning]
        │
[8. Exploratory Data Analysis] ➔ [9. Data Visualization (6 Key Plots)]
        │
[10. Encoding (Label + One-Hot)] ➔ [11. Train-Test Split (80/20, Stratified)]
        │
[12. Logistic Regression] ➔ [13. Decision Tree] ➔ [14. 5-Fold Cross-Validation]
        │
[15. Hyperparameter Tuning Concepts] ➔ [16. GridSearchCV (LR)] ➔ [17. RandomizedSearchCV (DT)]
        │
[18. Test Set Accuracy] ➔ [19. Confusion Matrices] ➔ [20. Classification Reports]
        │
[21. Model Comparison & Imbalance Analysis] ➔ [22. Project Conclusion & Business Strategies]
```

### Key Data Cleaning Highlights:
* Stripped unnecessary leading and trailing whitespace characters.
* Detected and converted `TotalCharges` from text (`object`) to numeric format.
* Handled 11 missing values in `TotalCharges` corresponding to brand-new customers with `tenure == 0` by imputing `0`.
* Dropped `customerID` to prevent data leakage and noise.

---

## 7. Model Performance & Comparison

All models were evaluated on the held-out 20% test dataset (**1,409 customers**: 1,035 No Churn, 374 Churn).

| Model Name | Test Accuracy | Churn Precision | Churn Recall | Churn F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **80.48%** | **0.6542** | **0.5615** | **0.6043** |
| **Decision Tree (Baseline)** | 72.60% | 0.4838 | 0.4786 | 0.4812 |
| **Logistic Regression (Tuned)** | 80.20% | 0.6489 | 0.5535 | 0.5974 |
| **Decision Tree (Tuned)** | 78.71% | 0.6682 | 0.3930 | 0.4950 |

### Why Accuracy Alone is Insufficient for Churn Prediction:
1. **Class Imbalance**: Approximately 73.5% of customers do not churn. A naive model predicting that *nobody ever churns* achieves **73.5% accuracy**, yet provides **zero business value** because it catches zero churners.
2. **Business Cost Asymmetry**:
   * **False Negative (FN)**: Missing a customer who leaves means losing their future recurring revenue and customer lifetime value (LTV).
   * **False Positive (FP)**: Inconveniently offering a loyalty perk or discount to a customer who planned to stay anyway carries a minor cost.
3. **Winner Analysis**:
   * While the **Tuned Decision Tree** reaches **78.71% accuracy**, its Churn Recall drops to **39.30%** (missing 227 out of 374 churners).
   * **Logistic Regression** achieves **80.48% accuracy** and a much higher Churn Recall of **56.15%** (identifying 210 churners). It is the recommended champion model for production deployment.

---

## 8. Business Recommendations

Based on empirical patterns uncovered during EDA and model evaluation:
1. **Target Month-to-Month Contracts**: Over 42% of month-to-month subscribers churn. Provide promotional pricing or streaming bundles to incentivize moving to 1- or 2-year contracts.
2. **Early Tenure Onboarding (First 1–6 Months)**: Churn heavily concentrates in early customer tenure. Introduce proactive milestone check-ins at 30 and 90 days.
3. **Incentivize Automated Billing**: Customers paying via Electronic Check churn at ~45% compared to ~15% for automatic credit card or bank transfer. Offer a small recurring billing discount for automated payments.
4. **Value Bundles for High Monthly Spenders**: Churned customers pay significantly higher monthly rates (~$74 vs ~$61). Proactively review accounts paying over $80/month for retention discounts.

---

## 9. Project Structure

```
Customer Churn Prediction/
│
├── Customer_Churn_Prediction.ipynb   # Complete, executed Jupyter Notebook with all 22 sections
├── Telco-Customer-Churn.csv          # IBM Telco Customer Churn dataset
├── requirements.txt                  # Python dependencies
├── README.md                         # Detailed project documentation and interview guide
└── read.txt                          # Project notes and guidelines
```


## 10. Interview Preparation Tips

When presenting this project in a fresher Data Science interview:
* **Explain Data Cleaning**: Mention how missing `TotalCharges` values were concealed as blank whitespace strings (`" "`) and why filling them with 0 was domain-appropriate (customers had zero tenure).
* **Explain Data Leakage**: Highlight that preprocessing and scaling must be fit on training data only, and `customerID` was removed because identifiers cause artificial memorization.
* **Explain Stratification**: Explain why `stratify=y` is necessary in `train_test_split` for imbalanced targets.
* **Explain Overfitting in Trees**: Explain why the baseline Decision Tree scored 72.6% while the tuned tree reached 78.7% (pruning with `max_depth` and `min_samples_split`).
* **Explain the Precision-Recall Trade-off**: Explain why Recall for the positive class (Churn) matters more to telecom stakeholders than raw accuracy.

---

## License
This project is open-source and available under the [MIT License](LICENSE).
