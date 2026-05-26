End-to-End Insurance Risk Analytics & Predictive Modeling
Project Overview

This project focuses on insurance risk analytics and predictive modeling using historical South African auto insurance data from AlphaCare Insurance Solutions (ACIS). The objective is to identify low-risk customer segments, evaluate claim behavior, and build predictive models that support risk-based pricing.

The project combines:

Exploratory Data Analysis (EDA)
Statistical Hypothesis Testing
Data Version Control (DVC)
Machine Learning Modeling
Business-Focused Recommendations
Business Objective

AlphaCare Insurance Solutions (ACIS) aims to improve pricing accuracy and optimize marketing strategies by leveraging historical insurance claims data.

The primary goals are:

Identify low-risk customer segments
Understand factors influencing claims and profitability
Build predictive models for claim severity and claim probability
Support dynamic, risk-based premium pricing
Project Structure
insurance-risk-analytics/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── raw/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
│
├── reports/
│   └── interim_report.md
│
├── tests/
│
├── .dvc/
├── .gitignore
├── dvc.yaml
├── requirements.txt
└── README.md
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
SHAP
DVC
GitHub Actions
Task 1: Exploratory Data Analysis

Performed:

Data summarization
Missing value analysis
Univariate analysis
Multivariate analysis
Geographic trend analysis
Outlier detection
Loss ratio analysis

Key metrics analyzed:

TotalPremium
TotalClaims
Loss Ratio
Margin
Task 2: Data Version Control (DVC)

DVC was configured to:

Track datasets
Maintain reproducibility
Support versioned data pipelines
Store datasets outside Git

Commands used:

pip install dvc


dvc init


dvc add data/raw/insurance_data.csv


dvc remote add -d localstorage ../dvc_storage


dvc push
How to Run
Clone Repository
git clone <repo-link>
cd insurance-risk-analytics
Create Virtual Environment
python -m venv venv
Activate Environment

Windows:

.\venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Launch Jupyter Notebook
jupyter notebook
