
# Customer Churn Intelligence

An end-to-end Machine Learning project that predicts customer churn risk and provides business-oriented insights to support customer retention decisions.

## Project Overview

Customer churn is a major business challenge for subscription-based companies.

The goal of this project is to build a Machine Learning system that predicts whether a customer is likely to churn based on demographic information, services, contract details, and billing information.

The project includes data preprocessing, exploratory data analysis, machine learning model comparison, hyperparameter tuning, model interpretation, and an interactive Streamlit web application.

## Business Problem

Businesses need to identify customers who may be at risk of leaving so they can take appropriate retention actions.

This system provides a predicted churn probability for individual customers and presents the prediction through an interactive web application.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information including:

- Demographics
- Tenure
- Contract type
- Internet services
- Support services
- Payment method
- Monthly charges
- Total charges
- Churn status

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Encoding Categorical Features
6. Train/Test Split
7. Machine Learning Model Training
8. Model Evaluation
9. Hyperparameter Tuning
10. Model Interpretation
11. Model Saving
12. Interactive Web Application

## Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

Logistic Regression was used as the main model for the final application because it achieved the highest ROC-AUC among the evaluated models.

## Model Performance

The final evaluation results on the test set were:

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1 Score | 60.40% |
| ROC-AUC | 84.19% |

Hyperparameter tuning was also performed using GridSearchCV with 5-fold cross-validation.

The tuned model did not improve the test-set ROC-AUC compared with the baseline Logistic Regression model, so the baseline model was retained for the application.

## Key Features

The model uses customer information such as:

- Tenure
- Monthly Charges
- Total Charges
- Contract
- Internet Service
- Online Security
- Tech Support
- Payment Method
- Streaming Services
- Customer demographics

## Web Application

The project includes an interactive Streamlit application where users can enter customer information and receive:

- Churn prediction
- Churn probability
- Risk classification
- Top model-driven churn factors
- Business-oriented interpretation

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── src/
│   ├── customer_churn_model.pkl
│   └── feature_importance.csv
│
├── app.py
├── requirements.txt
└── README.md
````

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib
* Jupyter Notebook

## How to Run

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project:

```bash
cd customer-churn-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

## Disclaimer

The predicted churn probability represents the output of a Machine Learning model and should not be interpreted as a guarantee that a customer will churn.

## Author

Ziad Mahana

Artificial Intelligence Student | Aspiring Machine Learning Engineer
