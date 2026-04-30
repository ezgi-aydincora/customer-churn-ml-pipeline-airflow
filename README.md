# Customer Churn ML Pipeline with Airflow and Cuckoo Search

## Project Overview

This project is an end-to-end machine learning pipeline developed to predict customer churn using Apache Airflow.

The pipeline includes data validation, data preprocessing, baseline model training, feature selection with Cuckoo Search optimization, optimized model training, and churn prediction generation.

The main goal of this project is to combine workflow orchestration, machine learning, and optimization techniques in a practical business problem.

## Problem Statement

Customer churn is a critical business problem, especially in subscription-based industries such as telecommunications, banking, and SaaS.

The objective of this project is to predict whether a customer is likely to churn and to generate churn probabilities that can support customer retention actions.

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset includes customer demographic information, account details, service usage information, payment method, monthly charges, total charges, and churn status.

### Example Features

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `InternetService`
- `Contract`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

> Note: The dataset is not included in this repository.  
> It can be downloaded from Kaggle and placed under the `data/raw/` folder.

## Pipeline Architecture

The machine learning workflow is orchestrated with Apache Airflow.

```text
check_dataset_exists
        ↓
validate_columns
        ↓
clean_churn_data
        ↓
train_baseline_model
        ↓
cuckoo_search_feature_selection
        ↓
train_optimized_model
        ↓
generate_churn_predictions
```

## Methodology

### 1. Data Validation

The pipeline first checks whether the dataset exists in the expected directory.  
Then it validates whether the required columns are available in the dataset.

### 2. Data Cleaning and Preprocessing

The preprocessing step includes:

- Removing unnecessary columns such as `customerID`
- Converting `TotalCharges` to numeric format
- Handling missing values
- Encoding the target variable `Churn`
- Applying one-hot encoding to categorical variables

### 3. Baseline Model

A Random Forest Classifier is trained using all available features.  
This model is used as the baseline for performance comparison.

### 4. Feature Selection with Cuckoo Search

Cuckoo Search optimization is applied for feature selection.

Each solution represents a feature subset:

```text
[1, 0, 1, 1, 0, 1, ...]
```

Where:

- `1` means the feature is selected
- `0` means the feature is not selected

The objective is to find a feature subset that improves the F1-score while reducing the number of features.

### 5. Optimized Model

After selecting the best feature subset, a new Random Forest model is trained using only the selected features.

### 6. Churn Prediction

The final model generates:

- Churn prediction
- Churn probability
- Risk segment: low, medium, high

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Feature Count |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Random Forest | 0.789 | 0.632 | 0.492 | 0.553 | 0.818 | 30 |
| Cuckoo Optimized Random Forest | 0.746 | 0.516 | 0.684 | 0.589 | 0.792 | 12 |

## Key Findings

- The optimized model reduced the number of features from 30 to 12.
- F1-score improved from 0.553 to 0.589.
- Recall improved from 0.492 to 0.684.
- The optimized model is better at identifying customers with churn risk.

In churn prediction problems, recall is especially important because missing high-risk customers can lead to lost revenue.

## Project Outputs

The pipeline generates the following outputs:

| File | Description |
|---|---|
| `telco_churn_processed.csv` | Cleaned and preprocessed dataset |
| `baseline_churn_model.pkl` | Baseline Random Forest model |
| `baseline_metrics.csv` | Baseline model performance metrics |
| `selected_features.csv` | Features selected by Cuckoo Search |
| `cuckoo_feature_selection_metrics.csv` | Feature selection optimization results |
| `optimized_churn_model.pkl` | Optimized Random Forest model |
| `optimized_metrics.csv` | Optimized model performance metrics |
| `model_comparison.csv` | Baseline vs optimized model comparison |
| `churn_predictions.csv` | Customer-level churn predictions and risk segments |

## Technologies Used

- Python
- Apache Airflow
- Docker
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Cuckoo Search Optimization

## Repository Structure

```text
customer-churn-ml-pipeline-airflow/
│
├── dags/
│   └── churn_ml_pipeline.py
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
│
├── docs/
│   └── project_documentation.pdf
│
├── results/
│   ├── baseline_metrics.csv
│   ├── optimized_metrics.csv
│   ├── model_comparison.csv
│   ├── selected_features.csv
│   └── sample_churn_predictions.csv
│
├── models/
│   └── README.md
│
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ezgi-aydincora/customer-churn-ml-pipeline-airflow.git
cd customer-churn-ml-pipeline-airflow
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download the Telco Customer Churn dataset and place it under:

```text
data/raw/telco_churn.csv
```

### 4. Start Airflow

If using Docker, start Airflow with:

```bash
docker compose up
```

### 5. Run the DAG

Open the Airflow UI and trigger the DAG manually:

```text
churn_ml_pipeline
```

## Future Improvements

- Add MLflow for experiment tracking and model versioning
- Add SHAP analysis for model explainability
- Create a Power BI dashboard for churn risk segments
- Add hyperparameter optimization with Cuckoo Search
- Schedule the pipeline for periodic execution
- Add automated data quality checks

## Business Value

This project demonstrates how a machine learning pipeline can support customer retention strategies by identifying customers with high churn risk.

The generated risk segments can help business teams prioritize retention campaigns and take proactive actions before customers leave.

## Author

**Ezgi Aydın Cora**  
Data Analyst / Analytics Engineering Portfolio Project
