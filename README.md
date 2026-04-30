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

Example features:

- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- InternetService
- Contract
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn

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
