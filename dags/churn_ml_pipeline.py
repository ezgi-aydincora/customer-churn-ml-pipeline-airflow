from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


BASE_PATH = Path("/opt/airflow/data")

RAW_FILE = BASE_PATH / "raw" / "telco_churn.csv"
PROCESSED_FILE = BASE_PATH / "processed" / "telco_churn_processed.csv"

MODEL_FILE = BASE_PATH / "models" / "baseline_churn_model.pkl"
BASELINE_METRICS_FILE = BASE_PATH / "results" / "baseline_metrics.csv"

SELECTED_FEATURES_FILE = BASE_PATH / "results" / "selected_features.csv"
CUCKOO_METRICS_FILE = BASE_PATH / "results" / "cuckoo_feature_selection_metrics.csv"

OPTIMIZED_MODEL_FILE = BASE_PATH / "models" / "optimized_churn_model.pkl"
OPTIMIZED_METRICS_FILE = BASE_PATH / "results" / "optimized_metrics.csv"
MODEL_COMPARISON_FILE = BASE_PATH / "results" / "model_comparison.csv"


PREDICTIONS_FILE = BASE_PATH / "results" / "churn_predictions.csv"


def check_dataset_exists():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {RAW_FILE}")

    print(f"Dataset bulundu: {RAW_FILE}")


def validate_columns():
    df = pd.read_csv(RAW_FILE)

    expected_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Eksik kolonlar var: {missing_columns}")

    print("Kolon kontrolü başarılı.")
    print(f"Toplam kolon sayısı: {len(df.columns)}")
    print(f"Toplam satır sayısı: {len(df)}")


def clean_churn_data():
    df = pd.read_csv(RAW_FILE)

    print("İlk veri boyutu:")
    print(df.shape)

    # customerID model için anlamlı değil
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges bazı satırlarda boş string gelebiliyor
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    print("Eksik değer sayıları:")
    print(df.isnull().sum())

    # Eksik TotalCharges kayıtlarını çıkarıyoruz
    df = df.dropna()

    # Target değişkenini 1/0 yapıyoruz
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Kategorik değişkenleri one-hot encode ediyoruz
    df_encoded = pd.get_dummies(df, drop_first=True)

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(PROCESSED_FILE, index=False)

    print("Temizlenmiş/model input veri boyutu:")
    print(df_encoded.shape)

    print(f"Processed dosya oluşturuldu: {PROCESSED_FILE}")


def train_baseline_model():
    df = pd.read_csv(PROCESSED_FILE)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": X.shape[1],
    }

    print("Baseline model metrikleri:")
    print(metrics)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame([metrics]).to_csv(BASELINE_METRICS_FILE, index=False)

    print(f"Model kaydedildi: {MODEL_FILE}")
    print(f"Metrikler kaydedildi: {BASELINE_METRICS_FILE}")
  
    
def cuckoo_search_feature_selection():
    df = pd.read_csv(PROCESSED_FILE)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    feature_names = X.columns.tolist()
    n_features = len(feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    n_nests = 10
    n_iterations = 8
    discovery_rate = 0.25

    rng = np.random.default_rng(42)

    def create_random_solution():
        solution = rng.integers(0, 2, size=n_features)

        if solution.sum() == 0:
            solution[rng.integers(0, n_features)] = 1

        return solution

    def evaluate_solution(solution):
        selected_indices = np.where(solution == 1)[0]
        selected_features = [feature_names[i] for i in selected_indices]

        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        model = RandomForestClassifier(
            n_estimators=80,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(X_train_selected, y_train)

        y_pred = model.predict(X_test_selected)

        f1 = f1_score(y_test, y_pred)

        # Çok fazla feature seçilmesini biraz cezalandırıyoruz
        feature_penalty = len(selected_features) / n_features * 0.01

        score = f1 - feature_penalty

        return score, f1, selected_features

    def levy_flight_binary(solution):
        new_solution = solution.copy()

        # Basitleştirilmiş Cuckoo Search yaklaşımı:
        # her iterasyonda rastgele birkaç feature seçimini tersine çeviriyoruz.
        n_changes = rng.integers(1, max(2, int(n_features * 0.2)))

        change_indices = rng.choice(n_features, size=n_changes, replace=False)
        new_solution[change_indices] = 1 - new_solution[change_indices]

        if new_solution.sum() == 0:
            new_solution[rng.integers(0, n_features)] = 1

        return new_solution

    nests = [create_random_solution() for _ in range(n_nests)]

    scores = []
    f1_scores = []
    selected_feature_sets = []

    for nest in nests:
        score, f1, selected_features = evaluate_solution(nest)
        scores.append(score)
        f1_scores.append(f1)
        selected_feature_sets.append(selected_features)

    best_index = int(np.argmax(scores))
    best_solution = nests[best_index]
    best_score = scores[best_index]
    best_f1 = f1_scores[best_index]
    best_features = selected_feature_sets[best_index]

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}")

        for i in range(n_nests):
            new_solution = levy_flight_binary(nests[i])
            new_score, new_f1, new_features = evaluate_solution(new_solution)

            if new_score > scores[i]:
                nests[i] = new_solution
                scores[i] = new_score
                f1_scores[i] = new_f1
                selected_feature_sets[i] = new_features

            if new_score > best_score:
                best_solution = new_solution
                best_score = new_score
                best_f1 = new_f1
                best_features = new_features

        # En kötü yuvaların bir kısmını yeniden oluşturuyoruz
        n_abandon = max(1, int(discovery_rate * n_nests))
        worst_indices = np.argsort(scores)[:n_abandon]

        for idx in worst_indices:
            new_solution = create_random_solution()
            new_score, new_f1, new_features = evaluate_solution(new_solution)

            nests[idx] = new_solution
            scores[idx] = new_score
            f1_scores[idx] = new_f1
            selected_feature_sets[idx] = new_features

            if new_score > best_score:
                best_solution = new_solution
                best_score = new_score
                best_f1 = new_f1
                best_features = new_features

        print(f"Best F1 so far: {best_f1}")
        print(f"Selected feature count: {len(best_features)}")

    SELECTED_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUCKOO_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    selected_features_df = pd.DataFrame({
        "selected_feature": best_features
    })

    selected_features_df.to_csv(SELECTED_FEATURES_FILE, index=False)

    metrics = {
        "best_f1_score": best_f1,
        "best_optimization_score": best_score,
        "selected_feature_count": len(best_features),
        "total_feature_count": n_features,
        "n_nests": n_nests,
        "n_iterations": n_iterations,
        "discovery_rate": discovery_rate,
    }

    pd.DataFrame([metrics]).to_csv(CUCKOO_METRICS_FILE, index=False)

    print("Cuckoo Search tamamlandı.")
    print(f"Best F1-score: {best_f1}")
    print(f"Selected features: {best_features}")
    print(f"Seçilen feature dosyası: {SELECTED_FEATURES_FILE}")
    print(f"Cuckoo metrik dosyası: {CUCKOO_METRICS_FILE}")    
    

def train_optimized_model():
    df = pd.read_csv(PROCESSED_FILE)

    selected_features_df = pd.read_csv(SELECTED_FEATURES_FILE)
    selected_features = selected_features_df["selected_feature"].tolist()

    X = df[selected_features]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    optimized_metrics = {
        "model": "optimized_random_forest_cuckoo_features",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": X.shape[1],
    }

    print("Optimized model metrikleri:")
    print(optimized_metrics)

    OPTIMIZED_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    OPTIMIZED_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OPTIMIZED_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    optimized_metrics_df = pd.DataFrame([optimized_metrics])
    optimized_metrics_df.to_csv(OPTIMIZED_METRICS_FILE, index=False)

    baseline_metrics_df = pd.read_csv(BASELINE_METRICS_FILE)
    baseline_metrics_df.insert(0, "model", "baseline_random_forest")

    comparison_df = pd.concat(
        [baseline_metrics_df, optimized_metrics_df],
        ignore_index=True,
    )

    comparison_df.to_csv(MODEL_COMPARISON_FILE, index=False)

    print(f"Optimized model kaydedildi: {OPTIMIZED_MODEL_FILE}")
    print(f"Optimized metrikler kaydedildi: {OPTIMIZED_METRICS_FILE}")
    print(f"Model karşılaştırması kaydedildi: {MODEL_COMPARISON_FILE}")


def generate_churn_predictions():
    df = pd.read_csv(PROCESSED_FILE)

    selected_features_df = pd.read_csv(SELECTED_FEATURES_FILE)
    selected_features = selected_features_df["selected_feature"].tolist()

    with open(OPTIMIZED_MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    X = df[selected_features]

    churn_prediction = model.predict(X)
    churn_probability = model.predict_proba(X)[:, 1]

    predictions_df = pd.DataFrame({
        "customer_row_id": range(1, len(df) + 1),
        "churn_prediction": churn_prediction,
        "churn_probability": churn_probability,
    })

    predictions_df["risk_segment"] = pd.cut(
        predictions_df["churn_probability"],
        bins=[0, 0.33, 0.66, 1],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

    print("Prediction dosyası oluşturuldu.")
    print(f"Toplam tahmin sayısı: {len(predictions_df)}")
    print(predictions_df.head())
    print(f"Dosya yolu: {PREDICTIONS_FILE}")

    
    
with DAG(
    dag_id="churn_ml_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "churn", "cuckoo-search"],
) as dag:

    task_check_dataset_exists = PythonOperator(
        task_id="check_dataset_exists",
        python_callable=check_dataset_exists,
    )

    task_validate_columns = PythonOperator(
        task_id="validate_columns",
        python_callable=validate_columns,
    )

    task_clean_churn_data = PythonOperator(
        task_id="clean_churn_data",
        python_callable=clean_churn_data,
    )
    
    task_train_baseline_model = PythonOperator(
    task_id="train_baseline_model",
    python_callable=train_baseline_model,
)
    
    task_cuckoo_search_feature_selection = PythonOperator(
    task_id="cuckoo_search_feature_selection",
    python_callable=cuckoo_search_feature_selection,
)
    
    task_train_optimized_model = PythonOperator(
    task_id="train_optimized_model",
    python_callable=train_optimized_model,
)
    
    task_generate_churn_predictions = PythonOperator(
    task_id="generate_churn_predictions",
    python_callable=generate_churn_predictions,
)



    task_check_dataset_exists >> task_validate_columns >> task_clean_churn_data >> task_train_baseline_model >> task_cuckoo_search_feature_selection >> task_train_optimized_model >> task_generate_churn_predictions
