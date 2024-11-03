import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
from datetime import datetime
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


set_seed(42)


def train_and_evaluate(X_train, X_test, y_train, y_test, symptom, metric, params):
    print(f"\nTraining Adaptive Boosting for {symptom} - {metric}")
    print(f"Parameters: {params}")
    print(f"Input dimensions: {X_train.shape[1]}, Samples: {X_train.shape[0]}")

    model = AdaBoostClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=42,
        n_jobs=-1
    )

    print("Starting training...")
    train_start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start_time
    print(f"Training completed in {train_time:.2f}s")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")

    return accuracy, predictions, model


def cluster_based_oversampling(X, y, cluster_labels, minority_class=0):
    X_resampled = list(X)
    y_resampled = list(y)
    minority_samples = X[y == minority_class]
    minority_clusters = cluster_labels[y == minority_class]

    for cluster in np.unique(minority_clusters):
        cluster_samples = minority_samples[minority_clusters == cluster]
        num_samples_in_cluster = len(cluster_samples)

        if num_samples_in_cluster < 2:
            X_resampled.extend(cluster_samples)
            y_resampled.extend([minority_class] * num_samples_in_cluster)
            continue

        num_samples_to_generate = num_samples_in_cluster
        for _ in range(num_samples_to_generate):
            sample_1, sample_2 = random.sample(list(cluster_samples), 2)
            synthetic_sample = (sample_1 + sample_2) / 2
            X_resampled.append(synthetic_sample)
            y_resampled.append(minority_class)

    return np.array(X_resampled), np.array(y_resampled)


def analyze_all_symptoms(data_path):
    start_time = time.time()
    print(f"\nStarting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Hyperparameters to test
    param_combinations = [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 15}
    ]

    print("\nLoading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset dimensions: {df.shape}")

    symptoms = {
        'Acidity': {
            'severity': 'How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Acidity/Burning]',
            'frequency': 'How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Acidity]'
        },
        'Bloating': {
            'severity': 'How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Bloating]',
            'frequency': 'How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Bloating]'
        },
        'Constipation': {
            'severity': 'How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Constipation]',
            'frequency': 'How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Constipation]'
        },
        'Diarrhea': {
            'severity': 'How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Loose Motion/Diarrhea]',
        },
        'Flatulence': {
            'severity': 'How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Flatulence/Gas/Fart]',
            'frequency': 'How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Flatulence/Gas/Fart]'
        },
        'Burping': {
            'severity': 'How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Burping]',
            'frequency': 'How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Burping]'
        }
    }

    print("\nExtracting features...")
    feature_columns = [col for col in df.columns if ';' in col]
    X = df[feature_columns]
    print(f"Number of microbial features: {len(feature_columns)}")

    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Feature scaling completed")

    print("\nApplying PCA...")
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Number of components selected by PCA: {pca.n_components_}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    results = {}

    for symptom_name, symptom_cols in symptoms.items():
        print(f"\n{'=' * 20} Processing {symptom_name} {'=' * 20}")
        results[symptom_name] = {'severity': {}, 'frequency': {}}

        for metric in ['severity', 'frequency']:
            if metric in symptom_cols:
                metric_start_time = time.time()
                print(f"\nAnalyzing {symptom_name} - {metric}")

                target_column = symptom_cols[metric]
                y = df[target_column]

                print("\nClass distribution:")
                print(y.value_counts())

                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                n_classes = len(label_encoder.classes_)
                print(f"Number of classes: {n_classes}")
                print("Class mapping:", dict(zip(label_encoder.classes_, range(n_classes))))

                print("\nApplying cluster-based oversampling...")
                kmeans = KMeans(n_clusters=10, random_state=42)
                clusters = kmeans.fit_predict(X_pca)
                X_resampled, y_resampled = cluster_based_oversampling(X_pca, y_encoded, clusters)

                print(f'Original class distribution: {np.bincount(y_encoded.astype(int))}')
                print(f'Resampled class distribution: {np.bincount(y_resampled.astype(int))}')

                best_accuracy = 0
                best_params = None
                best_model = None
                best_predictions = None
                best_report = None

                print("\nCreating train/test split...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled,
                    y_resampled,
                    test_size=0.2,
                    stratify=y_resampled,
                    random_state=42,
                )
                print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

                for params in param_combinations:
                    print(f"\nTesting parameters: {params}")
                    accuracy, predictions, model = train_and_evaluate(
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        symptom=symptom_name,
                        metric=metric,
                        params=params
                    )

                    report = classification_report(
                        y_test,
                        predictions,
                        output_dict=True
                    )

                    if accuracy > best_accuracy:
                        print(f"New best accuracy: {accuracy:.4f}")
                        best_accuracy = accuracy
                        best_params = params
                        best_model = model
                        best_predictions = predictions
                        best_report = report

                metric_time = time.time() - metric_start_time
                print(f"\nCompleted {symptom_name} - {metric} in {metric_time:.2f}s")

                feature_importance = np.zeros(len(feature_columns))
                for i, importance in enumerate(best_model.feature_importances_):
                    feature_importance += importance * np.abs(pca.components_[i])

                results[symptom_name][metric] = {
                    'accuracy': best_accuracy,
                    'parameters': best_params,
                    'report': best_report,
                    'feature_importance': pd.DataFrame({
                        'feature': feature_columns,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False),
                    'class_distribution': dict(zip(label_encoder.classes_, range(n_classes)))
                }

                print("\nBest Results:")
                print(f"Accuracy: {best_accuracy:.4f}")
                print(f"Best parameters: {best_params}")
                print("\nClassification Report:")
                print(classification_report(y_test, best_predictions))
                print("\nTop 10 Most Important Features:")
                print(results[symptom_name][metric]['feature_importance'].head(20))
            else:
                print(f"Metric '{metric}' is not available for {symptom_name}. Skipping...")

    total_time = time.time() - start_time
    print(f"\nCompleted all symptoms analysis in {total_time:.2f}s")

    return results
    # Save results to a file
    print("\nSaving results...")
    result_summary = []
    for symptom, metrics in results.items():
        for metric, data in metrics.items():
            if data:
                accuracy = data['accuracy']
                best_params = data['parameters']
                feature_importance = data['feature_importance']

                # Store basic results for summary
                result_summary.append({
                    'Symptom': symptom,
                    'Metric': metric,
                    'Accuracy': accuracy,
                    'Best Parameters': best_params,
                    'Top Features': feature_importance.head(10).to_dict(orient='records')
                })

                # Save each metric’s feature importance separately
                feature_importance_filename = f"{symptom}_{metric}_feature_importance.csv"
                feature_importance.to_csv(feature_importance_filename, index=False)
                print(f"Feature importance saved to {feature_importance_filename}")

    # Convert summary results to a DataFrame and save as a summary CSV file
    summary_df = pd.DataFrame(result_summary)
    summary_df_filename = "symptom_analysis_summary.csv"
    summary_df.to_csv(summary_df_filename, index=False)
    print(f"\nSummary of all symptom analyses saved to {summary_df_filename}")

    return results

# Example of running the analysis
data_path = "/Users/anavgupta/Developer/models/dataset_new.csv"  # Replace with your actual file path
results = analyze_all_symptoms(data_path)