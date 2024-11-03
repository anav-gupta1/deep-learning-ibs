import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import time
from sklearn.metrics import roc_auc_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def calculate_multiclass_metrics(y_true, y_pred, label_encoder):
    """Calculate sensitivity and specificity for each class"""
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Binarize the labels for each class
    y_true_bin = np.eye(n_classes)[y_true]
    y_pred_bin = np.eye(n_classes)[y_pred]

    class_metrics = {}

    for i in range(n_classes):
        true_pos = np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 1))
        true_neg = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 0))
        false_pos = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 1))
        false_neg = np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 0))

        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0

        class_metrics[label_encoder.inverse_transform([i])[0]] = {
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    return class_metrics


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


class MatrixBasedAcidityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.abundance_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.interaction_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        abundance_features = self.abundance_network(x)
        interaction_features = self.interaction_network(x)
        combined = torch.cat([abundance_features, interaction_features], dim=1)
        attention_weights = torch.softmax(self.attention(combined), dim=1)
        weighted_features = combined * attention_weights
        output = self.final_layers(weighted_features)
        return output


def analyze_symptom_severity_matrix(data_path, symptom_name, n_components=1024):
    print(f"\nStarting analysis for {symptom_name.upper()} severity...")
    set_seed(42)

    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df.fillna(0)

    symptom_column = "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Burping]" #essentially putting the symptom column name here eitheer for frequency or severity
    print("Preparing features and target...")
    X = df.drop(columns=[symptom_column, "SAMPLE BARCODE"])
    y = df[symptom_column]

    print("\nInitial class distribution:")
    print(y.value_counts().sort_index())
    print(f"Total samples: {len(y)}")

    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)
    print(f"Number of classes: {n_classes}")
    print("Class mapping:", dict(zip(label_encoder.classes_, range(n_classes))))

    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Original feature dimension: {X_scaled.shape[1]}")

    print(f"\nApplying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Reduced feature dimension: {X_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    print("\nClustering and oversampling...")
    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    X_resampled, y_resampled = cluster_based_oversampling(X_pca, y_encoded, clusters)

    print(f"Final dataset size: {len(X_resampled)}")
    unique_classes, class_counts = np.unique(y_resampled, return_counts=True)
    print("Class distribution after oversampling:")
    for class_label, count in zip(unique_classes, class_counts):
        print(f"Class {label_encoder.inverse_transform([class_label])[0]}: {count}")

    print("\nPreparing tensors and dataloaders...")
    X_tensor = torch.FloatTensor(X_resampled).to(device)
    y_tensor = torch.LongTensor(y_resampled).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    print(f"\nTraining samples: {train_size}")
    print(f"Testing samples: {test_size}")

    print("\nInitializing model...")
    model = MatrixBasedAcidityNet(
        input_dim=n_components,
        hidden_dim=128,
        output_dim=n_classes
    ).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    print("\nStarting training...")
    best_accuracy = 0
    patience = 15
    patience_counter = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(100):
        model.train()
        total_loss = 0
        train_predictions = []
        train_labels = []

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_predictions)

        if (epoch + 1) % 5 == 0:
            model.eval()
            test_predictions = []
            test_labels = []

            with torch.no_grad():
                for features, labels in test_loader:
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    test_predictions.extend(predicted.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())

            test_acc = accuracy_score(test_labels, test_predictions)
            test_precision = precision_score(test_labels, test_predictions, average='weighted')
            test_recall = recall_score(test_labels, test_predictions, average='weighted')
            test_f1 = f1_score(test_labels, test_predictions, average='weighted')

            class_metrics = calculate_multiclass_metrics(test_labels, test_predictions, label_encoder)
            avg_sensitivity = np.mean([m['sensitivity'] for m in class_metrics.values()])
            avg_specificity = np.mean([m['specificity'] for m in class_metrics.values()])

            print(f"\nEpoch {epoch + 1}/100:")
            print(f"Loss: {total_loss / len(train_loader):.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Test Metrics:")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  Precision: {test_precision:.4f}")
            print(f"  Recall: {test_recall:.4f}")
            print(f"  F1-Score: {test_f1:.4f}")
            print(f"  Average Sensitivity: {avg_sensitivity:.4f}")
            print(f"  Average Specificity: {avg_specificity:.4f}")

            scheduler.step(test_acc)

            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model_state = model.state_dict()
                patience_counter = 0
                print(f"New best accuracy: {best_accuracy:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("\nEarly stopping triggered!")
                    break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\nPerforming final evaluation...")
    model.eval()
    final_predictions = []
    final_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            final_predictions.extend(predicted.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    final_class_metrics = calculate_multiclass_metrics(final_labels, final_predictions, label_encoder)

    metrics = {
        'accuracy': accuracy_score(final_labels, final_predictions),
        'precision': precision_score(final_labels, final_predictions, average='weighted'),
        'recall': recall_score(final_labels, final_predictions, average='weighted'),
        'f1': f1_score(final_labels, final_predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(final_labels, final_predictions),
        'class_metrics': final_class_metrics
    }

    return model, scaler, pca, label_encoder, metrics


if __name__ == "__main__":
    data_path = '/Users/anavgupta/Developer/models/dataset_new.csv'
    model, scaler, pca, label_encoder, metrics = analyze_symptom_severity_matrix(data_path, 'acidity')

    print("\nFinal Results for Acidity Severity:")
    print(f"Overall Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")

    print("\nClass-specific Metrics:")
    print("Severity Level | Sensitivity | Specificity")
    print("-" * 45)
    for severity, class_metrics in metrics['class_metrics'].items():
        print(f"Level {severity:8} | {class_metrics['sensitivity']:.4f} | {class_metrics['specificity']:.4f}")

    avg_sensitivity = np.mean([m['sensitivity'] for m in metrics['class_metrics'].values()])
    avg_specificity = np.mean([m['specificity'] for m in metrics['class_metrics'].values()])

    print(f"\nAverage Sensitivity: {avg_sensitivity:.4f}")
    print(f"Average Specificity: {avg_specificity:.4f}")

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])