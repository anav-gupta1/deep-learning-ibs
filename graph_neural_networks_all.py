import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
import time
from torch_geometric.utils import add_self_loops, degree

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class SymptomGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(SymptomGNN, self).__init__()
        
        # Graph Convolution Layers with edge weights
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Final classification layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
        # First Graph Convolution Block with edge weights
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second Graph Convolution Block
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third Graph Convolution Block
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_sum], dim=1)
        
        # Final classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def construct_cooccurrence_graph(features, threshold=0.3):
    """
    Construct a graph based on symptom co-occurrence patterns
    """
    n_samples, n_features = features.shape
    
    # Initialize co-occurrence matrix
    cooccurrence_matrix = np.zeros((n_features, n_features))
    
    # Calculate support for each symptom
    support = np.sum(features > 0, axis=0) / n_samples
    
    # Calculate co-occurrence scores
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Find samples where both symptoms occur
            cooccurrence = np.sum((features[:, i] > 0) & 
                                (features[:, j] > 0)) / n_samples
            
            # Calculate lift score
            if support[i] * support[j] > 0:
                lift = cooccurrence / (support[i] * support[j])
            else:
                lift = 0
                
            # Calculate Jaccard similarity
            jaccard = cooccurrence / (support[i] + support[j] - cooccurrence + 1e-10)
            
            # Combine metrics
            combined_score = 0.7 * lift + 0.3 * jaccard
            
            cooccurrence_matrix[i, j] = combined_score
            cooccurrence_matrix[j, i] = combined_score
    
    # Create edges where co-occurrence is above threshold
    edges = np.where(cooccurrence_matrix > threshold)
    edge_weights = cooccurrence_matrix[edges]
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_weight

def cluster_based_oversampling(features, labels, cluster_labels, minority_class):
    features_resampled = list(features)
    labels_resampled = list(labels)
    
    minority_samples = features[labels == minority_class]
    minority_clusters = cluster_labels[labels == minority_class]
    
    for cluster in np.unique(minority_clusters):
        cluster_samples = minority_samples[minority_clusters == cluster]
        num_samples_in_cluster = len(cluster_samples)
        
        if num_samples_in_cluster < 2:
            features_resampled.extend(cluster_samples)
            labels_resampled.extend([minority_class] * num_samples_in_cluster)
            continue
        
        num_samples_to_generate = num_samples_in_cluster
        for _ in range(num_samples_to_generate):
            idx1, idx2 = random.sample(range(num_samples_in_cluster), 2)
            synthetic_sample = (cluster_samples[idx1] + cluster_samples[idx2]) / 2
            features_resampled.append(synthetic_sample)
            labels_resampled.append(minority_class)
    
    return np.array(features_resampled), np.array(labels_resampled)

def prepare_graph_data(df, symptom_column, n_components=128, threshold=0.3):
    # Separate features and target
    X = df.drop(columns=[symptom_column, "SAMPLE BARCODE"])
    y = df[symptom_column]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Perform clustering for oversampling
    print("\nPerforming clustering for oversampling...")
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    # Apply oversampling
    X_resampled, y_resampled = cluster_based_oversampling(
        X_pca, y_encoded, cluster_labels, minority_class=0
    )
    
    # Construct graph using co-occurrence
    edge_index, edge_weight = construct_cooccurrence_graph(X_resampled, threshold)
    
    # Create graph data objects
    graph_data_list = []
    for i in range(len(X_resampled)):
        data = Data(
            x=torch.tensor(X_resampled[i], dtype=torch.float).unsqueeze(0),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([y_resampled[i]], dtype=torch.long)
        )
        graph_data_list.append(data)
    
    return graph_data_list, label_encoder, scaler, pca

def analyze_symptom_severity_gnn(data_path, symptom_name, hidden_dim=128, n_components=128):
    print(f"\nStarting GNN analysis for {symptom_name.upper()} severity...")
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df.fillna(0)
    
    symptom_column = "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Burping]"
    
    # Prepare graph data
    graph_data_list, label_encoder, scaler, pca = prepare_graph_data(
        df, symptom_column, n_components
    )
    
    # Print class distribution
    y_all = [data.y.item() for data in graph_data_list]
    print("\nClass distribution after oversampling:")
    for class_label in np.unique(y_all):
        count = sum(1 for y in y_all if y == class_label)
        print(f"Class {label_encoder.inverse_transform([class_label])[0]}: {count}")
    
    # Split data
    train_data, test_data = train_test_split(graph_data_list, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # Initialize model
    num_features = graph_data_list[0].x.size(1)
    num_classes = len(label_encoder.classes_)
    model = SymptomGNN(num_features, hidden_dim, num_classes).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_accuracy = 0
    patience = 15
    patience_counter = 0
    start_time = time.time()
    
    print("\nStarting training...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        train_pred = []
        train_true = []
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            train_pred.extend(pred.cpu().numpy())
            train_true.extend(data.y.cpu().numpy())
        
        # Validation
        model.eval()
        test_pred = []
        test_true = []
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                test_pred.extend(pred.cpu().numpy())
                test_true.extend(data.y.cpu().numpy())
        
        # Calculate metrics
        train_acc = accuracy_score(train_true, train_pred)
        test_acc = accuracy_score(test_true, test_pred)
        
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch + 1}/100:")
            print(f"Loss: {total_loss / len(train_loader):.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
        
        scheduler.step(test_acc)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Load best model and compute final metrics
    model.load_state_dict(best_model_state)
    model.eval()
    final_pred = []
    final_true = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            final_pred.extend(pred.cpu().numpy())
            final_true.extend(data.y.cpu().numpy())
    
    # Calculate final metrics
    metrics = {
        'accuracy': accuracy_score(final_true, final_pred),
        'precision': precision_score(final_true, final_pred, average='weighted'),
        'recall': recall_score(final_true, final_pred, average='weighted'),
        'f1': f1_score(final_true, final_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(final_true, final_pred)
    }
    
    return model, metrics, label_encoder, scaler, pca

if __name__ == "__main__":
    data_path = '/path/to/your/dataset.csv'
    model, metrics, label_encoder, scaler, pca = analyze_symptom_severity_gnn(
        data_path, 'symptom_severity'
    )
    
    print("\nFinal Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])