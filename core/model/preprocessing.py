import json
import pandas as pd
import numpy as np

def clear_data(filepath: str):
    with open('hyperparams.json', 'r') as f:
        hyperparams = json.load(f)
    df = pd.read_csv(filepath)
    
    selected_features = []
    for feature, is_selected in hyperparams['features'].items():
        if is_selected:
            selected_features.append(feature)
    

    features_to_keep = ['Hogwarts House'] + selected_features
    df = df[features_to_keep]
    df = df.dropna(subset=selected_features)
    X = df[selected_features].values.astype(float)
    y = df.values[:,  0]
    
    return X, y

def split_randomize(X, y, test_size=0.3, random_state=None):
    """Split arrays into random train and test subsets"""
    if random_state:
        np.random.seed(random_state)
    
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []    

    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        
        np.random.shuffle(class_indices)

        n_test = int(len(class_indices) * test_size)
        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]