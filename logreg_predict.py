import sys
import pandas as pd
from core.model.preprocessing import clear_data
from core.model.classify import LogisticRegression
import json


def bye_data(filepath: str):
    with open('hyperparams.json', 'r') as f:
        hyperparams = json.load(f)
    
    df = pd.read_csv(filepath)
    
    selected_features = []
    for feature, is_selected in hyperparams['features'].items():
        if is_selected:
            selected_features.append(feature)
    

    features_to_keep = ['Index'] + selected_features  # Use Index instead of Hogwarts House for test data
    df = df[features_to_keep]
    
    # Fill missing values with median instead of dropping rows
    for feature in selected_features:
        median_val = df[feature].median()
        df[feature] = df[feature].fillna(median_val)
    
    X = df[selected_features].values.astype(float)
    indices = df['Index'].values
    
    return X, indices

def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: python logreg_predict.py dataset_test.csv")
        
        X_test, indices = bye_data(filepath=sys.argv[1])
        
        model = LogisticRegression()
        model.load_model('weights.json')
        
        X_test_norm = model.transform(X_test)
        predictions = model.predict(X_test_norm)
        
        # create output DataFrame with required format
        output_df = pd.DataFrame({
            'Index': indices.astype(int),
            'Hogwarts House': predictions
        })
        output_df.to_csv('houses.csv', index=False)
        print(f"Predictions saved to houses.csv ({len(predictions)} samples)")
        
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()