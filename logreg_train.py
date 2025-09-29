import sys
import pandas as pd
import json
from core.operations import validate
from core.model.preprocessing import split_randomize

def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: missing .csv or arg issue.")        
        
        # loading selected features
        with open('hyperparams.json', 'r') as f:
            hyperparams = json.load(f)
        
        # validating arg path, and converting into df form custom
        # it also reads csv file
        df = validate(sys.argv[1])
        df_dict = {}
        for col_name, col_data in df.items():
            df_dict[col_name] = col_data

        df = pd.DataFrame(df_dict)
        
        selected_features = []
        for feature, is_selected in hyperparams['features'].items():
            if is_selected:
                selected_features.append(feature)
        
        
        features_to_keep = ['Hogwarts House'] + selected_features
        df = df[features_to_keep]
        
        df = df.dropna(subset=selected_features)
        
        # Extract features and target
        X = df[selected_features].values.astype(float)  # Use selected features directly
        y = df.values[:,  0]
        
        
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()