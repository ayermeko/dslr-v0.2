import sys
import pandas as pd
from core.model.preprocessing import clear_data
from core.model.classify import LogisticRegression

def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: python logreg_predict.py dataset_test.csv")
        
        X_test, _ = clear_data(filepath=sys.argv[1])
        
        model = LogisticRegression()
        model.load_model('weights.json')
        
        X_test_norm = model.transform(X_test)
        predictions = model.predict(X_test_norm)
        
        # create output DataFrame with required format
        output_df = pd.DataFrame({
            'Index': range(len(predictions)),
            'Hogwarts House': predictions
        })
        output_df.to_csv('houses.csv', index=False)
        print(f"Predictions saved to houses.csv ({len(predictions)} samples)")
        
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()