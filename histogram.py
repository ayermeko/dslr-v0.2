from core.visualizations import histo
from core.operations import validate

if __name__ == "__main__":
    try:
        df = validate("datasets/dataset_train.csv")
        histo(df)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

