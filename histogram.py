from core.visualizations import histo
from core.operations import validate

if __name__ == "__main__":
    try:
        df = validate("/Users/alibiyermekov/MyProjects/dslr-v0.2/datasets/dataset_trains.csv")
        histo(df)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

