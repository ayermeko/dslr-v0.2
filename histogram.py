from core.visualizations import histo
from core.operations import validate

def main():
    df = validate("/Users/alibiyermekov/MyProjects/dslr-v0.2/datasets/dataset_train.csv")
    try:
        histo(df)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
