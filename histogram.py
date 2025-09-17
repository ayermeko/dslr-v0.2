from core.visualizations import histo
from core.operations import validate

def main():
    df = validate("/Users/alibiyermekov/MyProjects/dslr-v0.2/datasets/dataset_train.csv")
    histo(df)


if __name__ == "__main__":
    main()
