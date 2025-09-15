from core.operations import validate
from core.visualizations import scatterplot

if __name__ == "__main__":
    df = validate("datasets/dataset_train.csv")

    scatterplot(df)