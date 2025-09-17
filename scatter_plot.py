from core.operations import validate
from core.visualizations import scatterplot

if __name__ == "__main__":
    dataset = validate("/Users/alibiyermekov/MyProjects/dslr-v0.2/datasets/dataset_train.csv")
    scatterplot(dataset)
