from core.operations import validate
from core.visualizations import scatterplot

if __name__ == "__main__":
    try:
        dataset = validate("/Users/alibiyermekov/MyProjects/dslr-v0.2/datasets/dataset_train.csv")
        scatterplot(dataset)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
