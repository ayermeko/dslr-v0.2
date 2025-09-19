from core.operations import validate
from core.visualizations import pair_plot

if __name__ == "__main__":
    try:
        dataset = validate("datasets/dataset_train.csv")
        specific_features = ["Arithmancy", "Herbology", "Transfiguration", "Potions"]
        pair_plot(dataset, features=specific_features)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")