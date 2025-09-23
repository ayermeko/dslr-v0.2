from core.operations import validate
from core.visualizations import pair_plot

if __name__ == "__main__":
    try:
        dataset = validate("datasets/dataset_train.csv")
        hi = ["Arithmancy",
              "Astronomy",
              "Herbology",
              "Defense Against the Dark Arts",
              "Divination",
              "Muggle Studies",
              "Ancient Runes"]
        pair_plot(dataset, features=hi)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")