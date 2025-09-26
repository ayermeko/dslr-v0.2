from core.operations import validate
from core.visualizations import pair_plot

if __name__ == "__main__":
    try:
        dataset = validate("datasets/dataset_train.csv")
        hi = ['Defense Against the Dark Arts', 
              'Charms',
              'Herbology',
              'Divination',
              'Muggle Studies']
        pair_plot(dataset, features=hi)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")