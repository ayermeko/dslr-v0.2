from core.operations import validate, sortout_col
from core.visualizations import pair_plot

if __name__ == "__main__":
    try:
        dataset = validate("datasets/dataset_train.csv")
        hi = [
                'Defense Against the Dark Arts',
                'Charms',
                'Herbology',
                'Divination',
                'Muggle Studies'
        ]
        # _, hi = sortout_col(dataset)
        # truing = []
        # for cal_name in hi:
        #     if cal_name == 'Ancient Runes' or cal_name == 'Defense Against the Dark Arts':
        #         truing.append(cal_name)
        #     continue
        # print(truing)
        print(hi)
        pair_plot(dataset, features=hi)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")