from core.visualizations import histo
from core.operations import read_csv


def main():
    histo(read_csv("dataset_train.csv"))

if __name__ == "__main__":
    main()