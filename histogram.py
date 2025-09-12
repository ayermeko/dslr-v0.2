from core.visualizations import histo
from core.operations import validate

def main():
    try:
        df = validate("datasets/dataset_train.csv")
        
        # Show counts of each house
        houses = {}
        for i in range(df.shape[0]): # 0 to get the length of rows
            house = df["Hogwarts House"][i]
            if isinstance(house, str):
                houses[house] = houses.get(house, 0) + 1
        
        print(houses)
        
        # Generate the histogram
        histo(df)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()