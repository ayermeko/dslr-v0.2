from core.visualizations import histo
from core.operations import validate

def main():
    try:
        df = validate("dataset_train.csv")
        
        # Show counts of each house
        houses = {}
        for i in range(df.shape[0]):
            house = df["Hogwarts House"][i]
            
            if isinstance(house, str):
                houses[house] = houses.get(house, 0) + 1
        
        print("\nHouse counts:")
        for house, count in houses.items():
            print(f"\t{house}: {count}")
        
        # Generate the histogram
        histo(df)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()