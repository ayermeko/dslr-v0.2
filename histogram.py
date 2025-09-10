from core.visualizations import histo
from core.operations import read_csv

def main():
    df = read_csv("dataset_train.csv")
    
    # Check the first few values to confirm house names are preserved
    print("First 5 house values:")
    for i in range(5):
        print(f"  {i}: {df['Hogwarts House'][i]}")
    
    # Show counts of each house
    houses = {}
    for i in range(df.shape[0]):
        house = df["Hogwarts House"][i]
        if isinstance(house, str):
            houses[house] = houses.get(house, 0) + 1
    
    print("\nHouse counts:")
    for house, count in houses.items():
        print(f"  {house}: {count}")
    
    # Generate the histogram
    histo(df)

if __name__ == "__main__":
    main()