import matplotlib.pyplot as plt
from .operations import filter_numeric_values

def histo(df):
    """
    Plot histograms for all numeric columns in the custom DataFrame
    """
    for col_name, col in df.items():
        # Filter numeric values (ignore NaNs)
        values = filter_numeric_values(col)
        if not values:
            continue  # Skip non-numeric or empty columns

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=20, edgecolor="black")
        plt.title(f"Histogram of {col_name}")
        plt.xlabel(col_name)
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.5)
        plt.show()