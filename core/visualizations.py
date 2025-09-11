import matplotlib.pyplot as plt
import numpy as np
from .operations import filter_numeric_values, min_max
from enum import Enum

class Colors(Enum):
    DARK = "#000000"
    RED = "#570101"
    YELLOW = "#FFFF0000"
    BLUE = "#34B5BC8A"
    GREEN = "#4FC714FF"


def histo(df, subject="Care of Magical Creatures", bins=30):
    """
    Create a histogram of scores for a subject, separated by house
    
    Args:
        df: DataFrame containing the data
        subject: Subject to plot (column name)
        bins: Number of bins for the histogram
    """
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    colors = {
        "Gryffindor": Colors.RED.value,
        "Hufflepuff": Colors.YELLOW.value,
        "Ravenclaw": Colors.BLUE.value,
        "Slytherin": Colors.GREEN.value
    }

    # Validate column names
    if "Hogwarts House" not in df.columns:
        raise ValueError("DataFrame must contain 'Hogwarts House' column")
    if subject not in df.columns:
        raise ValueError(f"DataFrame does not contain column '{subject}'")
    
    # Collect scores by house
    house_scores = {house: [] for house in houses}
    for i in range(df.shape[0]):
        house = df["Hogwarts House"][i]
        score = df[subject][i]
        
        # Check if both house and score are valid
        if isinstance(house, str) and house in houses and isinstance(score, (int, float)) and not np.isnan(score):
            house_scores[house].append(score)
    
    # Check if we have data to plot
    if all(len(scores) == 0 for scores in house_scores.values()):
        print(f"No valid data found for subject '{subject}'")
        return
    
    # Plot histograms for all houses together
    plt.figure(figsize=(10, 6))
    
    # Determine common x-axis range for better comparison
    all_scores = []
    for scores in house_scores.values():
        all_scores.extend(scores)
    
    if all_scores:
        # Create common bins for all histograms
        min_score = min_max(all_scores, find="min")
        max_score = min_max(all_scores, find="max")
        bin_edges = np.linspace(min_score, max_score, bins+1)
        
        for house in houses:
            if house_scores[house]:  # Only plot if there are scores
                plt.hist(
                    house_scores[house],
                    bins=bin_edges,
                    alpha=0.6,
                    label=f"{house} (n={len(house_scores[house])})",
                    color=colors[house],
                    edgecolor=Colors.DARK.value
                )
    
    plt.title(f"Distribution of {subject} Scores by House")
    plt.xlabel("Scores")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save to file and show
    plt.savefig('histogram.png')
    print(f"Histogram saved to 'histogram.png'")
    plt.show()
