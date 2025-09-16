import matplotlib.pyplot as plt
import numpy as np
from .operations import is_numeric_valid, min_max, correlation_matrix, filter_numeric_values
from enum import Enum


class Colors(Enum):
    DARK = "#000000"
    RED = "#570101"
    YELLOW = "#FFFF0000"
    BLUE = "#34B5BC8A"
    GREEN = "#4FC714FF"


def histo(df, subject="Care of Magical Creatures", bins=100):
    """
    Create a histogram of scores for a subject, separated by house
    """
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    colors = {
        "Gryffindor": Colors.RED.value,
        "Hufflepuff": Colors.YELLOW.value,
        "Ravenclaw": Colors.BLUE.value,
        "Slytherin": Colors.GREEN.value
    }

    if subject not in df.columns:
        raise ValueError(f"DataFrame does not contain column '{subject}'")
    
    house_scores = {house: [] for house in houses}
    for i in range(df.shape[0]):
        house = df["Hogwarts House"][i]
        score = df[subject][i]
        
        if isinstance(house, str) and is_numeric_valid(score):
            house_scores[house].append(score)

    if all(len(scores) == 0 for scores in house_scores.values()):
        print(f"No valid data found for subject '{subject}'")
        return
    
    plt.figure(figsize=(10, 6))
    all_scores = []
    for scores in house_scores.values():
        all_scores.extend(scores)
    
    if all_scores:
        min_score = min_max(all_scores, find="min")
        max_score = min_max(all_scores, find="max")
        bin_edges = np.linspace(min_score, max_score, bins+1)
        
        for house in houses:
            if house_scores[house]:
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
    plt.show()


def scatterplot(dataset) -> None:
    indexed_columns = {}
    for col_name, col in dataset.items():
        if col_name != 'Index':
            test_values = filter_numeric_values(col)
            if len(test_values) > 0:
                indexed_columns[col_name] = filter_numeric_values(col, remove_nan=True, preserve_indices=True)
                print(f"Column '{col_name}' has {len(indexed_columns[col_name])} numeric values")

    corr_matrix = correlation_matrix(indexed_columns)


    # Find the highest correlation
    max_corr = 0
    max_pair = None
    
    for col1 in corr_matrix:
        for col2 in corr_matrix:
            if col1 != col2:  # Skip self-correlations
                curr_corr = abs(corr_matrix[col1][col2])
                if curr_corr > max_corr:
                    max_corr = curr_corr
                    max_pair = (col1, col2)
    
    if max_pair:
        try:
            correlation = corr_matrix[max_pair[0]][max_pair[1]]
            print(f"Most correlated features: {max_pair[0]} and {max_pair[1]}")
            print(f"Correlation coefficient: {correlation}")
            
            # Create scatter plot
            plt.figure(figsize=(10, 6))
            
            # Get common indices for these two columns
            common_indices = set(indexed_columns[max_pair[0]].keys()) & set(indexed_columns[max_pair[1]].keys())
            
            # Extract values
            x = [indexed_columns[max_pair[0]][idx] for idx in sorted(common_indices)]
            y = [indexed_columns[max_pair[1]][idx] for idx in sorted(common_indices)]
            
            plt.scatter(x, y, alpha=0.5, color=Colors.RED.value)
            plt.title(f"Scatter Plot: {max_pair[0]} vs {max_pair[1]}\nPearson Correlation: {correlation:.6f}")
            plt.xlabel(max_pair[0])
            plt.ylabel(max_pair[1])
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            print("Program was Interrupted!")
    else:
        print("No valid correlation pairs found")

