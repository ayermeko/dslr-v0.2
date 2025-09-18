import matplotlib.pyplot as plt
import numpy as np
from .operations import is_numeric_valid, min_max, correlation_matrix, filter_numeric_values


def histo(df, subject="Care of Magical Creatures", freq="Best Hand", bins=100):
    """
    Create a histogram of scores for a subject, separated by house
    """

    if freq not in df.columns:
        raise ValueError(f"DataFrame does not have '{freq}'")
    if not all(isinstance(value, str) for value in df[freq]):
        raise TypeError("Non-string value for catigories or NaN")
    if subject not in df.columns:
        raise ValueError(f"DataFrame does not contain column '{subject}'")
    
    cats = list(set(df[freq]))
    
    col_scores = {cat: [] for cat in cats}
    for i in range(df.shape[0]): # 0 defines a length of row
        freq_cat = df[freq][i]
        score = df[subject][i]
        
        if isinstance(freq_cat, str) and is_numeric_valid(score):
            col_scores[freq_cat].append(score) # adding a score by catigories
    
    plt.figure(figsize=(10, 6))
    all_scores = filter_numeric_values(df[subject], remove_nan=True)

    try:
        min_score = min_max(all_scores, find="min")
        max_score = min_max(all_scores, find="max")
        bin_edges = np.linspace(min_score, max_score, bins+1)
        
        for cat in cats:
            if col_scores[cat]:
                plt.hist(
                    col_scores[cat],
                    bins=bin_edges,
                    alpha=0.7,
                    label=f"{cat} (n={len(col_scores[cat])})",
                    edgecolor="black"
                )
        
        plt.title(f"Distribution of {subject} Scores")
        plt.xlabel("Scores (Hight)")
        plt.ylabel("Number of Students (Frequency)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("Display was Interrupted!")


def scatterplot(dataset) -> None:
    indexed_columns = {}
    for col_name, col in dataset.items():
        if col_name == 'Index':
            continue
        threshold = filter_numeric_values(col)
        if len(threshold) > 0:
            indexed_columns[col_name] = filter_numeric_values(col, remove_nan=True, preserve_indices=True)
            print(f"'{col_name}' has {len(indexed_columns[col_name])} numeric values")

    corr_matrix = correlation_matrix(indexed_columns)

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
            
            plt.figure(figsize=(10, 6))
            
            common_indices = set(indexed_columns[max_pair[0]].keys()) & set(indexed_columns[max_pair[1]].keys())
            
            # Extract values
            x = [indexed_columns[max_pair[0]][idx] for idx in sorted(common_indices)]
            y = [indexed_columns[max_pair[1]][idx] for idx in sorted(common_indices)]
            
            plt.scatter(x, y, alpha=0.5, color="gray")
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

def pair_plot(dataset):
    pass