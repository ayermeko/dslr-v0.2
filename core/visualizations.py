import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from .operations import (
    is_numeric_valid, 
    min_max, 
    corr, 
    filter, 
    get_keys,
    extract,
    sortout_col
)


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
    all_scores = filter(df[subject], remove_nan=True)

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
    _, numerical_keys = sortout_col(dataset)
    
    idxed_cols = {}
    for col_name in numerical_keys:
        idxed_cols[col_name] = filter(dataset[col_name], preserve_indices=True)
        print(f"'{col_name}' has {len(idxed_cols[col_name])} numeric values")

    corr_matrix = corr(idxed_cols)

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
            
            common_indices = get_keys(idxed_cols, max_pair[0], max_pair[1])
            # Extract values
            x = extract(idxed_cols[max_pair[0]], common_indices)
            y = extract(idxed_cols[max_pair[1]], common_indices)
            
            plt.scatter(x, y, alpha=0.5, color="gray")
            plt.title(f"Scatter Plot: {max_pair[0]} vs {max_pair[1]}")
            plt.xlabel(max_pair[0])
            plt.ylabel(max_pair[1])
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            print("Program was Interrupted!")
    else:
        print("No valid correlation pairs found")


def pair_plot(dataset, features=None, target='Hogwarts House') -> None:
    cats, all_numerical_keys = sortout_col(dataset)

    df_dict = {}
    for col_name, col_data in dataset.items():
        df_dict[col_name] = col_data

    df = pd.DataFrame(df_dict)
    
    if features is not None:
        numerical_cols = [f for f in features if f in all_numerical_keys]
        if len(numerical_cols) != len(features):
            raise ValueError("Missing features.")
    else:
        numerical_cols = all_numerical_keys
    
    if len(numerical_cols) < 2:
        print(f"Not enough numerical columns for pair plot.")
        return
    if target not in cats:
        print(f"'{target}' is not in Dataset.")
        return

    plot_cols = numerical_cols + [target]
    df_subset = df[plot_cols].copy()

    df_subset = df_subset.dropna(subset=[target])
    df_subset = df_subset.dropna(subset=numerical_cols, how='all')
    
    if df_subset.empty:
        print("No valid data remaining after filtering")
        return
    
    try:
        graph = sns.pairplot(
            df_subset, 
            hue=target,
            diag_kind='hist',
        )
        graph.figure.set_size_inches(10, 6)
        plt.show()
    except KeyboardInterrupt:
        print("Pair plot display was interrupted!")
    