import numpy as np
import matplotlib.pyplot as plt
from core.operations import validate, filter_numeric_values, sorting_algorithm

def improved_correlation_matrix(indexed_columns):
        col_names = list(indexed_columns.keys())
        result = {col1: {col2: 0.0 for col2 in col_names} for col1 in col_names}
        for col in col_names:
            result[col][col] = 1.0

        for i, col1 in enumerate(col_names):
            for j in range(i+1, len(col_names)):
                col2 = col_names[j]

                # Find common indices between these two columns
                common_indices = set(indexed_columns[col1].keys()) & set(indexed_columns[col2].keys())
                
                if len(common_indices) > 1:
                    x = [indexed_columns[col1][idx] for idx in common_indices]
                    y = [indexed_columns[col2][idx] for idx in common_indices]
                    
                    try:
                        corr = np.corrcoef(x, y)[0, 1]
                        if not np.isnan(corr):
                            result[col1][col2] = corr
                            result[col2][col1] = corr
                    except:
                        raise ValueError("Failed to calculate corr cofficinet")
        return result

if __name__ == "__main__":
    numeric_columns = {}
    indexed_columns = {}
    
    dataset = validate("/Users/alibiyermekov/MyProjects/dslr-v0.2/datasets/dataset_train.csv")

    for col_name, col in dataset.items():
        if col_name != 'Index':
            test_values = filter_numeric_values(col)
            if len(test_values) > 0:
                indexed_columns[col_name] = filter_numeric_values(col, remove_nan=True, preserve_indices=True)
                print(f"Column '{col_name}' has {len(indexed_columns[col_name])} numeric values")

    corr_matrix = improved_correlation_matrix(indexed_columns)

    print(corr_matrix)    

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
        
        plt.scatter(x, y, alpha=0.5)
        plt.title(f"Scatter Plot: {max_pair[0]} vs {max_pair[1]}\nPearson Correlation: {correlation:.6f}")
        plt.xlabel(max_pair[0])
        plt.ylabel(max_pair[1])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid correlation pairs found")

