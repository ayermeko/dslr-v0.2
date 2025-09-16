# Original pandas-based approach (kept for reference)
# # Select numeric features (skip first column)
# numeric_features = df.iloc[:, 1:].select_dtypes(include='number')
#
# # Compute correlation matrix
# corr_matrix = numeric_features.corr()
#
# # Flatten the matrix
# corr_matrix_unstacked = corr_matrix.unstack()
#
# # Remove self-correlation
# corr_matrix_unstacked = corr_matrix_unstacked[corr_matrix_unstacked != 1]
#
# # Find the pair with the largest absolute correlation (positive or negative)
# most_similar_pair = corr_matrix_unstacked.abs().idxmax()
# correlation_value = corr_matrix_unstacked[most_similar_pair]
#
# print("Most similar features:", most_similar_pair)
# print("Correlation value:", correlation_value)

from core.operations import validate, filter_numeric_values
from core.visualizations import scatterplot

if __name__ == "__main__":
    numeric_columns = {}
    indexed_columns = {}
    
    dataset = validate("/Users/alibiyermekov/MyProjects/dslr-v0.2/datasets/dataset_train.csv")

    # Step 1: Collect numeric values with indices preserved
    for col_name, col in dataset.items():
        if col_name != 'Index':  # Skip the Index column
            indexed_columns[col_name] = filter_numeric_values(col, remove_nan=False, preserve_indices=True)
    
    # Step 2: Find common indices across all columns
    all_indices = set.intersection(*[set(col_dict.keys()) for col_dict in indexed_columns.values()])
    print(f"Found {len(all_indices)} rows with data in all columns")
    
    # Step 3: Create aligned data using only common indices
    for col_name, indices_dict in indexed_columns.items():
        numeric_columns[col_name] = [indices_dict[idx] for idx in sorted(all_indices)]
    
    # Now all columns in numeric_columns have the same length and are properly aligned
    # You can safely calculate correlations
    # scatterplot(numeric_columns)

