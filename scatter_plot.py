# # Select numeric features (skip first column)
# numeric_features = df.iloc[:, 1:].select_dtypes(include='number')

# # Compute correlation matrix
# corr_matrix = numeric_features.corr()

# # Flatten the matrix
# corr_matrix_unstacked = corr_matrix.unstack()

# # Remove self-correlation
# corr_matrix_unstacked = corr_matrix_unstacked[corr_matrix_unstacked != 1]

# # Find the pair with the largest absolute correlation (positive or negative)
# most_similar_pair = corr_matrix_unstacked.abs().idxmax()
# correlation_value = corr_matrix_unstacked[most_similar_pair]

# print("Most similar features:", most_similar_pair)
# print("Correlation value:", correlation_value)

from core.operations import validate, filter_numeric_values
from core.visualizations import scatterplot

if __name__ == "__main__":
    numeric_col = {}
    df = validate("/Users/alibiyermekov/MyProjects/dslr-v0.2/datasets/dataset_train.csv")
    for col_name, col in df.items():
        values = filter_numeric_values(col)
        if values:
            numeric_col[col_name] = values
    
    if 'Index' in numeric_col:
        del numeric_col['Index']

    scatterplot(numeric_col)