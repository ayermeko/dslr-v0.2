import numpy as np
import csv
import os
from dataclasses import dataclass, field
from typing import Any, Optional, List
from prettytable import PrettyTable

@dataclass
class DataFrame:
    """A simple DataFrame-like class without pandas dependency"""
    
    data: Any
    columns: Optional[List[str]] = field(default_factory=list)

    def __init__(self, data, columns=None):
        self.data = np.array(data, dtype=object)
        self.columns = columns

        
    def __getitem__(self, key):
        """Allow column access with df['column_name']"""
        if isinstance(key, str):
            if self.columns is None:
                raise ValueError("DataFrame has no column names")
            col_idx = self.columns.index(key)
            return self.data[:, col_idx]
        return self.data[:, key]
    
    def items(self):
        """Return an iterator of (column_name, column_data) pairs"""
        if self.columns is None:
            raise ValueError("DataFrame has no column names")
        for i, col_name in enumerate(self.columns):
            yield (col_name, self.data[:, i])
    
    @property
    def shape(self):
        """Return the shape of the data"""
        return self.data.shape


def read_csv(filename: str):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        
        dataset = []
        for row in reader:
            processed_row = []
            for value in row:
                if value.strip() == '':
                    # Handle empty cells
                    processed_row.append(np.nan)
                else:
                    try:
                        # Try to convert to float for numeric columns
                        processed_value = float(value)
                        processed_row.append(processed_value)
                    except ValueError:
                        # For non-numeric data (like house names), keep as string
                        processed_row.append(value)
            dataset.append(processed_row)
    return DataFrame(dataset, header)

def validate(filename: str):
    """
    This function is to validate the file, and data
    converting a raw data into a df
    @param filename || path
    @returns DataFrame
    """
    if not filename.endswith(".csv"):
        raise ValueError("Incorrect file extention.")
    if not os.path.exists(filename):
        raise FileExistsError("Given filename does not exists.")
    if not os.path.isfile(filename):
        raise FileNotFoundError("Given argument is not a file.")
    if not os.access(filename, os.R_OK):
        raise PermissionError("File is not readable.")
    
    return read_csv(filename)


def mean(values):
    """Calculate mean of values"""
    return sum(values) / len(values)

def std(values):
    """Calculate standard deviation"""
    n = len(values)
    if n < 2:
        raise ValueError("Sample size lower than 2")
    var = sum((x - mean(values)) ** 2 for x in values) / (n - 1)
    return var ** 0.5

def sorting_algorithm(values):
    """Simple bubble sort algorithm"""
    values = values.copy()  # Make a copy to avoid modifying the original
    for i in range(len(values) - 1):
        swapped = False
        for j in range(len(values) - i - 1):
            if values[j + 1] < values[j]:
                swapped = True
                values[j + 1], values[j] = values[j], values[j + 1]
        if not swapped:
            break
    return values

def min_max(values, find):
    """Find minimum or maximum value"""
    sorted_values = sorting_algorithm(values)
    if find == 'min':
        return sorted_values[0]
    else:
        return sorted_values[-1]

def percentile(values, p):
    """Calculate percentile"""
    sorted_values = sorting_algorithm(values)
    n = len(sorted_values)
    if n == 0:
        return 0
        
    idx = (p / 100) * (n - 1)
    
    if idx.is_integer():
        return sorted_values[int(idx)]
        
    lower_idx = int(idx)
    upper_idx = lower_idx + 1
    lower_val = sorted_values[lower_idx]
    upper_val = sorted_values[upper_idx]
    
    fraction = idx - lower_idx
    return lower_val + fraction * (upper_val - lower_val)


def adjust_col_names(original: List[str]) -> List[str]:
    result = []
    for col_name in original:
        adjusted_col = col_name[:10] + "." if len(col_name) > 10 else col_name
        result.append(adjusted_col)
    return result


def format_results(results):
    """Format results as a table using prettytable"""
    if not results:
        return "No numerical features found."

    original_col_names = list(results.keys())
    
    adjusted_col_names = adjust_col_names(original_col_names)
    name_mapping = dict(zip(adjusted_col_names, original_col_names))

    table = PrettyTable()
    table.field_names = [""] + adjusted_col_names
    
    for stat in results[adjusted_col_names[1]].keys():
        row = [stat]
        for adjusted_col in adjusted_col_names:
            original_col = name_mapping[adjusted_col]
            row.append(f"{results[original_col][stat]:.6f}")
        table.add_row(row)
    
    return str(table)

def is_numeric_valid(value):
    """Check if a single value is numberic"""
    return isinstance(value, (int, float)) and not np.isnan(value)

def filter_numeric_values(columns, remove_nan=True, preserve_indices=False):
    """
    Filter a column to keep only numeric values (int, float) and optionally remove NaN values
    """
    if columns is None or len(columns) == 0:
        raise ValueError("No columns or empty columns.")
    
    if preserve_indices:
        result = {}
        for i, val in enumerate(columns):
            is_numeric = isinstance(val, (int, float))
            if is_numeric and (not remove_nan or not np.isnan(val)):
                result[i] = val
        return result
    else:
        if remove_nan:
            return [val for val in columns if is_numeric_valid(val)]
        else:
            return [val for val in columns if isinstance(val, (int, float))]

def describe(dataset):
    """Generate descriptive statistics for the dataset"""
    
    results = {}
    for col_name, col in dataset.items():
        # Filter number values, ignore NaN
        values = filter_numeric_values(col)
        # Skip in case of empty column
        if not values:
            continue
            
        results[col_name] = {
            "Count": float(len(values)),
            "Mean": mean(values),
            "Std": std(values),
            "Min": min_max(values, find="min"),
            "25%": percentile(values, 25),
            "50%": percentile(values, 50),
            "75%": percentile(values, 75),
            "Max": min_max(values, find="max")
        }
    
    formatted_output = format_results(results)
    print(formatted_output)
    return results