import json
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

def analyze_and_plot_errors(results_file='results/results_20250607_150413.json', error_threshold=100):
    """
    Analyzes and prints cases with high errors from a results file.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    # The last item is a summary, so we exclude it.
    df = pd.DataFrame(data[:-1])

    # The input values are nested in a dictionary. Let's flatten it.
    df_input = pd.json_normalize(df['input'])
    df = pd.concat([df.drop('input', axis=1), df_input], axis=1)

    # Convert columns to numeric types, coercing errors
    for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output', 'actual_output', 'error']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where conversion failed (e.g., the summary row)
    df.dropna(subset=['error'], inplace=True)
    
    # Filter for bad results
    bad_results = df[df['error'].abs() > error_threshold].copy()
    print(f"Found {len(bad_results)} cases with absolute error > ${error_threshold}")
    print("Descriptive statistics for high-error cases:")
    print(bad_results[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'error']].describe())

    if len(bad_results) == 0:
        print("No high-error cases to report.")
        return
    
    print("\nTop 50 high-error cases (sorted by absolute error):")
    # Use abs() for sorting to see largest discrepancies, positive or negative
    bad_results['abs_error'] = bad_results['error'].abs()
    print(bad_results.sort_values(by='abs_error', ascending=False).head(50)[['case_number', 'trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output', 'actual_output', 'error']])


if __name__ == '__main__':
    analyze_and_plot_errors() 