import json
import pandas as pd
import numpy as np

def analyze_latest_errors(results_file='results/results_20250607_153804.json'):
    """
    Analyzes the latest error report to identify patterns in the new high-error cases.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data[:-1])
    df_input = pd.json_normalize(df['input'])
    df = pd.concat([df.drop('input', axis=1), df_input], axis=1)

    for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output', 'actual_output', 'error']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['error'], inplace=True)
    df['abs_error'] = df['error'].abs()
    
    high_error_df = df.sort_values(by='abs_error', ascending=False)

    print("Top 25 High-Error Cases from Latest Run:")
    print(high_error_df.head(25)[['case_number', 'trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output', 'actual_output', 'error']])
    
    # Focus on the cases where our formula produced a negative reimbursement
    negative_output_cases = high_error_df[high_error_df['actual_output'] < 0]
    
    print("\nAnalysis of Cases Resulting in Negative Reimbursement:")
    print(f"Found {len(negative_output_cases)} cases with negative output.")
    
    if len(negative_output_cases) > 0:
        # Group by trip duration to see if there are patterns
        for day, group in negative_output_cases.groupby('trip_duration_days'):
            print(f"\n--- {day}-day Trips with Negative Output ---")
            print(group[['case_number', 'miles_traveled', 'total_receipts_amount', 'expected_output', 'actual_output']].describe())


if __name__ == '__main__':
    analyze_latest_errors() 