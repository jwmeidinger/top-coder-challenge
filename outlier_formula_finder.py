import json
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import Parallel, delayed
import itertools
import numpy as np

def find_outlier_formula(results_file='results/results_20250607_150413.json'):
    """
    Analyzes high-error, high-receipt cases to find a penalty formula.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data[:-1])
    df_input = pd.json_normalize(df['input'])
    df = pd.concat([df.drop('input', axis=1), df_input], axis=1)

    for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output', 'actual_output', 'error']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['error'], inplace=True)
    
    # Isolate the "buggy" cases based on our previous analysis
    # High error, and an expected output that is much lower than our formula's output
    outlier_df = df[(df['error'].abs() > 150) & (df['expected_output'] < df['actual_output'])].copy()
    print(f"Found {len(outlier_df)} potential 'bug' cases to analyze.")

    if len(outlier_df) == 0:
        print("No outlier cases found with the current criteria.")
        return

    # Let's assume the "correct" part of our calculation is per_diem + mileage.
    # The 'error' is the difference between what the black box *should* have paid for receipts
    # and what it *actually* paid.
    # Our 'actual_output' is based on the v3 parameters. Let's re-calculate the non-receipt portion.
    
    # Using the parameters from Current_Cursor_thoughts.md
    rules = {
        1: {'per_diem': 40, 'mileage_rate1': 0.5, 'mileage_threshold': 75, 'mileage_rate2': 0.3},
        2: {'per_diem': 100, 'mileage_rate1': 0.52, 'mileage_threshold': 75, 'mileage_rate2': 0.35},
        3: {'per_diem': 240, 'mileage_rate1': 0.7, 'mileage_threshold': 100, 'mileage_rate2': 0.25},
        4: {'per_diem': 260, 'mileage_rate1': 0.7, 'mileage_threshold': 125, 'mileage_rate2': 0.25},
        5: {'per_diem': 310, 'mileage_rate1': 0.5, 'mileage_threshold': 75, 'mileage_rate2': 0.45},
        6: {'per_diem': 380, 'mileage_rate1': 0.7, 'mileage_threshold': 125, 'mileage_rate2': 0.45},
        7: {'per_diem': 490, 'mileage_rate1': 0.64, 'mileage_threshold': 125, 'mileage_rate2': 0.50}
    }

    def calculate_base_reimbursement(row):
        day = int(row['trip_duration_days'])
        if day not in rules: return 0
        
        rule = rules[day]
        miles = row['miles_traveled']
        
        if miles > rule['mileage_threshold']:
            mileage = (rule['mileage_threshold'] * rule['mileage_rate1']) + ((miles - rule['mileage_threshold']) * rule['mileage_rate2'])
        else:
            mileage = miles * rule['mileage_rate1']
            
        return rule['per_diem'] + mileage

    outlier_df['base_reimbursement'] = outlier_df.apply(calculate_base_reimbursement, axis=1)
    outlier_df['receipt_component_actual'] = outlier_df['expected_output'] - outlier_df['base_reimbursement']

    print("\nAnalysis of the 'receipt component' for buggy cases:")
    print(outlier_df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'base_reimbursement', 'expected_output', 'receipt_component_actual']].head(10))

    # Now, let's try to find a formula for 'receipt_component_actual'
    # Hypothesis: It's a simple percentage of the receipts, but a *different* percentage.
    
    penalty_rates = [x / 100 for x in range(-50, 51, 5)] # Testing negative (penalty) and positive rates
    
    best_mae = float('inf')
    best_rate = 0

    for rate in penalty_rates:
        outlier_df['predicted_receipt_comp'] = outlier_df['total_receipts_amount'] * rate
        mae = mean_absolute_error(outlier_df['receipt_component_actual'], outlier_df['predicted_receipt_comp'])
        if mae < best_mae:
            best_mae = mae
            best_rate = rate
            
    print(f"\nBest single penalty/reimbursement rate found for buggy receipts: {best_rate:.2f}")
    print(f"Mean Absolute Error with this rate: ${best_mae:.2f}")

if __name__ == '__main__':
    find_outlier_formula() 