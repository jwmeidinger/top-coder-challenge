import json
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
import joblib

# Re-using the class structure from solution.py for consistency
class ReimbursementCalculator:
    def __init__(self, rules, outlier_thresholds):
        self.long_trip_model = joblib.load('reimbursement_model_8_to_14.joblib')
        self.rules = rules
        self.outlier_thresholds = outlier_thresholds

    def calculate(self, trip_duration_days, miles_traveled, total_receipts_amount):
        # This function will be called repeatedly by the solver
        # with different rule sets.
        day = int(trip_duration_days)
        miles = int(miles_traveled)
        receipts = float(total_receipts_amount)

        if day not in self.rules:
            return 0 # Should not happen in this script

        rule = self.rules[day]
        
        is_outlier = receipts > self.outlier_thresholds.get(day, float('inf'))

        if miles > rule['mileage_threshold']:
            mileage_reimbursement = (rule['mileage_threshold'] * rule['mileage_rate1']) + ((miles - rule['mileage_threshold']) * rule['mileage_rate2'])
        else:
            mileage_reimbursement = miles * rule['mileage_rate1']

        if is_outlier:
            receipt_reimbursement = receipts * rule.get('outlier_receipt_rate', 0.45) # Use a new outlier rate
        else:
            receipt_reimbursement = min(receipts * rule['receipt_rate'], rule['receipt_cap'])
            
        reimbursement = rule['per_diem'] + mileage_reimbursement + receipt_reimbursement
        
        if not is_outlier and 0 < receipts < rule['low_receipt_threshold']:
            reimbursement -= rule['low_receipt_penalty']
            
        return round(reimbursement, 2)


def run_solver():
    """
    Finds the optimal 'outlier_receipt_rate' for each day by focusing only 
    on the high-error outlier cases for that specific day.
    """
    with open('public_cases.json', 'r') as f:
        data = json.load(f)

    df_full = pd.json_normalize(data)
    df_full.rename(columns={
        'input.trip_duration_days': 'trip_duration_days',
        'input.miles_traveled': 'miles_traveled',
        'input.total_receipts_amount': 'total_receipts_amount'
    }, inplace=True)
    
    for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output']:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce')

    df_full.dropna(inplace=True)

    # These are the original rules, we will use them as a base
    current_rules = {
        1: {'per_diem': 40, 'mileage_rate1': 0.5, 'mileage_threshold': 75, 'mileage_rate2': 0.3, 'receipt_rate': 0.75, 'receipt_cap': 1100, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        2: {'per_diem': 100, 'mileage_rate1': 0.52, 'mileage_threshold': 75, 'mileage_rate2': 0.35, 'receipt_rate': 0.8, 'receipt_cap': 1100, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        3: {'per_diem': 240, 'mileage_rate1': 0.7, 'mileage_threshold': 100, 'mileage_rate2': 0.25, 'receipt_rate': 0.75, 'receipt_cap': 1000, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        4: {'per_diem': 260, 'mileage_rate1': 0.7, 'mileage_threshold': 125, 'mileage_rate2': 0.25, 'receipt_rate': 0.75, 'receipt_cap': 1100, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        5: {'per_diem': 310, 'mileage_rate1': 0.5, 'mileage_threshold': 75, 'mileage_rate2': 0.45, 'receipt_rate': 0.8, 'receipt_cap': 1000, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        6: {'per_diem': 380, 'mileage_rate1': 0.7, 'mileage_threshold': 125, 'mileage_rate2': 0.45, 'receipt_rate': 0.75, 'receipt_cap': 1000, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        7: {'per_diem': 490, 'mileage_rate1': 0.64, 'mileage_threshold': 125, 'mileage_rate2': 0.5, 'receipt_rate': 0.8, 'receipt_cap': 900, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
    }

    # Define the new, lower outlier thresholds based on analysis
    outlier_thresholds = {
        1: 1800,
        2: 1900,
        3: 2000,
        4: 1000,
        5: 1800,
        6: 2200,
        7: 2300,
    }
    
    def is_outlier(row):
        day = row['trip_duration_days']
        receipts = row['total_receipts_amount']
        return receipts > outlier_thresholds.get(day, float('inf'))
        
    df_full['is_outlier'] = df_full.apply(is_outlier, axis=1)
    
    # Initialize a calculator for the initial error calculation
    initial_calc_rules = {day: {**rule, 'outlier_receipt_rate': 0.45} for day, rule in current_rules.items()}
    calculator = ReimbursementCalculator(rules=initial_calc_rules, outlier_thresholds=outlier_thresholds)
    df_full['actual_output'] = df_full.apply(lambda row: calculator.calculate(row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']), axis=1)
    df_full['error'] = df_full['actual_output'] - df_full['expected_output']

    final_outlier_rates = {}

    for day in range(1, 8):
        # We now optimize over all identified outliers for the day
        outliers_day = df_full[(df_full['is_outlier']) & (df_full['trip_duration_days'] == day)].copy()
        
        if outliers_day.empty:
            print(f"\nNo bug cases to optimize for day {day}.")
            # If no outliers, rate doesn't matter, but we can set a default.
            final_outlier_rates[day] = 0.45 
            continue

        print(f"\nFound {len(outliers_day)} 'bug' cases to optimize for day {day}.")

        best_mae = float('inf')
        best_rate = 0
        
        for rate in [x / 100 for x in range(0, 101, 1)]:
            temp_rules = current_rules.copy()
            temp_rules[day]['outlier_receipt_rate'] = rate
                
            calc = ReimbursementCalculator(rules=temp_rules, outlier_thresholds=outlier_thresholds)
            
            predicted = outliers_day.apply(lambda row: calc.calculate(
                row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']
            ), axis=1)
            
            mae = mean_absolute_error(outliers_day['expected_output'], predicted)

            if mae < best_mae:
                best_mae = mae
                best_rate = rate
        
        final_outlier_rates[day] = best_rate
        print(f"--- Day {day} Optimization Complete ---")
        print(f"Best MAE for outlier cases: ${best_mae:.2f}")
        print(f"New optimal 'outlier_receipt_rate' for day {day}: {best_rate}")

    print("\n\n--- Final Optimized Outlier Rates ---")
    print(final_outlier_rates)
    print("\n--- Associated Outlier Thresholds ---")
    print(outlier_thresholds)

if __name__ == '__main__':
    run_solver() 