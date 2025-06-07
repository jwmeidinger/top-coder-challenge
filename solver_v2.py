import json
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
import joblib

# Re-using the class structure from solution.py for consistency
class ReimbursementCalculator:
    def __init__(self, rules, model_path='reimbursement_model_8_to_14.joblib'):
        self.long_trip_model = joblib.load(model_path)
        self.rules = rules

    def calculate(self, trip_duration_days, miles_traveled, total_receipts_amount):
        # This function will be called repeatedly by the solver
        # with different rule sets.
        day = int(trip_duration_days)
        miles = int(miles_traveled)
        receipts = float(total_receipts_amount)

        if day not in self.rules:
            return 0 # Should not happen in this script

        rule = self.rules[day]
        
        is_outlier = False
        if day == 1 and receipts > 1900: is_outlier = True
        elif day == 2 and receipts > 1950: is_outlier = True
        elif day == 3 and receipts > 2100: is_outlier = True
        elif day == 4 and receipts > 2100: is_outlier = True
        elif day == 5 and receipts > 2200: is_outlier = True
        elif day == 6 and receipts > 2300: is_outlier = True
        elif day == 7 and receipts > 2400: is_outlier = True

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

    current_rules = {
        1: {'per_diem': 40, 'mileage_rate1': 0.5, 'mileage_threshold': 75, 'mileage_rate2': 0.3, 'receipt_rate': 0.75, 'receipt_cap': 1100, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0, 'outlier_receipt_rate': 0.45},
        2: {'per_diem': 100, 'mileage_rate1': 0.52, 'mileage_threshold': 75, 'mileage_rate2': 0.35, 'receipt_rate': 0.8, 'receipt_cap': 1100, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0, 'outlier_receipt_rate': 0.45},
        3: {'per_diem': 240, 'mileage_rate1': 0.7, 'mileage_threshold': 100, 'mileage_rate2': 0.25, 'receipt_rate': 0.75, 'receipt_cap': 1000, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0, 'outlier_receipt_rate': 0.45},
        4: {'per_diem': 260, 'mileage_rate1': 0.7, 'mileage_threshold': 125, 'mileage_rate2': 0.25, 'receipt_rate': 0.75, 'receipt_cap': 1100, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0, 'outlier_receipt_rate': 0.45},
        5: {'per_diem': 310, 'mileage_rate1': 0.5, 'mileage_threshold': 75, 'mileage_rate2': 0.45, 'receipt_rate': 0.8, 'receipt_cap': 1000, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0, 'outlier_receipt_rate': 0.45},
        6: {'per_diem': 380, 'mileage_rate1': 0.7, 'mileage_threshold': 125, 'mileage_rate2': 0.45, 'receipt_rate': 0.75, 'receipt_cap': 1000, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0, 'outlier_receipt_rate': 0.45},
        7: {'per_diem': 490, 'mileage_rate1': 0.64, 'mileage_threshold': 125, 'mileage_rate2': 0.5, 'receipt_rate': 0.8, 'receipt_cap': 900, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0, 'outlier_receipt_rate': 0.45},
    }
    
    def is_outlier(row):
        day = row['trip_duration_days']
        receipts = row['total_receipts_amount']
        if day == 1 and receipts > 1900: return True
        if day == 2 and receipts > 1950: return True
        if day == 3 and receipts > 2100: return True
        if day == 4 and receipts > 2100: return True
        if day == 5 and receipts > 2200: return True
        if day == 6 and receipts > 2300: return True
        if day == 7 and receipts > 2400: return True
        return False
        
    df_full['is_outlier'] = df_full.apply(is_outlier, axis=1)
    
    calculator = ReimbursementCalculator(rules=current_rules)
    df_full['actual_output'] = df_full.apply(lambda row: calculator.calculate(row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']), axis=1)
    df_full['error'] = df_full['actual_output'] - df_full['expected_output']

    final_outlier_rates = {}

    for day in range(1, 8):
        failing_outliers_day = df_full[(df_full['is_outlier']) & (df_full['trip_duration_days'] == day) & (df_full['error'].abs() > 1)].copy()
        
        if failing_outliers_day.empty:
            print(f"\nNo failing bug cases to optimize for day {day}.")
            final_outlier_rates[day] = current_rules[day]['outlier_receipt_rate'] # Keep the old rate
            continue

        print(f"\nFound {len(failing_outliers_day)} failing 'bug' cases to optimize for day {day}.")

        best_mae = float('inf')
        best_rate = 0
        
        for rate in [x / 100 for x in range(0, 101, 1)]:
            temp_rules = current_rules.copy()
            temp_rules[day]['outlier_receipt_rate'] = rate
                
            calc = ReimbursementCalculator(rules=temp_rules)
            
            predicted = failing_outliers_day.apply(lambda row: calc.calculate(
                row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']
            ), axis=1)
            
            mae = mean_absolute_error(failing_outliers_day['expected_output'], predicted)

            if mae < best_mae:
                best_mae = mae
                best_rate = rate
        
        final_outlier_rates[day] = best_rate
        print(f"--- Day {day} Optimization Complete ---")
        print(f"Best MAE for failing outlier cases: ${best_mae:.2f}")
        print(f"New optimal 'outlier_receipt_rate' for day {day}: {best_rate}")

    print("\n\n--- Final Optimized Outlier Rates ---")
    print(final_outlier_rates)

if __name__ == '__main__':
    run_solver() 