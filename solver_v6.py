import json
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
import joblib

# Re-using the class structure from solution.py for consistency
class ReimbursementCalculator:
    def __init__(self, rules, params):
        self.long_trip_model = joblib.load('reimbursement_model_8_to_14.joblib')
        self.rules = rules
        self.params = params

    def calculate(self, trip_duration_days, miles_traveled, total_receipts_amount):
        # This function will be called repeatedly by the solver
        # with different rule sets.
        day = int(trip_duration_days)
        miles = int(miles_traveled)
        receipts = float(total_receipts_amount)

        if day not in self.rules:
            return 0 # Should not happen in this script

        rule = self.rules[day]
        day_params = self.params.get(day, {})
        
        anomaly_threshold = day_params.get('anomaly_threshold', float('inf'))
        bug_threshold = day_params.get('bug_threshold', float('inf'))
        
        is_bug = receipts > bug_threshold
        is_anomaly = not is_bug and receipts > anomaly_threshold

        if miles > rule['mileage_threshold']:
            mileage_reimbursement = (rule['mileage_threshold'] * rule['mileage_rate1']) + ((miles - rule['mileage_threshold']) * rule['mileage_rate2'])
        else:
            mileage_reimbursement = miles * rule['mileage_rate1']

        if is_bug:
            receipt_reimbursement = receipts * day_params.get('bug_rate', 0)
        elif is_anomaly:
            receipt_reimbursement = receipts * day_params.get('anomaly_rate', 0)
        else:
            receipt_reimbursement = min(receipts * rule['receipt_rate'], rule['receipt_cap'])
            
        reimbursement = rule['per_diem'] + mileage_reimbursement + receipt_reimbursement
        
        if not is_bug and not is_anomaly and 0 < receipts < rule['low_receipt_threshold']:
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

    final_params = {}
    
    # Define a fixed threshold for "bug" cases
    BUG_THRESHOLD = 2500
    BUG_RATE = 0.005 # A near-zero rate for bug cases

    for day in range(1, 8):
        df_day = df_full[df_full['trip_duration_days'] == day].copy()
        if df_day.empty:
            continue
            
        print(f"\n--- Optimizing for Day {day} ---")

        # Separate the bug cases first
        df_bugs = df_day[df_day['total_receipts_amount'] > BUG_THRESHOLD]
        df_non_bugs = df_day[df_day['total_receipts_amount'] <= BUG_THRESHOLD]

        best_day_mae = float('inf')
        best_day_params = {
            'bug_threshold': BUG_THRESHOLD,
            'bug_rate': BUG_RATE,
            'anomaly_threshold': -1, # Using -1 to signify no anomaly found yet
            'anomaly_rate': -1
        }

        anomaly_threshold_search = np.arange(1000, BUG_THRESHOLD, 200)
        anomaly_rate_search = [x / 100 for x in range(30, 76, 2)]

        for anomaly_thresh in anomaly_threshold_search:
            # Anomalies are cases above the current threshold but below the bug threshold
            df_anomalies = df_non_bugs[df_non_bugs['total_receipts_amount'] > anomaly_thresh]
            
            if df_anomalies.empty:
                continue

            best_anomaly_rate = 0
            best_anomaly_mae = float('inf')

            for rate in anomaly_rate_search:
                params = {day: {'anomaly_threshold': anomaly_thresh, 'anomaly_rate': rate}}
                calc = ReimbursementCalculator(rules=current_rules, params=params)
                predicted = df_anomalies.apply(lambda r: calc.calculate(r.trip_duration_days, r.miles_traveled, r.total_receipts_amount), axis=1)
                mae = mean_absolute_error(df_anomalies['expected_output'], predicted)
                if mae < best_anomaly_mae:
                    best_anomaly_mae = mae
                    best_anomaly_rate = rate
            
            # Now, calculate the total MAE for the day with this set of parameters
            # for anomalies and the fixed parameters for bugs
            temp_params = {
                day: {
                    'anomaly_threshold': anomaly_thresh,
                    'anomaly_rate': best_anomaly_rate,
                    'bug_threshold': BUG_THRESHOLD,
                    'bug_rate': BUG_RATE
                }
            }
            
            calc = ReimbursementCalculator(rules=current_rules, params=temp_params)
            predicted = df_day.apply(lambda r: calc.calculate(r.trip_duration_days, r.miles_traveled, r.total_receipts_amount), axis=1)
            total_mae = mean_absolute_error(df_day['expected_output'], predicted)

            if total_mae < best_day_mae:
                best_day_mae = total_mae
                best_day_params.update({
                    'anomaly_threshold': anomaly_thresh,
                    'anomaly_rate': best_anomaly_rate,
                })
                print(f"  New best MAE for day {day}: ${total_mae:.2f} with params: {best_day_params}")
        
        final_params[day] = best_day_params

    print("\n\n--- Final Optimized Parameters ---")
    print(final_params)

if __name__ == '__main__':
    run_solver() 