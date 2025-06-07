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
    
    for day in range(1, 8):
        df_day = df_full[df_full['trip_duration_days'] == day].copy()
        if df_day.empty:
            continue
            
        print(f"\n--- Optimizing for Day {day} ---")

        # --- Data-Driven Threshold Detection ---
        df_day_sorted = df_day.sort_values('total_receipts_amount', ascending=False)
        top_5_percent_index = int(len(df_day_sorted) * 0.05)
        
        if top_5_percent_index < 2: # Ensure we have enough data points
             if len(df_day_sorted) > 1:
                 bug_threshold = df_day_sorted.iloc[0]['total_receipts_amount']
                 anomaly_threshold = df_day_sorted.iloc[1]['total_receipts_amount']
             else: # If only one or zero points, use arbitrary high values
                 bug_threshold = 3000
                 anomaly_threshold = 2800
        else:
            top_cases = df_day_sorted.head(top_5_percent_index)
            bug_threshold = top_cases['total_receipts_amount'].max()
            anomaly_threshold = top_cases['total_receipts_amount'].quantile(0.9)

        # Ensure thresholds are not equal and have a reasonable gap
        if anomaly_threshold >= bug_threshold:
            anomaly_threshold = bug_threshold * 0.9 

        print(f"  Calculated Bug Threshold: ${bug_threshold:.2f}")
        print(f"  Calculated Anomaly Threshold: ${anomaly_threshold:.2f}")

        # --- Rate Optimization ---
        df_bugs = df_day[df_day['total_receipts_amount'] > bug_threshold]
        df_anomalies = df_day[(df_day['total_receipts_amount'] > anomaly_threshold) & (df_day['total_receipts_amount'] <= bug_threshold)]

        best_bug_rate = 0.005 # Default bug rate
        if not df_bugs.empty:
             # For bugs, we assume a near-zero rate is best
            pass # Keep the default rate

        best_anomaly_rate = 0.45 # Default anomaly rate
        if not df_anomalies.empty:
            best_mae = float('inf')
            for rate in [x/100 for x in range(30, 76, 2)]:
                params = {day: {'anomaly_rate': rate}}
                calc = ReimbursementCalculator(rules=current_rules, params=params)
                predicted = df_anomalies.apply(lambda r: calc.calculate(r.trip_duration_days, r.miles_traveled, r.total_receipts_amount), axis=1)
                mae = mean_absolute_error(df_anomalies['expected_output'], predicted)
                if mae < best_mae:
                    best_mae = mae
                    best_anomaly_rate = rate

        final_params[day] = {
            'bug_threshold': bug_threshold,
            'bug_rate': best_bug_rate,
            'anomaly_threshold': anomaly_threshold,
            'anomaly_rate': best_anomaly_rate,
        }
        print(f"  Optimized Rates: Anomaly={best_anomaly_rate}, Bug={best_bug_rate}")


    print("\n\n--- Final Optimized Parameters ---")
    print(final_params)

if __name__ == '__main__':
    run_solver() 