import pandas as pd
import joblib
import sys
import json
import itertools
from sklearn.metrics import mean_absolute_error

class ReimbursementCalculator:
    def __init__(self, rules, model_path='reimbursement_model_8_to_14.joblib'):
        self.long_trip_model = joblib.load(model_path)
        self.rules = rules

    def calculate(self, trip_duration_days, miles_traveled, total_receipts_amount):
        trip_duration_days = int(trip_duration_days)
        miles_traveled = int(miles_traveled)
        total_receipts_amount = float(total_receipts_amount)
        
        is_outlier_case = False
        if trip_duration_days == 1 and total_receipts_amount > 1900: is_outlier_case = True
        elif trip_duration_days == 2 and total_receipts_amount > 1950: is_outlier_case = True
        elif trip_duration_days == 3 and total_receipts_amount > 2100: is_outlier_case = True
        elif trip_duration_days == 4 and total_receipts_amount > 2100: is_outlier_case = True
        elif trip_duration_days == 5 and total_receipts_amount > 2200: is_outlier_case = True
        elif trip_duration_days == 6 and total_receipts_amount > 2300: is_outlier_case = True
        elif trip_duration_days == 7 and total_receipts_amount > 2400: is_outlier_case = True

        if trip_duration_days in self.rules:
            rule = self.rules[trip_duration_days]
            per_diem = rule['per_diem']

            if miles_traveled > rule['mileage_threshold']:
                mileage_reimbursement = (rule['mileage_threshold'] * rule['mileage_rate1']) + \
                                        ((miles_traveled - rule['mileage_threshold']) * rule['mileage_rate2'])
            else:
                mileage_reimbursement = miles_traveled * rule['mileage_rate1']

            if is_outlier_case:
                receipt_reimbursement = total_receipts_amount * 0.45
            else:
                receipt_reimbursement = min(total_receipts_amount * rule['receipt_rate'], rule['receipt_cap'])
            
            reimbursement = per_diem + mileage_reimbursement + receipt_reimbursement

            if not is_outlier_case and 0 < total_receipts_amount < rule['low_receipt_threshold']:
                reimbursement -= rule['low_receipt_penalty']

        elif 8 <= trip_duration_days <= 14:
            input_data = pd.DataFrame({
                'trip_duration_days': [trip_duration_days],
                'miles_traveled': [miles_traveled],
                'total_receipts_amount': [total_receipts_amount]
            })
            prediction = self.long_trip_model.predict(input_data)
            reimbursement = prediction[0]
        else:
            reimbursement = 0

        return round(reimbursement, 2)


def solve_bad_cases():
    """
    Identifies high-error cases and iterates on parameters to find a better fit.
    """
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame([d for d in data])
    df_input = pd.json_normalize(df['input'])
    df = pd.concat([df.drop(['input'], axis=1), df_input], axis=1)
    df = df.astype(float)

    # Initial rules from solution.py
    initial_rules = {
        1: {'per_diem': 40, 'mileage_rate1': 0.5, 'mileage_threshold': 75, 'mileage_rate2': 0.3, 'receipt_rate': 0.75, 'receipt_cap': 1100, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        2: {'per_diem': 100, 'mileage_rate1': 0.52, 'mileage_threshold': 75, 'mileage_rate2': 0.35, 'receipt_rate': 0.8, 'receipt_cap': 1100, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        3: {'per_diem': 240, 'mileage_rate1': 0.7, 'mileage_threshold': 100, 'mileage_rate2': 0.25, 'receipt_rate': 0.75, 'receipt_cap': 1000, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        4: {'per_diem': 260, 'mileage_rate1': 0.7, 'mileage_threshold': 125, 'mileage_rate2': 0.25, 'receipt_rate': 0.75, 'receipt_cap': 1100, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        5: {'per_diem': 310, 'mileage_rate1': 0.5, 'mileage_threshold': 75, 'mileage_rate2': 0.45, 'receipt_rate': 0.8, 'receipt_cap': 1000, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        6: {'per_diem': 380, 'mileage_rate1': 0.7, 'mileage_threshold': 125, 'mileage_rate2': 0.45, 'receipt_rate': 0.75, 'receipt_cap': 1000, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
        7: {'per_diem': 490, 'mileage_rate1': 0.64, 'mileage_threshold': 125, 'mileage_rate2': 0.5, 'receipt_rate': 0.8, 'receipt_cap': 900, 'low_receipt_threshold': 10, 'low_receipt_penalty': 0},
    }

    # Calculate initial error
    calculator = ReimbursementCalculator(rules=initial_rules)
    df['actual_output'] = df.apply(lambda row: calculator.calculate(row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']), axis=1)
    df['error'] = df['actual_output'] - df['expected_output']
    
    # Isolate bad cases (for trips 1-7 days)
    bad_cases = df[(df['error'].abs() > 1) & (df['trip_duration_days'] <= 7)].copy()
    print(f"Found {len(bad_cases)} bad cases (error > $1) for 1-7 day trips to optimize.")

    final_rules = {}

    for day in range(1, 8):
        day_cases = bad_cases[bad_cases['trip_duration_days'] == day].copy()
        if day_cases.empty:
            continue

        print(f"\n--- Optimizing for {len(day_cases)} bad cases for {day}-day trips ---")

        # Define parameter search grid
        per_diems = range(int(day * 30), int(day * 150), 10)
        mileage_rate1s = [x / 100 for x in range(30, 81, 2)]
        mileage_thresholds = [50, 75, 100, 125, 150]
        mileage_rate2s = [x / 100 for x in range(20, 61, 5)]
        receipt_rates = [x / 100 for x in range(40, 91, 5)]
        receipt_caps = range(800, 1601, 100)
        
        # Reduced grid for faster iteration
        param_grid = list(itertools.product(
            per_diems, mileage_rate1s, [100], mileage_rate2s,
            receipt_rates, receipt_caps
        ))

        best_mae = float('inf')
        best_params = None

        # This part is slow, will replace with a more efficient search or parallelize later
        for params in param_grid:
            pd, mr1, mt, mr2, rr, rc = params
            
            temp_rules = {day: {
                'per_diem': pd, 'mileage_rate1': mr1, 'mileage_threshold': mt, 
                'mileage_rate2': mr2, 'receipt_rate': rr, 'receipt_cap': rc,
                'low_receipt_threshold': 10, 'low_receipt_penalty': 0 # Keep these fixed for now
            }}
            
            calculator = ReimbursementCalculator(rules=temp_rules)
            
            predicted_output = day_cases.apply(lambda row: calculator.calculate(
                row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']
            ), axis=1)
            
            mae = mean_absolute_error(day_cases['expected_output'], predicted_output)

            if mae < best_mae:
                best_mae = mae
                best_params = temp_rules[day]
        
        final_rules[day] = best_params
        print(f"Best MAE for day {day}: ${best_mae:.2f}")
        print(f"Optimal parameters for day {day}: {best_params}")

    print("\n\n--- Final Optimized Rules ---")
    for day, params in final_rules.items():
        print(f"Day {day}: {params}")


if __name__ == '__main__':
    solve_bad_cases() 