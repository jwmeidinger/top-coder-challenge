import json
import pandas as pd
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import itertools
import numpy as np

def analyze_data_with_new_formula():
    """
    Analyzes travel expense data for 1-7 day trips with a more complex formula,
    including tiered mileage, low-receipt penalties, and efficiency bonuses.
    """
    with open('public_cases.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame([d['input'] for d in data])
    df['expected_output'] = [d['expected_output'] for d in data]
    df = df.astype(float)

    outlier_condition = (df['total_receipts_amount'] > 500) & (df['expected_output'] < 500)
    outlier_indices = df[outlier_condition].index
    print(f"Identified {len(outlier_indices)} outlier rows to exclude from base formula training.")

    all_best_params = {}

    for day in range(1, 8):
        print("\n" + "="*80)
        print(f"Refining formula for {day}-day trips.")
        print("="*80)

        df_day = df[df['trip_duration_days'] == day]
        df_day_clean = df_day.drop(outlier_indices, errors='ignore')
        
        if len(df_day_clean) < 5:
            print(f"Not enough data for {day}-day trips.")
            continue
            
        print(f"Analyzing {len(df_day_clean)} of {len(df_day)} {day}-day trips.")

        per_diems = range(int(day * 40), int(day * 120), 10)
        mileage_rate1s = [x / 100 for x in range(50, 71, 2)]
        mileage_thresholds = [75, 100, 125]
        mileage_rate2s = [x / 100 for x in range(25, 51, 5)]
        receipt_rates = [0.65, 0.7, 0.75, 0.8]
        receipt_caps = range(900, 1501, 100)
        low_receipt_thresholds = [10, 20, 30]
        low_receipt_penalties = [0, 25, 50, 75]
        
        param_grid = list(itertools.product(
            per_diems, mileage_rate1s, mileage_thresholds, mileage_rate2s,
            receipt_rates, receipt_caps, low_receipt_thresholds, low_receipt_penalties
        ))
        
        print(f"Optimizing over {len(param_grid)} combinations...")

        def find_best_params(params):
            p_diem, m_rate1, m_thresh, m_rate2, r_rate, r_cap, lr_thresh, lr_penalty = params
            if m_rate2 >= m_rate1: return -1, params

            def formula(row):
                miles = row['miles_traveled']
                receipts = row['total_receipts_amount']
                
                if miles > m_thresh:
                    mileage_reimbursement = (m_thresh * m_rate1) + ((miles - m_thresh) * m_rate2)
                else:
                    mileage_reimbursement = miles * m_rate1
                
                receipt_reimbursement = min(receipts * r_rate, r_cap)

                reimbursement = p_diem + mileage_reimbursement + receipt_reimbursement
                
                if 0 < receipts < lr_thresh:
                    reimbursement -= lr_penalty
                    
                return reimbursement

            test_output = df_day_clean.apply(formula, axis=1)
            r2 = r2_score(df_day_clean['expected_output'], test_output)
            return r2, params

        results = Parallel(n_jobs=-1, verbose=1)(delayed(find_best_params)(p) for p in param_grid)
        
        best_r2, best_params_tuple = max(results, key=lambda item: item[0])
        
        all_best_params[day] = {
            'per_diem': best_params_tuple[0], 'mileage_rate1': best_params_tuple[1],
            'mileage_threshold': best_params_tuple[2], 'mileage_rate2': best_params_tuple[3],
            'receipt_rate': best_params_tuple[4], 'receipt_cap': best_params_tuple[5],
            'low_receipt_threshold': best_params_tuple[6], 'low_receipt_penalty': best_params_tuple[7],
            'r2_score': best_r2
        }

        print(f"\nBest R2 score on cleaned {day}-day data: {best_r2:.6f}")
        print(f"Best parameters found: {all_best_params[day]}")
        
    print("\n\n" + "="*80)
    print("Final Parameters Found:")
    print("="*80)
    for day, params in all_best_params.items():
        print(f"Day {day}: {params}")

if __name__ == '__main__':
    analyze_data_with_new_formula() 