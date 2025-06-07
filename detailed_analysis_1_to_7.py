import json
import pandas as pd
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import itertools

def analyze_data_with_plots_part1():
    """
    Analyzes the travel expense data for trips lasting 1 to 7 days.
    """
    with open('public_cases.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame([d['input'] for d in data])
    df['expected_output'] = [d['expected_output'] for d in data]

    # --- Outlier/Bug Identification ---
    outlier_condition = (df['total_receipts_amount'] > 500) & (df['expected_output'] < 500)
    outliers = df[outlier_condition]
    outlier_indices = outliers.index
    print(f"Identified {len(outliers)} outlier rows to exclude from training.")

    # --- Analysis Loop for Days 1-7 ---
    for day in range(1, 8):
        print("\n" + "="*80)
        print(f"Refining formula for {day}-day trips (excluding identified outliers).")
        print("="*80)

        df_day = df[df['trip_duration_days'] == day]
        df_day_clean = df_day.drop(outlier_indices, errors='ignore')
        print(f"Analyzing {len(df_day_clean)} of {len(df_day)} {day}-day trips.")

        if len(df_day_clean) < 5:
            print("Not enough data to find a reliable formula.")
            continue

        # Define parameter ranges
        per_diems = range(50 * (day-1), 100 * day + 1, 10)
        mileage_rates = [x / 100 for x in range(30, 71, 5)]
        receipt_rates = [x / 100 for x in range(30, 71, 5)]
        receipt_caps = range(800, 1501, 50)
        
        param_grid = list(itertools.product(per_diems, mileage_rates, receipt_rates, receipt_caps))
        print(f"Optimizing over {len(param_grid)} combinations for {day}-day trips...")

        def find_best_params(params):
            from sklearn.metrics import r2_score
            per_diem, m_rate, r_rate, r_cap = params
            
            def formula(miles, receipts):
                mileage = miles * m_rate
                receipt_reimbursement = min(receipts * r_rate, r_cap)
                return per_diem + mileage + receipt_reimbursement

            test_output = df_day_clean.apply(lambda row: formula(row['miles_traveled'], row['total_receipts_amount']), axis=1)
            r2 = r2_score(df_day_clean['expected_output'], test_output)
            return r2, params

        results = Parallel(n_jobs=-1, verbose=1)(delayed(find_best_params)(p) for p in param_grid)
        
        best_r2, best_params_tuple = max(results, key=lambda item: item[0])
        best_params = {
            'per_diem': best_params_tuple[0],
            'mileage_rate': best_params_tuple[1],
            'receipt_rate': best_params_tuple[2],
            'receipt_cap': best_params_tuple[3]
        }

        print(f"\nBest R2 score on cleaned {day}-day data: {best_r2:.6f}")
        print(f"Best parameters found for {day}-day trips: {best_params}")

if __name__ == '__main__':
    analyze_data_with_plots_part1() 