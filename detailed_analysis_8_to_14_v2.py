import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import itertools

def analyze_data_with_plots_part2():
    """
    Analyzes the travel expense data for trips lasting 8 to 14 days.
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

    # --- Analysis Loop for Days 8-14 ---
    for day in range(8, 15):
        print("\n" + "="*80)
        print(f"Refining formula for {day}-day trips (excluding identified outliers).")
        print("="*80)

        df_day = df[df['trip_duration_days'] == day]
        df_day_clean = df_day.drop(outlier_indices, errors='ignore')
        print(f"Analyzing {len(df_day_clean)} of {len(df_day)} {day}-day trips.")

        if len(df_day_clean) < 5:
            print("Not enough data to find a reliable formula.")
            continue

        # Define parameter ranges for a "sweet spot" efficiency bonus
        per_diems = range(100 * (day-4), 100 * day + 1, 50)
        mileage_rates = [x / 100 for x in range(20, 51, 10)]
        receipt_rates = [x / 100 for x in range(10, 41, 10)]
        
        # Efficiency Sweet Spot
        threshold1s = [50, 100, 150]
        threshold2s = [200, 250, 300]
        flat_bonuses = [100, 200, 300, 400, 500]

        param_grid = list(itertools.product(per_diems, mileage_rates, receipt_rates, threshold1s, threshold2s, flat_bonuses))
        print(f"Optimizing over {len(param_grid)} combinations for {day}-day trips...")

        def find_best_params(params):
            from sklearn.metrics import r2_score
            per_diem, m_rate, r_rate, t1, t2, bonus = params
            
            # Use a fixed cap for simplicity, can be tuned later
            r_cap = 1000

            def formula(miles, receipts, duration):
                miles_per_day = miles / duration
                efficiency_bonus = 0
                if t1 <= miles_per_day < t2:
                    efficiency_bonus = bonus
                
                mileage = miles * m_rate
                receipt_reimbursement = min(receipts * r_rate, r_cap)
                return per_diem + mileage + receipt_reimbursement + efficiency_bonus

            test_output = df_day_clean.apply(lambda row: formula(row['miles_traveled'], row['total_receipts_amount'], row['trip_duration_days']), axis=1)
            r2 = r2_score(df_day_clean['expected_output'], test_output)
            return r2, params

        results = Parallel(n_jobs=-1, verbose=0)(delayed(find_best_params)(p) for p in param_grid)
        
        best_r2, best_params_tuple = max(results, key=lambda item: item[0])
        best_params = {
            'per_diem': best_params_tuple[0],
            'mileage_rate': best_params_tuple[1],
            'receipt_rate': best_params_tuple[2],
            'receipt_cap': 1000, # Hardcoded for now
            'efficiency_threshold_1': best_params_tuple[3],
            'efficiency_threshold_2': best_params_tuple[4],
            'flat_bonus': best_params_tuple[5]
        }

        print(f"\nBest R2 score on cleaned {day}-day data: {best_r2:.6f}")
        print(f"Best parameters found for {day}-day trips: {best_params}")

    print("\n" + "="*80)
    print("Training a Gradient Boosting Model for trips of 8-14 days.")
    print("="*80)

    df_long_trips = df[df['trip_duration_days'] >= 8]
    df_long_trips_clean = df_long_trips.drop(outlier_indices, errors='ignore')
    
    X = df_long_trips_clean[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y = df_long_trips_clean['expected_output']
    
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=5, learning_rate=0.05)
    gbr.fit(X, y)
    
    from sklearn.metrics import r2_score
    print(f"R2 score for the long-trip model: {r2_score(y, gbr.predict(X)):.6f}")

    import joblib
    joblib.dump(gbr, 'reimbursement_model_8_to_14.joblib')
    print("Long-trip model saved to reimbursement_model_8_to_14.joblib")

    print("\n" + "="*80)
    print("Training a Gradient Boosting Model for LONG-TRIP OUTLIERS.")
    print("="*80)

    df_outliers = df.loc[outlier_indices]
    
    X_outliers = df_outliers[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y_outliers = df_outliers['expected_output']

    if not df_outliers.empty:
        gbr_outliers = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3, learning_rate=0.05)
        gbr_outliers.fit(X_outliers, y_outliers)
        
        print(f"R2 score for the long-trip OUTLIER model: {r2_score(y_outliers, gbr_outliers.predict(X_outliers)):.6f}")
        joblib.dump(gbr_outliers, 'reimbursement_model_8_to_14_outliers.joblib')
        print("Long-trip outlier model saved to reimbursement_model_8_to_14_outliers.joblib")
    else:
        print("No outliers found to train a separate model.")


if __name__ == '__main__':
    # We can run both parts if needed, but for now, we only need to train the long-trip model.
    # analyze_data_with_plots_part1() 
    analyze_data_with_plots_part2() 