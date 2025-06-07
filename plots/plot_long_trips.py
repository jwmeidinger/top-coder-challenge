import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_long_trip_data():
    """
    Creates focused plots for trips lasting 8 to 14 days to help with analysis.
    """
    with open('public_cases.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame([d['input'] for d in data])
    df['expected_output'] = [d['expected_output'] for d in data]

    # Filter for long trips
    df_long = df[df['trip_duration_days'] >= 8].copy()
    
    # --- Outlier/Bug Identification ---
    outlier_condition = (df_long['total_receipts_amount'] > 500) & (df_long['expected_output'] < 500)
    df_long['is_outlier'] = outlier_condition
    
    print("Long Trip Data Description:")
    print(df_long.describe())

    # Create plots for long trips
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df_long, x='miles_traveled', y='expected_output', hue='trip_duration_days', style='is_outlier', palette='viridis', alpha=0.8, s=80)
    plt.title('Long Trips (8-14 Days): Miles Traveled vs. Expected Output')
    plt.xlabel('Miles Traveled')
    plt.ylabel('Expected Output')
    plt.grid(True)
    plt.savefig('long_trips_miles_vs_output.png')
    print("\nGenerated long_trips_miles_vs_output.png")

    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df_long, x='total_receipts_amount', y='expected_output', hue='trip_duration_days', style='is_outlier', palette='viridis', alpha=0.8, s=80)
    plt.title('Long Trips (8-14 Days): Total Receipts Amount vs. Expected Output')
    plt.xlabel('Total Receipts Amount')
    plt.ylabel('Expected Output')
    plt.grid(True)
    plt.savefig('long_trips_receipts_vs_output.png')
    print("Generated long_trips_receipts_vs_output.png")

    # Efficiency Plot
    df_long['miles_per_day'] = df_long['miles_traveled'] / df_long['trip_duration_days']
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df_long, x='miles_per_day', y='expected_output', hue='trip_duration_days', style='is_outlier', palette='viridis', alpha=0.8, s=80)
    plt.title('Long Trips (8-14 Days): Efficiency (Miles/Day) vs. Expected Output')
    plt.xlabel('Miles per Day')
    plt.ylabel('Expected Output')
    plt.grid(True)
    plt.savefig('long_trips_efficiency_vs_output.png')
    print("Generated long_trips_efficiency_vs_output.png")

if __name__ == '__main__':
    plot_long_trip_data() 