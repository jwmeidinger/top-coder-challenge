import matplotlib
matplotlib.use('Agg')
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_scatter_plots(json_path, output_image_path, top_n=20):
    """
    Analyzes trip results, focusing on 1-7 day trips, and generates
    a set of scatter plots to visualize error patterns.

    Args:
        json_path (str): Path to the input JSON file.
        output_image_path (str): Path to save the output PNG image.
        top_n (int): Number of worst cases to highlight.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    cases = [item for item in data if 'case_number' in item]
    if not cases:
        print("No cases found in the JSON file.")
        return

    df = pd.DataFrame(cases)
    df_input = pd.json_normalize(df['input'])
    df = pd.concat([df.drop('input', axis=1), df_input], axis=1)

    numeric_cols = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount',
                    'expected_output', 'actual_output', 'error']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=numeric_cols, inplace=True)
    df['trip_duration_days'] = df['trip_duration_days'].astype(int)

    # Filter for trip duration between 1 and 7 days
    filtered_df = df[df['trip_duration_days'].between(1, 7)].copy()

    # Identify the top N worst cases
    worst_cases = filtered_df.sort_values('error', ascending=False).head(top_n)
    
    # Print the worst cases table
    print(f"Top {top_n} worst cases by error (1-7 day trips) from {os.path.basename(json_path)}:")
    print(worst_cases[['case_number', 'trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output', 'actual_output', 'error']].to_string())

    # --- Plotting ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Error Analysis for 1-7 Day Trips\n(Source: {os.path.basename(json_path)})', fontsize=22, y=1.03)
    
    plot_vars = [('trip_duration_days', 'Trip Duration (Days)'), 
                 ('miles_traveled', 'Miles Traveled'), 
                 ('total_receipts_amount', 'Total Receipts Amount ($)')]

    # Plot Error vs. Input Variables
    for i, (var, title) in enumerate(plot_vars):
        ax = axes[i // 2, i % 2]
        # Plot all points in blue
        ax.scatter(filtered_df[var], filtered_df['error'], alpha=0.5, label='All Trips (1-7 days)', color='cornflowerblue')
        # Highlight worst cases in red
        ax.scatter(worst_cases[var], worst_cases['error'], color='red', s=80, edgecolor='white', linewidth=0.5, label=f'Top {top_n} Errors')
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('Error ($)', fontsize=12)
        ax.set_title(f'Error vs. {title}', fontsize=14, pad=15)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()

    # Plot Actual vs. Expected Output
    ax3 = axes[1, 1]
    min_val = min(filtered_df['expected_output'].min(), filtered_df['actual_output'].min())
    max_val = max(filtered_df['expected_output'].max(), filtered_df['actual_output'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'w--', label='Perfect Match', alpha=0.7)
    ax3.scatter(filtered_df['expected_output'], filtered_df['actual_output'], alpha=0.5, label='All Trips (1-7 days)', color='cornflowerblue')
    ax3.scatter(worst_cases['expected_output'], worst_cases['actual_output'], color='red', s=80, edgecolor='white', linewidth=0.5, label=f'Top {top_n} Errors')
    ax3.set_xlabel('Expected Output ($)', fontsize=12)
    ax3.set_ylabel('Actual Output ($)', fontsize=12)
    ax3.set_title('Actual vs. Expected Output', fontsize=14, pad=15)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the plot
    plt.savefig(output_image_path)
    print(f"\nScatter plots saved to {output_image_path}")

if __name__ == '__main__':
    json_file = 'results/results_20250607_172454.json'
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'error_scatter_analysis_172454.png')
    
    create_scatter_plots(json_file, output_file) 