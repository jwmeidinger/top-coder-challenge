#!/bin/bash

# Black Box Challenge Evaluation Script
# This script tests your reimbursement calculation implementation against 1,000 historical cases

set -e

echo "🧾 Black Box Challenge - Reimbursement System Evaluation"
echo "======================================================="
echo " "

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "❌ Error: jq is required but not installed!"
    echo "Please install jq to parse JSON files:"
    echo "  macOS: brew install jq"
    echo "  Ubuntu/Debian: sudo apt-get install jq"
    echo "  CentOS/RHEL: sudo yum install jq"
    exit 1
fi

# Check if bc is available for floating point arithmetic
if ! command -v bc &> /dev/null; then
    echo "❌ Error: bc (basic calculator) is required but not installed!"
    echo "Please install bc for floating point calculations:"
    echo "  macOS: brew install bc"
    echo "  Ubuntu/Debian: sudo apt-get install bc"
    echo "  CentOS/RHEL: sudo yum install bc"
    exit 1
fi

# Check if run.sh exists
if [ ! -f "run.sh" ]; then
    echo "❌ Error: run.sh not found!"
    echo "Please create a run.sh script that takes three parameters:"
    echo "  ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    echo "  and outputs the reimbursement amount"
    exit 1
fi

# Make run.sh executable
chmod +x run.sh

# Check if public cases exist
if [ ! -f "public_cases.json" ]; then
    echo "❌ Error: public_cases.json not found!"
    echo "Please ensure the public cases file is in the current directory."
    exit 1
fi

# Get current timestamp for the results file
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="results"
mkdir -p "$results_dir"
results_file="${results_dir}/results_${timestamp}.json"
summary_file="${results_dir}/summary_${timestamp}.txt"

echo " "

# Helper function to process a single test case in parallel
run_case() {
    local i=$1
    local case_data=$2
    local results_dir=$3

    # Pad case number for correct sorting of result files
    printf -v case_num_padded "%04d" $((i+1))
    local result_file="$results_dir/result_${case_num_padded}.json"

    IFS=':' read -r trip_duration miles_traveled receipts_amount expected <<< "$case_data"

    if script_output=$(./run.sh "$trip_duration" "$miles_traveled" "$receipts_amount" 2>/dev/null); then
        output=$(echo "$script_output" | tr -d '[:space:]')
        if [[ $output =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
            local actual="$output"
            # Calculate absolute error using bc
            local error
            error=$(echo "scale=10; if ($actual - $expected < 0) -1 * ($actual - $expected) else ($actual - $expected)" | bc)

            jq -n \
                --arg case_num "$((i+1))" --arg trip_duration "$trip_duration" --arg miles_traveled "$miles_traveled" \
                --arg receipts_amount "$receipts_amount" --arg expected "$expected" --arg actual "$actual" --arg error "$error" \
                '{case_number: $case_num, input: {trip_duration_days: $trip_duration, miles_traveled: $miles_traveled, total_receipts_amount: $receipts_amount}, expected_output: $expected, actual_output: $actual, error: $error}' > "$result_file"
        else
            # Invalid output format
            local error_msg="Invalid output format: $output"
            # Print error to stderr immediately, with newlines to avoid progress bar clutter
            echo -e "\n❌ Case $((i+1)): $error_msg" >&2
            jq -n \
                --arg case_num "$((i+1))" --arg trip_duration "$trip_duration" --arg miles_traveled "$miles_traveled" \
                --arg receipts_amount "$receipts_amount" --arg expected "$expected" --arg error_msg "$error_msg" \
                '{case_number: $case_num, input: {trip_duration_days: $trip_duration, miles_traveled: $miles_traveled, total_receipts_amount: $receipts_amount}, expected_output: $expected, error: $error_msg}' > "$result_file"
        fi
    else
        # Script failed
        local error_msg
        error_msg=$(./run.sh "$trip_duration" "$miles_traveled" "$receipts_amount" 2>&1 >/dev/null | tr -d '\n')
        # Print error to stderr immediately, with newlines to avoid progress bar clutter
        echo -e "\n❌ Case $((i+1)): Script failed with error: $error_msg" >&2
        jq -n \
            --arg case_num "$((i+1))" --arg trip_duration "$trip_duration" --arg miles_traveled "$miles_traveled" \
            --arg receipts_amount "$receipts_amount" --arg expected "$expected" --arg error_msg "$error_msg" \
            '{case_number: $case_num, input: {trip_duration_days: $trip_duration, miles_traveled: $miles_traveled, total_receipts_amount: $receipts_amount}, expected_output: $expected, error: $error_msg}' > "$result_file"
    fi
}
export -f run_case

echo "📊 Running evaluation against 1,000 test cases..."
echo "📝 Results will be saved in the '${results_dir}' directory."
echo " "

# Extract all test data upfront in a single jq call for better performance
echo "Extracting test data..."
test_data=$(jq -r '.[] | "\(.input.trip_duration_days):\(.input.miles_traveled):\(.input.total_receipts_amount):\(.expected_output)"' public_cases.json)

# Convert to arrays for faster access (compatible with bash 3.2+)
test_cases=()
while IFS= read -r line; do
    test_cases+=("$line")
done <<< "$test_data"
num_cases=${#test_cases[@]}

# Determine number of parallel jobs
if command -v nproc &> /dev/null; then
    num_parallel_jobs=$(nproc)
else
    num_parallel_jobs=4 # Fallback for systems without nproc
fi
echo "🚀 Running with up to $num_parallel_jobs parallel jobs."

# Create a temporary directory for parallel results and ensure cleanup
tmp_dir=$(mktemp -d -t ci-XXXXXXXXXX)
trap 'rm -rf -- "$tmp_dir"' EXIT

# Initialize counters and arrays
successful_runs=0
exact_matches=0
close_matches=0
total_error="0"
max_error="0"
max_error_case=""
results_array=()

# Process test cases in parallel
for ((i=0; i<num_cases; i++)); do
    run_case "$i" "${test_cases[i]}" "$tmp_dir" &

    if [ $(( (i+1) % num_parallel_jobs )) -eq 0 ] || [ $((i+1)) -eq $num_cases ]; then
        wait
        echo -ne "Progress: $((i+1))/$num_cases cases processed...\r" >&2
    fi
done

echo
echo "⚙️  Aggregating results..."

# Aggregate results from parallel runs
result_files=($(find "$tmp_dir" -name 'result_*.json' | sort -V))

# Create a JSON array for detailed results
echo "[" > "$results_file"
first_entry=true

for f in "${result_files[@]}"; do
    # Append to JSON file
    if [ "$first_entry" = true ]; then
        first_entry=false
    else
        echo "," >> "$results_file"
    fi
    cat "$f" >> "$results_file"

    if [[ $(jq -e '.actual_output' "$f") != "null" ]]; then
        successful_runs=$((successful_runs + 1))
        
        actual=$(jq -r '.actual_output' "$f")
        expected=$(jq -r '.expected_output' "$f")
        error=$(jq -r '.error' "$f")
        
        # Store result for in-memory sorting
        trip_duration=$(jq -r '.input.trip_duration_days' "$f")
        miles_traveled=$(jq -r '.input.miles_traveled' "$f")
        receipts_amount=$(jq -r '.input.total_receipts_amount' "$f")
        case_num=$(jq -r '.case_number' "$f")
        results_array+=("$case_num:$expected:$actual:$error:$trip_duration:$miles_traveled:$receipts_amount")
        
        # Check for exact match (within $0.01)
        if (( $(echo "$error < 0.01" | bc -l) )); then
            exact_matches=$((exact_matches + 1))
        fi
        
        # Check for close match (within $1.00)
        if (( $(echo "$error < 1.0" | bc -l) )); then
            close_matches=$((close_matches + 1))
        fi
        
        # Update total error
        total_error=$(echo "scale=10; $total_error + $error" | bc)
        
        # Track maximum error
        if (( $(echo "$error > $max_error" | bc -l) )); then
            max_error="$error"
            max_error_case="Case $case_num: $trip_duration days, $miles_traveled miles, \$$receipts_amount receipts"
        fi
    else
        # Error was already printed in real-time by the child process.
        # The failed case is still added to the results.json, but no console output is needed here.
        :
    fi
done

# Calculate and display results
if [ $successful_runs -eq 0 ]; then
    echo "❌ No successful test cases!"
    echo ""
    echo "Your script either:"
    echo "  - Failed to run properly"
    echo "  - Produced invalid output format"
    echo "  - Timed out on all cases"
    echo ""
    echo "Check the errors above for details."
    
    # Add summary to JSON
    echo "," >> "$results_file"
    jq -n '{summary: {status: "failed", message: "No successful test cases"}}' >> "$results_file"
else
    # Calculate average error
    avg_error=$(echo "scale=2; $total_error / $successful_runs" | bc)
    
    # Calculate percentages
    exact_pct=$(echo "scale=1; $exact_matches * 100 / $successful_runs" | bc)
    close_pct=$(echo "scale=1; $close_matches * 100 / $successful_runs" | bc)
    
    # Tee output to both console and summary file
    {
        echo "✅ Evaluation Complete!"
        echo ""
        echo "📈 Results Summary:"
        echo "  Total test cases: $num_cases"
        echo "  Successful runs: $successful_runs"
        echo "  Exact matches (±\$0.01): $exact_matches (${exact_pct}%)"
        echo "  Close matches (±\$1.00): $close_matches (${close_pct}%)"
        echo "  Average error: \$${avg_error}"
        echo "  Maximum error: \$${max_error}"
        echo ""
        
        # Calculate score (lower is better)
        score=$(echo "scale=2; $avg_error * 100 + ($num_cases - $exact_matches) * 0.1" | bc)
        echo "🎯 Your Score: $score (lower is better)"
        echo ""
        
        # Provide feedback based on exact matches
        if [ $exact_matches -eq $num_cases ]; then
            feedback="PERFECT SCORE! You have reverse-engineered the system completely!" ## THE ONLY THIG ACCEPTABLE
            echo "🏆 $feedback"
        elif [ $exact_matches -gt 950 ]; then
            feedback="Excellent! You are very close to the perfect solution."
            echo "🥇 $feedback"
        elif [ $exact_matches -gt 800 ]; then
            feedback="Great work! You have captured most of the system behavior."
            echo "🥈 $feedback"
        elif [ $exact_matches -gt 500 ]; then
            feedback="Good progress! You understand some key patterns."
            echo "🥉 $feedback"
        else
            feedback="Keep analyzing the patterns in the interviews and test cases."
            echo "📚 $feedback"
        fi
    } | tee "$summary_file"
    
    # Add summary to JSON
    echo "," >> "$results_file"
    jq -n \
        --arg total "$num_cases" \
        --arg successful "$successful_runs" \
        --arg exact "$exact_matches" \
        --arg exact_pct "$exact_pct" \
        --arg close "$close_matches" \
        --arg close_pct "$close_pct" \
        --arg avg_error "$avg_error" \
        --arg max_error "$max_error" \
        --arg max_case "$max_error_case" \
        --arg score "$score" \
        --arg feedback "$feedback" \
        --arg timestamp "$(date)" \
        '{summary: {timestamp: $timestamp, total_cases: $total, successful_runs: $successful, exact_matches: $exact, exact_matches_percent: $exact_pct, close_matches: $close, close_matches_percent: $close_pct, average_error: $avg_error, maximum_error: $max_error, maximum_error_case: $max_case, score: $score, feedback: $feedback}}' >> "$results_file"
    
    echo ""
    echo "💡 Tips for improvement:"
    if [ $exact_matches -lt $num_cases ]; then
        echo "  Check these high-error cases:"
        
        # Sort results by error (descending) in memory and show top 5
        IFS=$'\n' high_error_cases=($(printf '%s\n' "${results_array[@]}" | sort -t: -k4 -nr | head -5))
        for result in "${high_error_cases[@]}"; do
            IFS=: read -r case_num expected actual error trip_duration miles_traveled receipts_amount <<< "$result"
            printf "    Case %s: %s days, %s miles, \$%s receipts\n" "$case_num" "$trip_duration" "$miles_traveled" "$receipts_amount"
            printf "      Expected: \$%.2f, Got: \$%.2f, Error: \$%.2f\n" "$expected" "$actual" "$error"
        done
    fi
fi

# Close the JSON array
echo "]" >> "$results_file"

echo
echo "📝 Next steps:"
echo "  1. Fix any script errors shown above"
echo "  2. Ensure your run.sh outputs only a number"
echo "  3. Analyze the patterns in the interviews and public cases"
echo "  4. Test edge cases around trip length and receipt amounts"
echo "  5. Submit your solution via the Google Form when ready!"
echo
echo "📊 Detailed results saved to: ${results_file}"
echo "📝 Summary saved to: ${summary_file}" 