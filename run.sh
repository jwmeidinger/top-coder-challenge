#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Convert solution.py to Unix line endings if needed
if grep -q $'\r' solution.py; then
  # Create a temporary file with Unix line endings
  tr -d '\r' < solution.py > solution_unix.py
  # Use the converted file
  python solution_unix.py "$@"
else
  # Use the original file if already using Unix line endings
  python solution.py "$@"
fi 