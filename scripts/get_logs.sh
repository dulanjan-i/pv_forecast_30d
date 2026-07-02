#!/bin/bash

USER_NAME="dbfz018"
OUTPUT_FILE="my_hpc_job_history.txt"
START_DATE="1201-00:00"

echo "Generating audit for $USER_NAME starting from December..."

# Note: Using % scaling to prevent text cutting off in the WorkDir and NodeList
sacct -u $USER_NAME \
      -S $START_DATE \
      --allclusters \
      -X \
      --format=JobID,JobName%30,Partition,NodeList%20,State,ExitCode,Start,End,Elapsed,WorkDir%70 > $OUTPUT_FILE

if [ -s "$OUTPUT_FILE" ]; then
    echo "---------------------------------------------------"
    echo "SUCCESS: Log created: $OUTPUT_FILE"
    echo "Total jobs recorded: $(grep -c "^[0-9]" "$OUTPUT_FILE")"
    echo "---------------------------------------------------"
    echo "Top 5 most recent jobs:"
    tail -n 5 "$OUTPUT_FILE"
else
    echo "No data retrieved. Your account may be locked to a specific cluster."
    echo "Try: sacct -M <cluster_name> -u $USER_NAME"
fi