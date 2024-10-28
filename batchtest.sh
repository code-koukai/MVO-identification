#!/bin/bash  

# Configuration file path  
CONFIG_PATH=" "  
# Weight file path prefix  
WEIGHT_PATH_PREFIX=" "  
# AUC output log file  
LOG_FILE="auc_results.log"  

# Clear the previous log file  
> $LOG_FILE  

# Loop from 1000 to 10000, with a step of 1000  
for ((i=1000; i<=10000; i+=1000)); do  
    # Current weight file path  
    WEIGHT_PATH="${WEIGHT_PATH_PREFIX}${i}.pth"  
    
    # Print the current command being executed  
    echo "Running test with $WEIGHT_PATH"  
    
    # Run dist_test.sh and store the output in a temporary file  
    OUTPUT=$(./tools/dist_test.sh $CONFIG_PATH $WEIGHT_PATH 1 2>&1)  
    
    # Extract the AUC value  
    AUC=$(echo "$OUTPUT" | grep -oP 'auc:\s*\K[\d\.]+')  
    
    # If AUC extraction fails, notify and skip this iteration  
    if [ -z "$AUC" ]; then  
        echo "AUC not found for iteration ${i}. Skipping..." | tee -a $LOG_FILE  
        continue  
    fi  

    # Output the AUC value for the current iteration to the console and write to the log file  
    echo "Iteration ${i}, AUC: $AUC" | tee -a $LOG_FILE  
done  

echo "AUC results saved to $LOG_FILE"