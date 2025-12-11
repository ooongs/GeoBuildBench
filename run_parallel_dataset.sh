#!/bin/bash
# Parallel Dataset Creation Script
# Process multiple ranges in parallel to improve speed

set -e  # Stop on error

# Configuration
INPUT_DIR="${INPUT_DIR:-data-5/GeoQA3/json}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
BATCH_SIZE="${BATCH_SIZE:-100}"  # Size of each range
MAX_PARALLEL="${MAX_PARALLEL:-4}"  # Maximum number of parallel processes
MODEL="${MODEL:-gpt-4.1}"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Parallel Dataset Creation Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}✗ Error: Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check number of JSON files
TOTAL_FILES=$(find "$INPUT_DIR" -name "*.json" | wc -l | tr -d ' ')
echo "Total files: $TOTAL_FILES"
echo "Batch size: $BATCH_SIZE"
echo "Maximum parallel processes: $MAX_PARALLEL"
echo ""

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  Warning: OPENAI_API_KEY is not set.${NC}"
    echo "Classification and difficulty evaluation will not be available."
    echo ""
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi  
fi

# Calculate number of batches
NUM_BATCHES=$(( (TOTAL_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Total batches: $NUM_BATCHES"
echo ""

# Create log directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Create PID directory
PID_DIR="$OUTPUT_DIR/pids"
mkdir -p "$PID_DIR"

# Function to process each batch
process_batch() {
    local start=$1
    local end=$2
    local batch_num=$3
    
    local log_file="$LOG_DIR/batch_${start}_${end}.log"
    local pid_file="$PID_DIR/batch_${start}_${end}.pid"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch $batch_num started: $start ~ $end" | tee -a "$log_file"
    
    python scripts/create_dataset.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --start "$start" \
        --end "$end" \
        --model "$MODEL" \
        >> "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch $batch_num completed: $start ~ $end" | tee -a "$log_file"
        rm -f "$pid_file"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch $batch_num failed: $start ~ $end (exit code: $exit_code)" | tee -a "$log_file"
        rm -f "$pid_file"
        return $exit_code
    fi
}

# Manage parallel execution
declare -a pids=()
declare -a batch_starts=()
declare -a batch_ends=()
declare -a batch_nums=()

batch_num=0
for ((start=0; start<TOTAL_FILES; start+=BATCH_SIZE)); do
    end=$((start + BATCH_SIZE))
    if [ $end -gt $TOTAL_FILES ]; then
        end=$TOTAL_FILES
    fi
    
    batch_num=$((batch_num + 1))
    
    # Wait for maximum parallel processes
    while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
        # Check completed processes
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                # Process completed
                wait "${pids[$i]}"
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo -e "${RED}✗ Batch ${batch_nums[$i]} failed (${batch_starts[$i]} ~ ${batch_ends[$i]})${NC}"
                else
                    echo -e "${GREEN}✓ Batch ${batch_nums[$i]} completed (${batch_starts[$i]} ~ ${batch_ends[$i]})${NC}"
                fi
                unset pids[$i]
                unset batch_starts[$i]
                unset batch_ends[$i]
                unset batch_nums[$i]
            fi
        done
        
        # Reindex arrays
        pids=("${pids[@]}")
        batch_starts=("${batch_starts[@]}")
        batch_ends=("${batch_ends[@]}")
        batch_nums=("${batch_nums[@]}")
        
        sleep 1
    done
    
    # Start new batch
    echo -e "${YELLOW}→ Batch $batch_num started: $start ~ $end${NC}"
    process_batch "$start" "$end" "$batch_num" &
    pid=$!
    
    pids+=($pid)
    batch_starts+=($start)
    batch_ends+=($end)
    batch_nums+=($batch_num)
    
    # Save PID
    echo $pid > "$PID_DIR/batch_${start}_${end}.pid"
done

# Wait for all processes to complete
echo ""
echo "All batches started. Waiting for completion..."
echo ""

for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo -e "${RED}✗ Batch ${batch_nums[$i]} failed (${batch_starts[$i]} ~ ${batch_ends[$i]})${NC}"
    else
        echo -e "${GREEN}✓ Batch ${batch_nums[$i]} completed (${batch_starts[$i]} ~ ${batch_ends[$i]})${NC}"
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All batches processed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To merge the final dataset:"
echo "  python scripts/create_dataset.py --merge"
echo ""
echo "Log file location: $LOG_DIR"
echo ""



