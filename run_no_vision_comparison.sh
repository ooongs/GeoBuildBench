#!/bin/bash
# Run no-vision comparison on sampled problems from existing vision results

set -e

# Default values
MODEL="gpt-5.1"
N=200
SUCCESS_RATIO=0.7
MIN_ITERATIONS=2
MAX_ITER=5
SEED=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --success-ratio)
            SUCCESS_RATIO="$2"
            shift 2
            ;;
        --min-iterations)
            MIN_ITERATIONS="$2"
            shift 2
            ;;
        --max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --no-save-images)
            NO_SAVE_IMAGES="--no-save-images"
            shift
            ;;
        --vision-run-info)
            VISION_RUN_INFO="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL              Model to test (default: gpt-5.1)"
            echo "  --n N                      Number of problems to sample (default: 100)"
            echo "  --success-ratio RATIO      Ratio of successful problems (default: 0.7)"
            echo "  --min-iterations N         Minimum iterations to include (default: 2)"
            echo "  --max-iter N               Maximum iterations per problem (default: 5)"
            echo "  --seed N                   Random seed (default: 42)"
            echo "  --verbose                  Enable verbose output"
            echo "  --no-save-images           Disable saving intermediate images"
            echo "  --vision-run-info PATH     Path to vision run_info.json"
            echo "  --help                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --n 50 --success-ratio 0.8"
            echo "  $0 --model gpt-4o --verbose"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python scripts/run_no_vision_comparison.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --n $N"
CMD="$CMD --success-ratio $SUCCESS_RATIO"
CMD="$CMD --min-iterations $MIN_ITERATIONS"
CMD="$CMD --max-iter $MAX_ITER"
CMD="$CMD --seed $SEED"

if [ -n "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

if [ -n "$NO_SAVE_IMAGES" ]; then
    CMD="$CMD $NO_SAVE_IMAGES"
fi

if [ -n "$VISION_RUN_INFO" ]; then
    CMD="$CMD --vision-run-info $VISION_RUN_INFO"
fi

# Print configuration
echo "========================================"
echo "No-Vision Comparison Configuration"
echo "========================================"
echo "Model:              $MODEL"
echo "Sample size:        $N problems"
echo "Success ratio:      ${SUCCESS_RATIO} (${SUCCESS_RATIO}0% success, $((100 - ${SUCCESS_RATIO}0))% failure)"
echo "Min iterations:     >= $MIN_ITERATIONS (excludes first-attempt successes)"
echo "Max iterations:     $MAX_ITER"
echo "Random seed:        $SEED"
echo "Verbose:            ${VERBOSE:-disabled}"
echo "Save images:        ${NO_SAVE_IMAGES:+disabled}"
echo "========================================"
echo ""

# Run the script
echo "Running: $CMD"
echo ""
eval $CMD
