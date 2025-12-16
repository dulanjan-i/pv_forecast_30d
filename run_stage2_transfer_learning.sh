#!/bin/bash
#
# Stage 2: Transfer Learning - Run all 6 Germany plants in parallel batches
# Usage: bash run_stage2_transfer_learning.sh
#
# Hardware: calc02 with 4x NVIDIA L4 GPUs (24GB each)
# Strategy: Wave 1 (4 plants on GPU 0-3), Wave 2 (2 plants on GPU 0-1)

set -e  # Exit on error

# Activate virtual environment
source ~/.venvs/pvforecast/bin/activate

# Set working directory
cd ~/pv_forecast_30d

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Stage 2: Transfer Learning from Farm2107${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Pretrained weights: experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt"
echo "Target: 6 Germany plants (plant_01 through plant_06)"
echo "Hardware: 4x NVIDIA L4 GPUs"
echo ""

# Verify pretrained weights exist
if [ ! -f "experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt" ]; then
    echo -e "${RED}ERROR: Pretrained weights not found!${NC}"
    echo "Expected: experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt"
    exit 1
fi

# Verify all YAML configs exist
for plant_id in 01 02 03 04 05 06; do
    yaml_path="experiments/lstm/germany/pretrain_plant_${plant_id}.yaml"
    if [ ! -f "$yaml_path" ]; then
        echo -e "${RED}ERROR: Config not found: $yaml_path${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✓ All configs and pretrained weights verified${NC}"
echo ""

# Create log directory
LOG_DIR="experiments/lstm/runs/germany/logs"
mkdir -p "$LOG_DIR"

# Launch monitoring terminals
echo -e "${YELLOW}Launching monitoring windows...${NC}"

# Try to detect terminal emulator and launch monitors
if command -v gnome-terminal &> /dev/null; then
    # CPU monitoring
    gnome-terminal --title="CPU Monitor" -- bash -c "htop; exec bash" &
    
    # GPU monitoring (all 4 GPUs)
    gnome-terminal --title="GPU Monitor" -- bash -c "watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | column -t -s,'; exec bash" &
    
    echo "  ✓ Opened CPU monitor (htop)"
    echo "  ✓ Opened GPU monitor (nvidia-smi)"
elif command -v xterm &> /dev/null; then
    xterm -title "CPU Monitor" -e "htop" &
    xterm -title "GPU Monitor" -e "watch -n 1 nvidia-smi" &
    echo "  ✓ Opened monitors in xterm"
elif command -v konsole &> /dev/null; then
    konsole --title "CPU Monitor" -e htop &
    konsole --title "GPU Monitor" -e "watch -n 1 nvidia-smi" &
    echo "  ✓ Opened monitors in konsole"
else
    echo -e "  ${YELLOW}⚠ No supported terminal emulator found${NC}"
    echo "  Run these commands manually in separate terminals:"
    echo "    Terminal 1: htop"
    echo "    Terminal 2: watch -n 1 nvidia-smi"
fi

sleep 2  # Give terminals time to open
echo ""

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}WAVE 1: Plants 01-04 (Parallel on 4 GPUs)${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Wave 1: 4 plants in parallel
CUDA_VISIBLE_DEVICES=0 python3 src/training/pretrain_lstm.py \
    --config experiments/lstm/germany/pretrain_plant_01.yaml \
    > "$LOG_DIR/plant_01.log" 2>&1 &
PID1=$!
echo "  [GPU 0] Plant 01 started (PID: $PID1)"

CUDA_VISIBLE_DEVICES=1 python3 src/training/pretrain_lstm.py \
    --config experiments/lstm/germany/pretrain_plant_02.yaml \
    > "$LOG_DIR/plant_02.log" 2>&1 &
PID2=$!
echo "  [GPU 1] Plant 02 started (PID: $PID2)"

CUDA_VISIBLE_DEVICES=2 python3 src/training/pretrain_lstm.py \
    --config experiments/lstm/germany/pretrain_plant_03.yaml \
    > "$LOG_DIR/plant_03.log" 2>&1 &
PID3=$!
echo "  [GPU 2] Plant 03 started (PID: $PID3)"

CUDA_VISIBLE_DEVICES=3 python3 src/training/pretrain_lstm.py \
    --config experiments/lstm/germany/pretrain_plant_04.yaml \
    > "$LOG_DIR/plant_04.log" 2>&1 &
PID4=$!
echo "  [GPU 3] Plant 04 started (PID: $PID4)"

echo ""
echo "Waiting for Wave 1 to complete..."
echo "  → Logs: $LOG_DIR/plant_0{1,2,3,4}.log"
echo "  → Monitor: tail -f $LOG_DIR/plant_01.log"
echo ""

# Wait for all Wave 1 jobs
wait $PID1
EXIT1=$?
wait $PID2
EXIT2=$?
wait $PID3
EXIT3=$?
wait $PID4
EXIT4=$?

# Check Wave 1 results
WAVE1_FAILED=0
[ $EXIT1 -ne 0 ] && echo -e "${RED}✗ Plant 01 FAILED (exit code: $EXIT1)${NC}" && WAVE1_FAILED=1 || echo -e "${GREEN}✓ Plant 01 completed${NC}"
[ $EXIT2 -ne 0 ] && echo -e "${RED}✗ Plant 02 FAILED (exit code: $EXIT2)${NC}" && WAVE1_FAILED=1 || echo -e "${GREEN}✓ Plant 02 completed${NC}"
[ $EXIT3 -ne 0 ] && echo -e "${RED}✗ Plant 03 FAILED (exit code: $EXIT3)${NC}" && WAVE1_FAILED=1 || echo -e "${GREEN}✓ Plant 03 completed${NC}"
[ $EXIT4 -ne 0 ] && echo -e "${RED}✗ Plant 04 FAILED (exit code: $EXIT4)${NC}" && WAVE1_FAILED=1 || echo -e "${GREEN}✓ Plant 04 completed${NC}"

if [ $WAVE1_FAILED -eq 1 ]; then
    echo ""
    echo -e "${RED}Wave 1 had failures. Check logs in $LOG_DIR${NC}"
    echo -e "${YELLOW}Continuing with Wave 2 anyway...${NC}"
fi

echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}WAVE 2: Plants 05-06 (Parallel on 2 GPUs)${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Wave 2: 2 plants in parallel
CUDA_VISIBLE_DEVICES=0 python3 src/training/pretrain_lstm.py \
    --config experiments/lstm/germany/pretrain_plant_05.yaml \
    > "$LOG_DIR/plant_05.log" 2>&1 &
PID5=$!
echo "  [GPU 0] Plant 05 started (PID: $PID5)"

CUDA_VISIBLE_DEVICES=1 python3 src/training/pretrain_lstm.py \
    --config experiments/lstm/germany/pretrain_plant_06.yaml \
    > "$LOG_DIR/plant_06.log" 2>&1 &
PID6=$!
echo "  [GPU 1] Plant 06 started (PID: $PID6)"

echo ""
echo "Waiting for Wave 2 to complete..."
echo "  → Logs: $LOG_DIR/plant_0{5,6}.log"
echo ""

# Wait for all Wave 2 jobs
wait $PID5
EXIT5=$?
wait $PID6
EXIT6=$?

# Check Wave 2 results
WAVE2_FAILED=0
[ $EXIT5 -ne 0 ] && echo -e "${RED}✗ Plant 05 FAILED (exit code: $EXIT5)${NC}" && WAVE2_FAILED=1 || echo -e "${GREEN}✓ Plant 05 completed${NC}"
[ $EXIT6 -ne 0 ] && echo -e "${RED}✗ Plant 06 FAILED (exit code: $EXIT6)${NC}" && WAVE2_FAILED=1 || echo -e "${GREEN}✓ Plant 06 completed${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Stage 2 Transfer Learning COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Summary
TOTAL_FAILED=$(($WAVE1_FAILED + $WAVE2_FAILED))
if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All 6 plants completed successfully!${NC}"
    echo ""
    echo "Output encoders:"
    for plant_id in 01 02 03 04 05 06; do
        encoder_path="experiments/lstm/encoders/lstm_encoder_plant_${plant_id}.pt"
        if [ -f "$encoder_path" ]; then
            size=$(du -h "$encoder_path" | cut -f1)
            echo "  ✓ $encoder_path ($size)"
        else
            echo -e "  ${RED}✗ $encoder_path (NOT FOUND)${NC}"
        fi
    done
    echo ""
    echo "Next steps:"
    echo "  1. Check validation metrics in logs: grep 'val_loss' $LOG_DIR/*.log"
    echo "  2. Proceed to Stage 2B: TFT ensemble training"
else
    echo -e "${YELLOW}⚠ Some plants failed. Review logs:${NC}"
    echo "  ls -lh $LOG_DIR/"
    echo ""
    echo "To retry failed plants, run individually:"
    echo "  python3 src/training/pretrain_lstm.py --config experiments/lstm/germany/pretrain_plant_XX.yaml"
fi
