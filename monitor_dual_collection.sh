#!/bin/bash
# Monitor dual RL data collection runs
# Usage: bash monitor_dual_collection.sh

clear
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          DUAL RL DATA COLLECTION MONITOR                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    # Check if processes are running
    RUN_A_PID=$(pgrep -f "phase2_run_A")
    RUN_B_PID=$(pgrep -f "phase2_run_B")
    
    # Timestamp
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # GPU status
    echo "🖥️  GPU USAGE:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
        awk -F', ' '{printf "   GPU %s: %s util, %s/%s mem, %s°C\n", $1, $2, $3, $4, $5}'
    echo ""
    
    # Run A status
    echo "📊 RUN A (phase2_run_A.parquet):"
    if [ -n "$RUN_A_PID" ]; then
        echo "   Status: ✅ Running (PID: $RUN_A_PID)"
        LAST_LINE_A=$(grep "Collecting transitions:" logs/phase2_run_A.log | tail -1)
        PROGRESS_A=$(echo "$LAST_LINE_A" | grep -oP '\d+/\d+' | head -1)
        PERCENT_A=$(echo "$LAST_LINE_A" | grep -oP '\d+%' | head -1)
        SPEED_A=$(echo "$LAST_LINE_A" | grep -oP '\d+\.\d+s/it')
        ETA_A=$(echo "$LAST_LINE_A" | grep -oP '<\K\d+:\d+' | head -1)
        echo "   Progress: $PROGRESS_A samples ($PERCENT_A)"
        echo "   Speed: $SPEED_A"
        echo "   ETA: $ETA_A remaining"
    else
        echo "   Status: ⚠️  Not running"
        if [ -f "data/rl_transitions/phase2_run_A.parquet" ]; then
            SIZE=$(du -h data/rl_transitions/phase2_run_A.parquet | cut -f1)
            echo "   Output: ✅ Complete ($SIZE)"
        fi
    fi
    echo ""
    
    # Run B status
    echo "📊 RUN B (phase2_run_B.parquet):"
    if [ -n "$RUN_B_PID" ]; then
        echo "   Status: ✅ Running (PID: $RUN_B_PID)"
        LAST_LINE_B=$(grep "Collecting transitions:" logs/phase2_run_B.log | tail -1)
        PROGRESS_B=$(echo "$LAST_LINE_B" | grep -oP '\d+/\d+' | head -1)
        PERCENT_B=$(echo "$LAST_LINE_B" | grep -oP '\d+%' | head -1)
        SPEED_B=$(echo "$LAST_LINE_B" | grep -oP '\d+\.\d+s/it')
        ETA_B=$(echo "$LAST_LINE_B" | grep -oP '<\K\d+:\d+' | head -1)
        echo "   Progress: $PROGRESS_B samples ($PERCENT_B)"
        echo "   Speed: $SPEED_B"
        echo "   ETA: $ETA_B remaining"
    else
        echo "   Status: ⚠️  Not running"
        if [ -f "data/rl_transitions/phase2_run_B.parquet" ]; then
            SIZE=$(du -h data/rl_transitions/phase2_run_B.parquet | cut -f1)
            echo "   Output: ✅ Complete ($SIZE)"
        fi
    fi
    echo ""
    
    # Exit if both done
    if [ -z "$RUN_A_PID" ] && [ -z "$RUN_B_PID" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ BOTH COLLECTIONS COMPLETE!"
        echo ""
        echo "Compare results with:"
        echo "  python -c 'import pandas as pd; a=pd.read_parquet(\"data/rl_transitions/phase2_run_A.parquet\"); b=pd.read_parquet(\"data/rl_transitions/phase2_run_B.parquet\"); print(f\"A: {len(a)} samples, B: {len(b)} samples\")'"
        break
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Press Ctrl+C to exit (collections keep running in background)"
    echo ""
    
    sleep 30
    clear
done
