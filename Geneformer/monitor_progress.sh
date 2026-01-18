#!/bin/bash
# Monitor progress of all running evaluations

cd /home/nishioka/LUH/BioLLM/Geneformer

echo "============================================================"
echo "Monitoring Evaluation Progress"
echo "============================================================"
echo ""

# Check running processes
echo "=== Running Processes ==="
ps aux | grep -E "python.*run_.*\.py" | grep -v grep | awk '{print $2, $11, $12, $13, $14}'
echo ""

# Show latest logs
echo "=== Latest Logs ==="
for log in logs/*.log; do
    if [ -f "$log" ]; then
        echo ""
        echo "--- $(basename $log) (last 5 lines) ---"
        tail -5 "$log" 2>/dev/null | sed 's/^/  /'
    fi
done

echo ""
echo "=== Progress Summary ==="
echo "Geneformer Fine-tuning:"
if [ -f "logs/geneformer_finetune_v6.log" ]; then
    tail -3 "logs/geneformer_finetune_v6.log" 2>/dev/null | grep -E "(step|epoch|loss)" | tail -1
fi

echo ""
echo "Check logs/ directory for detailed logs"
echo "Press Ctrl+C to stop monitoring"
