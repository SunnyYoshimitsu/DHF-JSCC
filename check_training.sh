#!/bin/bash
# Quick Training Status Check

echo "════════════════════════════════════════════════════════════════"
echo "🎮 GPU STATUS"
echo "════════════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader | \
  awk -F', ' '{printf "GPU %s: %s\n  Temp: %s | Util: %s | Memory: %s / %s | Power: %s\n", $1, $2, $3, $4, $6, $7, $8}'

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🚂 TRAINING PROCESS"
echo "════════════════════════════════════════════════════════════════"
if ps aux | grep -q "[m]ain22.py"; then
    echo "✅ Training is RUNNING"
    ps aux | grep "[m]ain22.py" | awk '{printf "  PID: %s | CPU: %s%% | Memory: %s MB\n", $2, $3, int($6/1024)}'
else
    echo "❌ Training is NOT running"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📊 LATEST CHECKPOINTS"
echo "════════════════════════════════════════════════════════════════"
if [ -d "checkpoints/10_07/pth" ]; then
    echo "📁 .pth files (state dicts):"
    ls -lht checkpoints/10_07/pth/*.pth 2>/dev/null | head -3 | awk '{print "  " $9 " (" $5 ", " $6 " " $7 ")"}'
else
    echo "  No checkpoints yet"
fi

echo ""
if [ -d "checkpoints/10_07/pkl" ]; then
    echo "📁 .pkl files (full models):"
    ls -lht checkpoints/10_07/pkl/*.pkl 2>/dev/null | head -3 | awk '{print "  " $9 " (" $5 ", " $6 " " $7 ")"}'
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📝 RECENT TRAINING LOG (last 10 lines)"
echo "════════════════════════════════════════════════════════════════"
tail -10 training.log 2>/dev/null || echo "  No training log yet"
echo "════════════════════════════════════════════════════════════════"
