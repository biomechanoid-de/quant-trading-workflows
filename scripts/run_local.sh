#!/bin/bash
# Run WF1 Data Ingestion workflow locally
# Usage: ./scripts/run_local.sh [date]
#
# Examples:
#   ./scripts/run_local.sh              # Today's date
#   ./scripts/run_local.sh 2026-02-07   # Specific date

DATE=${1:-$(date +%Y-%m-%d)}

echo "============================================="
echo "  WF1: Data Ingestion Pipeline (local run)"
echo "============================================="
echo "Date:    $DATE"
echo "Symbols: AAPL, MSFT, GOOGL (dev subset)"
echo ""

pyflyte run src/wf1_data_ingestion/workflow.py data_ingestion_workflow \
    --symbols '["AAPL", "MSFT", "GOOGL"]' \
    --date "$DATE"
