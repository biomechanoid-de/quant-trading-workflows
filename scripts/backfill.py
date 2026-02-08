#!/usr/bin/env python3
"""Historical backfill script for WF1 Data Ingestion.

Triggers one Flyte workflow execution per trading day from start_date to end_date.
Respects US market holidays and weekends (Mon-Fri only).

Usage:
    # Backfill from Jan 1 to today
    python scripts/backfill.py --start 2026-01-01 --end 2026-02-08

    # Dry run (print dates only, don't execute)
    python scripts/backfill.py --start 2026-01-01 --end 2026-02-08 --dry-run

    # With custom batch size (concurrent executions)
    python scripts/backfill.py --start 2026-01-01 --end 2026-02-08 --batch-size 3

    # Use development domain
    python scripts/backfill.py --start 2026-01-01 --end 2026-02-08 --domain development
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# US Market Holidays 2026 (NYSE/NASDAQ closed)
US_HOLIDAYS_2026 = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas Day
}

PHASE1_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
]


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """Get list of US trading days (Mon-Fri, excluding holidays)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    trading_days = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        # Skip weekends (5=Saturday, 6=Sunday)
        if current.weekday() < 5 and date_str not in US_HOLIDAYS_2026:
            trading_days.append(date_str)
        current += timedelta(days=1)

    return trading_days


def trigger_workflow(date: str, domain: str, project: str, image: str) -> str:
    """Trigger a single WF1 execution on Flyte for the given date."""
    symbols_json = '["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","JNJ"]'

    cmd = [
        "pyflyte", "--config", f"{Path.home()}/.flyte/config.yaml",
        "run", "--remote",
        "--project", project,
        "--domain", domain,
        "--image", image,
        "src/wf1_data_ingestion/workflow.py", "data_ingestion_workflow",
        "--symbols", symbols_json,
        "--date", date,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return f"FAILED: {result.stderr.strip()}"

    # Extract execution URL from output
    for line in result.stdout.split("\n"):
        if "http" in line:
            return line.strip()
    return "Triggered (no URL found)"


def main():
    parser = argparse.ArgumentParser(description="Backfill historical market data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--domain", default="production", help="Flyte domain (default: production)")
    parser.add_argument("--project", default="quant-trading", help="Flyte project")
    parser.add_argument("--image", default="ghcr.io/biomechanoid-de/quant-trading-workflows:latest",
                        help="Docker image to use")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Number of concurrent executions (default: 3)")
    parser.add_argument("--wait-between-batches", type=int, default=120,
                        help="Seconds to wait between batches (default: 120)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print trading days without executing")
    args = parser.parse_args()

    trading_days = get_trading_days(args.start, args.end)

    print(f"=== WF1 Historical Backfill ===")
    print(f"Period:       {args.start} to {args.end}")
    print(f"Trading days: {len(trading_days)}")
    print(f"Domain:       {args.domain}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Symbols:      {len(PHASE1_SYMBOLS)} US Large Caps")
    print()

    if args.dry_run:
        print("DRY RUN — Trading days:")
        for i, day in enumerate(trading_days, 1):
            weekday = datetime.strptime(day, "%Y-%m-%d").strftime("%a")
            print(f"  {i:3d}. {day} ({weekday})")
        print(f"\nTotal: {len(trading_days)} trading days")
        return

    # Process in batches to avoid overloading the Pi cluster
    total = len(trading_days)
    for batch_start in range(0, total, args.batch_size):
        batch = trading_days[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (total + args.batch_size - 1) // args.batch_size

        print(f"--- Batch {batch_num}/{total_batches} ---")

        for date in batch:
            weekday = datetime.strptime(date, "%Y-%m-%d").strftime("%a")
            print(f"  Triggering {date} ({weekday})... ", end="", flush=True)
            result = trigger_workflow(date, args.domain, args.project, args.image)
            print(result)

        # Wait between batches (unless it's the last batch)
        if batch_start + args.batch_size < total:
            print(f"  Waiting {args.wait_between_batches}s for cluster to process...\n")
            time.sleep(args.wait_between_batches)

    print(f"\n=== Backfill complete: {total} trading days triggered ===")


if __name__ == "__main__":
    main()
