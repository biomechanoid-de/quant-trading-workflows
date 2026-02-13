#!/usr/bin/env python3
"""Historical backfill script for WF1 Data Ingestion.

Triggers one Flyte workflow execution per trading day from start_date to end_date.
Respects US market holidays and weekends (Mon-Fri only).

Uses the Flyte Admin REST API (not pyflyte run --remote, which has module import
issues with Docker images).

Usage:
    # Backfill all 49 Phase 2 symbols from Jan 1 to today
    python scripts/backfill.py --start 2025-01-02 --end 2026-02-11

    # Backfill only the 39 new symbols (not in Phase 1)
    python scripts/backfill.py --start 2025-01-02 --end 2026-02-11 --only-new

    # Dry run (print dates only, don't execute)
    python scripts/backfill.py --start 2025-01-02 --end 2026-02-11 --dry-run

    # Custom symbols
    python scripts/backfill.py --start 2026-01-01 --end 2026-02-11 --symbols AVGO,ADBE,CRM

    # With custom batch size (concurrent executions)
    python scripts/backfill.py --start 2025-01-02 --end 2026-02-11 --batch-size 5

    # Use development domain
    python scripts/backfill.py --start 2025-01-02 --end 2026-02-11 --domain development
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# US Market Holidays (NYSE/NASDAQ closed)
US_HOLIDAYS = {
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas Day
    # 2026
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

PHASE2_SYMBOLS = [
    # Technology (10)
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    "AVGO", "ADBE", "CRM", "CSCO", "INTC",
    # Consumer Discretionary (5)
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    # Financials (6)
    "JPM", "V", "BAC", "GS", "MS", "BLK",
    # Healthcare (5)
    "JNJ", "UNH", "PFE", "ABT", "TMO",
    # Industrials (4)
    "CAT", "HON", "UPS", "RTX",
    # Consumer Staples (4)
    "PG", "KO", "PEP", "WMT",
    # Energy (3)
    "XOM", "CVX", "COP",
    # Communication Services (3)
    "NFLX", "DIS", "CMCSA",
    # Utilities (3)
    "NEE", "DUK", "SO",
    # Real Estate (3)
    "AMT", "PLD", "CCI",
    # Materials (3)
    "LIN", "APD", "SHW",
]

# Symbols only in Phase 2 (not in Phase 1)
NEW_SYMBOLS = [s for s in PHASE2_SYMBOLS if s not in PHASE1_SYMBOLS]


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """Get list of US trading days (Mon-Fri, excluding holidays)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    trading_days = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        # Skip weekends (5=Saturday, 6=Sunday)
        if current.weekday() < 5 and date_str not in US_HOLIDAYS:
            trading_days.append(date_str)
        current += timedelta(days=1)

    return trading_days


def get_latest_version(project: str, domain: str, flyte_host: str) -> str:
    """Get the latest registered version of the data_ingestion_workflow launch plan."""
    cmd = [
        "curl", "-s",
        f"http://{flyte_host}/api/v1/launch_plans/{project}/{domain}",
        "-G",
        "--data-urlencode", "limit=10",
        "--data-urlencode", "sort_by.key=created_at",
        "--data-urlencode", "sort_by.direction=DESCENDING",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return ""

    try:
        data = json.loads(result.stdout)
        for lp in data.get("launchPlans", []):
            name = lp.get("id", {}).get("name", "")
            if "data_ingestion" in name:
                return lp["id"]["version"]
    except (json.JSONDecodeError, KeyError):
        pass
    return ""


def trigger_workflow_api(
    date: str,
    symbols: list[str],
    domain: str,
    project: str,
    version: str,
    flyte_host: str,
) -> str:
    """Trigger a single WF1 execution via Flyte Admin REST API."""
    payload = {
        "project": project,
        "domain": domain,
        "spec": {
            "launchPlan": {
                "resourceType": "LAUNCH_PLAN",
                "project": project,
                "domain": domain,
                "name": "src.wf1_data_ingestion.workflow.data_ingestion_workflow",
                "version": version,
            },
            "inputs": {
                "literals": {
                    "symbols": {
                        "collection": {
                            "literals": [
                                {"scalar": {"primitive": {"stringValue": s}}}
                                for s in symbols
                            ]
                        }
                    },
                    "date": {"scalar": {"primitive": {"stringValue": date}}},
                }
            },
        },
    }

    cmd = [
        "curl", "-s", "-w", "\n%{http_code}",
        f"http://{flyte_host}/api/v1/executions",
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return f"FAILED: curl error: {result.stderr.strip()}"

    lines = result.stdout.strip().split("\n")
    http_code = lines[-1] if lines else "???"
    body = "\n".join(lines[:-1])

    if http_code == "200":
        try:
            resp = json.loads(body)
            exec_name = resp.get("id", {}).get("name", "unknown")
            return f"OK ({exec_name})"
        except json.JSONDecodeError:
            return f"OK (HTTP 200)"
    else:
        return f"FAILED: HTTP {http_code}: {body[:200]}"


def trigger_workflow_pyflyte(
    date: str,
    symbols: list[str],
    domain: str,
    project: str,
    image: str,
) -> str:
    """Trigger a single WF1 execution via pyflyte run --remote (legacy fallback)."""
    symbols_json = json.dumps(symbols)

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
    parser.add_argument("--flyte-host", default="localhost:8089",
                        help="Flyte Admin API host (default: localhost:8089)")
    parser.add_argument("--image", default="ghcr.io/biomechanoid-de/quant-trading-workflows:latest",
                        help="Docker image (only used with --use-pyflyte)")
    parser.add_argument("--version", default="",
                        help="Flyte version (auto-detected if empty)")
    parser.add_argument("--symbols", default="",
                        help="Comma-separated symbols (default: all 49 Phase 2)")
    parser.add_argument("--only-new", action="store_true",
                        help="Only backfill the 39 symbols not in Phase 1")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Number of concurrent executions per batch (default: 3)")
    parser.add_argument("--wait-between-batches", type=int, default=120,
                        help="Seconds to wait between batches (default: 120)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print trading days without executing")
    parser.add_argument("--use-pyflyte", action="store_true",
                        help="Use pyflyte run --remote instead of REST API (legacy)")
    args = parser.parse_args()

    # Determine symbols to backfill
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.only_new:
        symbols = NEW_SYMBOLS
    else:
        symbols = PHASE2_SYMBOLS

    trading_days = get_trading_days(args.start, args.end)

    print(f"=== WF1 Historical Backfill ===")
    print(f"Period:       {args.start} to {args.end}")
    print(f"Trading days: {len(trading_days)}")
    print(f"Domain:       {args.domain}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Symbols:      {len(symbols)} stocks")
    if args.only_new:
        print(f"Mode:         Only new Phase 2 symbols (not in Phase 1)")
    print(f"Method:       {'pyflyte run --remote' if args.use_pyflyte else 'Flyte Admin REST API'}")
    print()

    if args.dry_run:
        print("DRY RUN — Trading days:")
        for i, day in enumerate(trading_days, 1):
            weekday = datetime.strptime(day, "%Y-%m-%d").strftime("%a")
            print(f"  {i:3d}. {day} ({weekday})")
        print(f"\nTotal: {len(trading_days)} trading days x {len(symbols)} symbols")
        print(f"Symbols: {', '.join(symbols)}")
        return

    # Auto-detect version if not provided (API mode only)
    version = args.version
    if not args.use_pyflyte and not version:
        print("Auto-detecting latest registered version...", end=" ", flush=True)
        version = get_latest_version(args.project, args.domain, args.flyte_host)
        if version:
            print(f"{version[:12]}...")
        else:
            print("FAILED — please specify --version")
            sys.exit(1)

    # Process in batches to avoid overloading the Pi cluster
    total = len(trading_days)
    succeeded = 0
    failed = 0

    for batch_start in range(0, total, args.batch_size):
        batch = trading_days[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (total + args.batch_size - 1) // args.batch_size

        print(f"--- Batch {batch_num}/{total_batches} ---")

        for date in batch:
            weekday = datetime.strptime(date, "%Y-%m-%d").strftime("%a")
            print(f"  {date} ({weekday}) [{len(symbols)} symbols]... ", end="", flush=True)

            if args.use_pyflyte:
                result = trigger_workflow_pyflyte(
                    date, symbols, args.domain, args.project, args.image,
                )
            else:
                result = trigger_workflow_api(
                    date, symbols, args.domain, args.project, version, args.flyte_host,
                )

            print(result)
            if result.startswith("OK") or result.startswith("http"):
                succeeded += 1
            else:
                failed += 1

        # Wait between batches (unless it's the last batch)
        if batch_start + args.batch_size < total:
            print(f"  Waiting {args.wait_between_batches}s for cluster to process...\n")
            time.sleep(args.wait_between_batches)

    print(f"\n=== Backfill complete ===")
    print(f"Triggered: {succeeded} succeeded, {failed} failed (out of {total} trading days)")


if __name__ == "__main__":
    main()
