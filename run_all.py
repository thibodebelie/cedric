#!/usr/bin/env python3
"""Run multiple NEW_TSSP_M1_*.py scripts sequentially.

Usage:
  python run_all.py            # runs all matching scripts
  python run_all.py --dry-run  # only list matching scripts
  python run_all.py -p "NEW_TSSP_M1_*.py" --continue-on-error
"""
from __future__ import annotations
import sys
import subprocess
import glob
import argparse
from typing import List


def find_files(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))


def run_file(path: str) -> int:
    cmd = [sys.executable, path]
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple TSSP_NEWNEWM1_temp_*.py scripts sequentially")
    parser.add_argument('-p', '--pattern', default='TSSP_NEWNEWM1_temp_*.py', help='glob pattern to match scripts')
    parser.add_argument('--dry-run', action='store_true', help='only list matching files')
    parser.add_argument('--continue-on-error', action='store_true', help='continue running remaining scripts when one fails')
    args = parser.parse_args()

    files = find_files(args.pattern)
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        return 2

    print(f"Found {len(files)} file(s) matching '{args.pattern}':")
    for f in files:
        print(' -', f)

    if args.dry_run:
        return 0

    overall_rc = 0
    for f in files:
        print('\n' + '=' * 72)
        print(f"Running: {f}")
        print('=' * 72)
        rc = run_file(f)
        if rc != 0:
            print(f"{f} exited with code {rc}")
            overall_rc = rc if overall_rc == 0 else overall_rc
            if not args.continue_on_error:
                return rc

    print('\nAll scripts finished.')
    return overall_rc


if __name__ == '__main__':
    raise SystemExit(main())
