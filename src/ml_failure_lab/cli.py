import argparse
import csv
import json
from pathlib import Path

from ml_failure_lab.registry import CASES
from ml_failure_lab.split_check import check_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and inspect small ML evaluation failures.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List runnable cases")
    split_parser = commands.add_parser("check-split", help="Inspect your train/test CSV files")
    split_parser.add_argument("train", type=Path)
    split_parser.add_argument("test", type=Path)
    split_parser.add_argument("--id-column", required=True)
    split_parser.add_argument("--features", nargs="+", default=[])
    split_parser.add_argument("--require-new-ids", action="store_true")
    split_parser.add_argument("--json", action="store_true")
    run_parser = commands.add_parser("run", help="Compare the incorrect and corrected approaches")
    run_parser.add_argument("case", choices=CASES)
    run_parser.add_argument("--json", action="store_true", help="Print structured measurements")
    verify_parser = commands.add_parser("verify", help="Check a case's declared requirement")
    verify_parser.add_argument("case", choices=CASES)
    verify_parser.add_argument("--approach", choices=["bad", "fixed"], required=True)
    args = parser.parse_args(argv)
    if args.command == "check-split":
        try:
            result = check_split(
                args.train, args.test, args.id_column, args.features, args.require_new_ids
            )
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            parser.exit(2, f"Invalid split input: {error}\n")
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"{result['status']}: {result['requirement']}")
            print(f"Shared IDs: {result['shared_ids']}")
            print(f"Test rows with shared IDs: {result['affected_test_rows']}")
            overlap = result["exact_feature_overlap"]
            if overlap is not None:
                print(f"Exact shared feature rows: {overlap['distinct_shared_rows']}")
                print(f"Test rows with matching features: {overlap['affected_test_rows']}")
            print(result["limitation"])
        return int(result["status"] == "FAIL")
    if args.command == "list":
        for name in CASES:
            print(name)
        return 0
    case = CASES[args.case]
    if args.command == "verify":
        try:
            case.verify(args.approach)
        except ValueError as error:
            print("FAIL: " + str(error))
            return 1
        print("PASS: " + args.case + " meets its declared requirement.")
        return 0
    result = case.run()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["question"])
        print()
        for row in result["measurements"]:
            print(" | ".join(f"{key}={value}" for key, value in row.items()))
        print("\nRequirement: " + result["requirement"])
        print(result["interpretation"])
        print("\nLimit: " + result["limitation"])
    return 0
