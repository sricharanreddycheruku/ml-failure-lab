import argparse
import json

from ml_failure_lab.registry import CASES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and inspect small ML evaluation failures.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List runnable cases")
    run_parser = commands.add_parser("run", help="Compare the incorrect and corrected approaches")
    run_parser.add_argument("case", choices=CASES)
    run_parser.add_argument("--json", action="store_true", help="Print structured measurements")
    verify_parser = commands.add_parser("verify", help="Check a case's declared requirement")
    verify_parser.add_argument("case", choices=CASES)
    verify_parser.add_argument("--approach", choices=["bad", "fixed"], required=True)
    args = parser.parse_args(argv)
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
