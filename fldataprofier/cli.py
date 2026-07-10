from __future__ import annotations

import argparse
from pathlib import Path

from fldataprofier.registry import get_module, list_modules
from fldataprofier.utils import (
    _is_supported_input_path,
    _supported_input_formats_message,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fldataprofier",
        description="Create reports that profile relationships between feature.csv and label.csv.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Generate a profiling report")
    fit.add_argument("feature_csv", type=Path, help="Path to feature.csv")
    fit.add_argument("label_csv", type=Path, help="Path to label.csv")
    fit.add_argument(
        "--module",
        default="statistics",
        choices=list_modules(),
        help="Report module to run",
    )
    fit.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated report artifacts",
    )
    fit.add_argument(
        "--join-key",
        help="Optional column name used to join feature and label rows. Defaults to common columns or row index.",
    )
    fit.add_argument(
        "--target",
        action="append",
        help="Label column to focus on. Can be passed multiple times. Defaults to all label columns.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fit":
        _validate_input_path(parser, "feature_csv", args.feature_csv)
        _validate_input_path(parser, "label_csv", args.label_csv)
        module = get_module(args.module)
        result = module.run(
            feature_csv=args.feature_csv,
            label_csv=args.label_csv,
            output_dir=args.output_dir,
            join_key=args.join_key,
            targets=args.target,
        )
        print(f"Report written to: {result.report_dir}")
        for artifact in result.artifacts:
            print(f"- {artifact}")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


def _validate_input_path(parser: argparse.ArgumentParser, name: str, path: Path) -> None:
    if not _is_supported_input_path(path):
        parser.error(
            f"{name} must be a {_supported_input_formats_message()} file: {path}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
