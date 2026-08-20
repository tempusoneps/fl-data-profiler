from __future__ import annotations

import argparse
import json
from pathlib import Path

from fldataprofiler.feature_pruner import PruneConfig, load_scores, prune_features
from fldataprofiler.registry import get_module, list_modules
from fldataprofiler.utils import (
    _full_row_mode,
    _input_row_limit,
    _is_supported_input_path,
    _supported_input_formats_message,
    load_dataframe,
)


def _register_prune_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "prune",
        help="Filter and export a clean feature dataset (drops collinear, null, and low-variance features)",
    )
    parser.add_argument(
        "feature_path",
        type=Path,
        help="Path to feature dataset (.parquet or .csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for pruned output file (default: datasets/selected_feature.<ext>)",
    )
    parser.add_argument(
        "--max-corr",
        type=float,
        default=0.85,
        help="Maximum allowable pairwise correlation between features (default: 0.85)",
    )
    parser.add_argument(
        "--corr-method",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Correlation method (default: pearson)",
    )
    parser.add_argument(
        "--max-null",
        type=float,
        default=0.20,
        help="Maximum fraction of null/NaN values allowed for a feature (default: 0.20)",
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=0.0,
        help="Minimum variance threshold to drop constant features (default: 0.0)",
    )
    parser.add_argument(
        "--scores-file",
        type=Path,
        help="Optional path to feature scores CSV file (e.g. from a profiling report) for tie-breaking",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        help="Name of importance/score column in scores file (auto-detected if omitted)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Keep at most top K features after pruning",
    )
    parser.add_argument(
        "--keep-col",
        action="append",
        dest="keep_cols",
        help="Column to preserve unconditionally (can be passed multiple times)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to write audit summary JSON (default: reports/prune_summary.json)",
    )
    return parser


def _run_prune_command(args: argparse.Namespace) -> int:
    df_raw = load_dataframe(args.feature_path)

    scores = None
    if args.scores_file:
        scores = load_scores(args.scores_file, score_col=args.score_col)

    config = PruneConfig(
        max_corr=args.max_corr,
        corr_method=args.corr_method,
        max_null=args.max_null,
        min_variance=args.min_variance,
        top_k=args.top_k,
        keep_cols=args.keep_cols,
    )

    result = prune_features(df_raw, config=config, scores=scores)

    ext = (
        args.feature_path.suffix
        if args.feature_path.suffix.lower() in (".parquet", ".csv")
        else ".parquet"
    )
    if args.output:
        output_path = args.output
    else:
        output_path = Path("datasets") / f"selected_feature{ext}"

    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        result.df_selected.to_parquet(output_path, index=False)
    else:
        result.df_selected.to_csv(output_path, index=False)

    summary_path = args.summary_json or (Path("reports") / "prune_summary.json")
    if summary_path.parent != Path(""):
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "input_path": str(args.feature_path),
        "output_path": str(output_path),
        **result.summary,
        "dropped_by_reason": result.dropped_by_reason,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Pruned dataset written to: {output_path}")
    print(
        f"Features: {result.summary['initial_features_count']} -> {result.summary['retained_features_count']} retained "
        f"({result.summary['dropped_features_count']} dropped)"
    )
    print(f"Audit summary saved to: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fldataprofiler",
        description="Create reports that profile relationships between feature.csv and label.csv.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _register_prune_parser(subparsers)

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
    fit.add_argument(
        "--limit",
        type=_positive_int,
        help="Limit both feature and label inputs to the first N rows before generating the report.",
    )
    fit.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Disable internal row subsampling to analyze all rows (may increase compute time for ML modules).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fit":
        _validate_input_path(parser, "feature_csv", args.feature_csv)
        _validate_input_path(parser, "label_csv", args.label_csv)
        module = get_module(args.module)
        with _input_row_limit(args.limit), _full_row_mode(args.full):
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

    if args.command == "prune":
        _validate_input_path(parser, "feature_path", args.feature_path)
        return _run_prune_command(args)

    raise ValueError(f"Unsupported command: {args.command}")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--limit must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer")
    return parsed


def _validate_input_path(parser: argparse.ArgumentParser, name: str, path: Path) -> None:
    if not _is_supported_input_path(path):
        parser.error(f"{name} must be a {_supported_input_formats_message()} file: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
