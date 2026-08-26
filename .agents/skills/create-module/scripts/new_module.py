#!/usr/bin/env python3
"""
Helper script to scaffold a new profiling module in fl-data-profiling.

Usage:
    python .agents/skills/create-module/scripts/new_module.py \
        --name probability_entropy \
        --class-name ProbabilityEntropyModule \
        --description "Quantile Shannon Entropy & Information Gain Profiling" \
        --aliases "entropy,prob_entropy"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _to_pascal_case(snake_str: str) -> str:
    components = snake_str.split("_")
    return "".join(x.capitalize() for x in components) + ("Module" if not snake_str.endswith("module") else "")


def scaffold_module(
    name: str,
    class_name: str | None,
    description: str,
    aliases: list[str],
    repo_root: Path,
) -> None:
    module_name = name.lower().strip()
    if not class_name:
        class_name = _to_pascal_case(module_name)

    all_aliases = [module_name, module_name.replace("_", "")] + [a.strip() for a in aliases if a.strip()]
    unique_aliases = list(dict.fromkeys(all_aliases))

    print(f"Scaffolding module '{module_name}' ({class_name})...")

    # 1. Create src/fldataprofiler/modules/<name>.py
    module_file = repo_root / "src" / "fldataprofiler" / "modules" / f"{module_name}.py"
    if not module_file.exists():
        module_code = f'''from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fldataprofiler.config import get_module_config
from fldataprofiler.modules.base import ModuleResult
from fldataprofiler.modules.progress import ModuleProgress
from fldataprofiler.modules.statistics import DatasetShape
from fldataprofiler.utils import (
    _date_columns,
    _format_duration,
    _markdown_table,
    _merge_inputs,
    _numeric_feature_columns,
    _numeric_series,
    _read_table_with_date_index,
    _sample_rows,
    _select_targets,
    _write_csv,
    _write_json,
)

MAX_ROWS = 50_000
MIN_NON_NULL = 10
RANDOM_STATE = 42
DEFAULT_TOP_FEATURES = 25


@dataclass
class {class_name.replace("Module", "")}Config:
    top_features: int = DEFAULT_TOP_FEATURES


@dataclass(frozen=True)
class {class_name.replace("Module", "")}RunMetadata:
    module: str
    created_at: str
    execution_time: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    features_count: int
    targets: list[str]
    model_rows: int


def _compute_feature_scores(
    feature_series: pd.Series,
    target_series: pd.Series,
    feature_name: str,
    target_name: str,
) -> list[dict[str, object]]:
    """Compute module-specific scoring metrics for a single feature vs target."""
    clean_x = _numeric_series(feature_series)
    valid_mask = clean_x.notna() & target_series.notna()
    x_val = clean_x[valid_mask]
    t_val = target_series[valid_mask]

    n_samples = len(x_val)
    if n_samples < 20 or t_val.nunique(dropna=True) < 2 or x_val.nunique(dropna=True) < 2:
        return []

    # Example baseline calculation
    score_val = float(np.corrcoef(x_val, pd.factorize(t_val)[0])[0, 1]) if np.std(x_val) > 0 else 0.0
    if np.isnan(score_val):
        score_val = 0.0

    return [{{
        "feature": feature_name,
        "target": target_name,
        "samples": n_samples,
        "score_primary": float(abs(score_val)),
        "score_raw": float(score_val),
    }}]


def _plot_visualizations(
    scores_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Generate high-resolution visualization chart for top scored features."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if scores_df.empty:
        ax.text(0.5, 0.5, "No scoring data available", ha="center", va="center")
        ax.axis("off")
    else:
        top_df = scores_df.sort_values("score_primary", ascending=True).tail(top_n)
        ax.barh(top_df["feature"], top_df["score_primary"], color="#3b82f6", edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Primary Score", fontsize=10)
        ax.set_title(f"Top {{len(top_df)}} Features by Score ({module_name})", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _generate_markdown_report(
    metadata: {class_name.replace("Module", "")}RunMetadata,
    scores_df: pd.DataFrame,
    image_rel_path: str,
) -> str:
    md: list[str] = [
        f"# {description} (`{{metadata.module}}`)",
        "",
        f"- **Generated At**: {{metadata.created_at}}",
        f"- **Execution Time**: {{metadata.execution_time}}",
        f"- **Merged Rows**: {{metadata.model_rows:,}}",
        f"- **Features Evaluated**: {{metadata.features_count}}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    if scores_df.empty:
        md.append("No features evaluated.")
        return "\\n".join(md)

    top_feat = scores_df.iloc[0]
    md.extend([
        f"- **Top Scored Feature**: `{{top_feat['feature']}}` for target `{{top_feat['target']}}` (Score: **{{top_feat['score_primary']:.4f}}**).",
        "",
        "---",
        "",
        "## Feature Scoring Leaderboard",
        "",
    ])

    sub_df = scores_df.head(25).copy()
    md.append(_markdown_table(sub_df))
    md.extend([
        "",
        "---",
        "",
        "## Visual Distribution",
        "",
        f"![{module_name} Chart]({{image_rel_path}})",
        "",
    ])

    return "\\n".join(md)


def _generate_html_report(
    metadata: {class_name.replace("Module", "")}RunMetadata,
    scores_df: pd.DataFrame,
    image_rel_path: str,
) -> str:
    table_rows = []
    for _, row in scores_df.head(50).iterrows():
        table_rows.append(f"""
        <tr>
            <td><strong>{{row['feature']}}</strong></td>
            <td><span class="badge badge-info">{{row['target']}}</span></td>
            <td style="text-align: right;">{{row['samples']:,}}</td>
            <td style="text-align: right; font-weight: bold; color: #2563eb;">{{row['score_primary']:.4f}}</td>
        </tr>
        """)

    tbody = "\\n".join(table_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{description} - {{metadata.module}}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #1e293b; padding: 24px; margin: 0; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #0f172a; color: white; padding: 24px; border-radius: 8px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
        .card {{ background: white; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
        th {{ background: #f1f5f9; }}
        .badge {{ padding: 3px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; background: #e0e7ff; color: #3730a3; }}
        .img-container {{ text-align: center; margin-top: 20px; }}
        .img-container img {{ max-width: 100%; border-radius: 6px; border: 1px solid #e2e8f0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{description}</h1>
            <p>Module: <code>{{metadata.module}}</code> | Rows: {{metadata.model_rows:,}} | Features: {{metadata.features_count}}</p>
        </div>
        <div class="card" style="padding: 0; overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Target</th>
                        <th style="text-align: right;">Samples</th>
                        <th style="text-align: right;">Primary Score</th>
                    </tr>
                </thead>
                <tbody>
                    {{tbody}}
                </tbody>
            </table>
        </div>
        <div class="card img-container">
            <img src="{{image_rel_path}}" alt="{module_name} Chart">
        </div>
    </div>
</body>
</html>
"""


class {class_name}:
    name: str = "{module_name}"
    description: str = "{description}"

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        start_time = time.time()
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        features_df = _read_table_with_date_index(feature_csv)
        labels_df = _read_table_with_date_index(label_csv)

        feature_shape = DatasetShape(rows=len(features_df), columns=len(features_df.columns))
        label_shape = DatasetShape(rows=len(labels_df), columns=len(labels_df.columns))

        merged, feature_cols, label_cols, join_strategy = _merge_inputs(
            features_df, labels_df, join_key
        )
        merged_shape = DatasetShape(rows=len(merged), columns=len(merged.columns))

        sampled_df = _sample_rows(merged, max_rows=MAX_ROWS, random_state=RANDOM_STATE)
        selected_targets = _select_targets(label_cols, targets)

        raw_cfg = get_module_config("{module_name}")
        cfg = {class_name.replace("Module", "")}Config(
            top_features=int(raw_cfg.get("top_features", DEFAULT_TOP_FEATURES)),
        )

        numeric_feature_cols = _numeric_feature_columns(sampled_df, feature_cols)

        with ModuleProgress(self.name, total=4) as progress_bar:
            progress_bar.step("Evaluating features")
            all_scores: list[dict[str, object]] = []
            for target in selected_targets:
                target_series = sampled_df[target]
                if target_series.nunique(dropna=True) < 2:
                    continue
                for col in numeric_feature_cols:
                    scores = _compute_feature_scores(
                        feature_series=sampled_df[col],
                        target_series=target_series,
                        feature_name=col,
                        target_name=target,
                    )
                    all_scores.extend(scores)

            progress_bar.step("Ranking scores")
            if all_scores:
                scores_df = pd.DataFrame(all_scores).sort_values("score_primary", ascending=False).reset_index(drop=True)
            else:
                scores_df = pd.DataFrame(columns=["feature", "target", "samples", "score_primary", "score_raw"])

            progress_bar.step("Rendering charts")
            img_path = run_dir / "{module_name}_chart.png"
            _plot_visualizations(scores_df, img_path, top_n=cfg.top_features)

            progress_bar.step("Writing reports and summaries")
            scores_path = _write_csv(run_dir / "feature_scores.csv", scores_df)

            duration = _format_duration(time.time() - start_time)

            metadata = {class_name.replace("Module", "")}RunMetadata(
                module=self.name,
                created_at=now,
                execution_time=duration,
                feature_csv=str(feature_csv),
                label_csv=str(label_csv),
                join_strategy=join_strategy,
                feature_shape=feature_shape,
                label_shape=label_shape,
                merged_shape=merged_shape,
                features_count=len(numeric_feature_cols),
                targets=selected_targets,
                model_rows=len(sampled_df),
            )

            summary_payload = {{
                "metadata": asdict(metadata),
                "top_features": scores_df.head(cfg.top_features).to_dict(orient="records") if not scores_df.empty else [],
            }}
            summary_path = _write_json(run_dir / "summary.json", summary_payload)

            report_md_content = _generate_markdown_report(metadata, scores_df, "{module_name}_chart.png")
            report_md_path = run_dir / "report.md"
            report_md_path.write_text(report_md_content, encoding="utf-8")

            report_html_content = _generate_html_report(metadata, scores_df, "{module_name}_chart.png")
            report_html_path = run_dir / "report.html"
            report_html_path.write_text(report_html_content, encoding="utf-8")

        artifacts = [
            report_md_path,
            report_html_path,
            summary_path,
            scores_path,
            img_path,
        ]

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)
'''
        module_file.write_text(module_code, encoding="utf-8")
        print(f"  -> Created {module_file}")
    else:
        print(f"  -> Exists: {module_file}")

    # 2. Create tests/test_<name>.py
    test_file = repo_root / "tests" / f"test_{module_name}.py"
    if not test_file.exists():
        test_code = f'''from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.{module_name} import {class_name}, _compute_feature_scores
from fldataprofiler.registry import get_module


def make_test_datasets(base_dir: Path, rows: int = 200, seed: int = 42) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")
    x1 = np.linspace(-2.0, 2.0, rows)
    noise = rng.standard_normal(rows)
    target = (x1 > 0.0).astype(int)

    f_path = base_dir / "feature.csv"
    l_path = base_dir / "label.csv"

    pd.DataFrame({{"Date": dates, "x1": x1, "noise": noise}}).to_csv(f_path, index=False)
    pd.DataFrame({{"Date": dates, "target": target}}).to_csv(l_path, index=False)
    return f_path, l_path


class {class_name}Tests(unittest.TestCase):
    def test_registry_lookup(self) -> None:
        mod = get_module("{module_name}")
        self.assertIsInstance(mod, {class_name})

    def test_feature_scoring(self) -> None:
        x = pd.Series(np.linspace(1, 100, 100))
        target = pd.Series([0] * 50 + [1] * 50)
        scores = _compute_feature_scores(x, target, "x1", "target")
        self.assertTrue(len(scores) > 0)
        self.assertGreater(scores[0]["score_primary"], 0.0)

    def test_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            f_path, l_path = make_test_datasets(base, rows=150)
            out_dir = base / "reports"

            module = {class_name}()
            result = module.run(
                feature_csv=f_path,
                label_csv=l_path,
                output_dir=out_dir,
            )

            report_dir = out_dir / "{module_name}"
            self.assertEqual(result.report_dir, report_dir)
            self.assertTrue((report_dir / "report.md").exists())
            self.assertTrue((report_dir / "report.html").exists())
            self.assertTrue((report_dir / "summary.json").exists())
            self.assertTrue((report_dir / "feature_scores.csv").exists())
            self.assertTrue((report_dir / "{module_name}_chart.png").exists())


if __name__ == "__main__":
    unittest.main()
'''
        test_file.write_text(test_code, encoding="utf-8")
        print(f"  -> Created {test_file}")
    else:
        print(f"  -> Exists: {test_file}")

    # 3. Create docs/<name>.md
    doc_file = repo_root / "docs" / f"{module_name}.md"
    if not doc_file.exists():
        doc_code = f'''# {description} (`{module_name}`)

Module `{module_name}` cung cấp khả năng phân tích và đánh giá các đặc trưng (`features`) đối với nhãn (`labels`).

---

## 1. Nguyên lý Toán học & Chỉ số Đo lường

Mô tả nguyên lý toán học và các công thức tính toán cốt lõi.

---

## 2. Cấu hình Tham số (`config.default.json`)

```json
"{module_name}": {{
  "top_features": 25
}}
```

---

## 3. Hướng dẫn Sử dụng CLI

```bash
# Chạy với tên lệnh
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv \\
  --module {module_name} \\
  --target allow_entry
```

---

## 4. Danh sách Kết quả Đầu ra (Artifacts)

Tất cả báo cáo được lưu tại `reports/{module_name}/`:

1. `feature_scores.csv`: Bảng điểm đánh giá các đặc trưng.
2. `{module_name}_chart.png`: Biểu đồ trực quan hóa.
3. `summary.json`: Metadata tổng hợp và top kết quả.
4. `report.md`: Báo cáo Markdown.
5. `report.html`: Báo cáo HTML tương tác.
'''
        doc_file.write_text(doc_code, encoding="utf-8")
        print(f"  -> Created {doc_file}")
    else:
        print(f"  -> Exists: {doc_file}")

    # 4. Update config.default.json
    cfg_path = repo_root / "src" / "fldataprofiler" / "config.default.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        if module_name not in cfg_data:
            cfg_data[module_name] = {"top_features": 25}
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)
            print(f"  -> Updated {cfg_path} with default config.")

    # 5. Update registry.py
    registry_path = repo_root / "src" / "fldataprofiler" / "registry.py"
    if registry_path.exists():
        reg_text = registry_path.read_text(encoding="utf-8")
        import_stmt = f"from fldataprofiler.modules.{module_name} import {class_name}"
        if import_stmt not in reg_text:
            # Insert import before _MODULES
            reg_text = re.sub(
                r"(from fldataprofiler\.modules\.[^\n]+\n)(_MODULES)",
                r"\1" + import_stmt + r"\n\2",
                reg_text,
                count=1,
            )
            # Insert aliases into _MODULES dict
            alias_lines = "".join(f'    "{alias}": {class_name},\n' for alias in unique_aliases)
            reg_text = re.sub(
                r"(_MODULES:\s*dict\[str,\s*type\[ProfilingModule\]\]\s*=\s*\{)",
                r"\1\n" + alias_lines,
                reg_text,
                count=1,
            )
            registry_path.write_text(reg_text, encoding="utf-8")
            print(f"  -> Registered {class_name} and aliases {unique_aliases} in {registry_path}.")

    print(f"\nScaffolding complete for module '{module_name}'.")
    print(f"Next steps:")
    print(f"  1. Implement specific math & scoring in {module_file}")
    print(f"  2. Update documentation in {doc_file}")
    print(f"  3. Update docs/README.md, docs/.ai/STRUCTURE.md, docs/.ai/MODULES.md")
    print(f"  4. Run `bash scripts/generate_agents_markdown.sh`")
    print(f"  5. Verify tests via `uv run pytest tests/test_{module_name}.py -v`")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new profiling module.")
    parser.add_argument("--name", required=True, help="Snake_case module name (e.g. probability_entropy)")
    parser.add_argument("--class-name", help="PascalCase class name (e.g. ProbabilityEntropyModule)")
    parser.add_argument("--description", default="New Profiling Module", help="Short description")
    parser.add_argument("--aliases", default="", help="Comma-separated aliases (e.g. 'entropy,prob_entropy')")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]

    scaffold_module(
        name=args.name,
        class_name=args.class_name,
        description=args.description,
        aliases=aliases,
        repo_root=repo_root,
    )


if __name__ == "__main__":
    main()
