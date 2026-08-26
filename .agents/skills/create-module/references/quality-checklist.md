# Quality Checklist for New Profiling Modules

Use this 8-step verification checklist before declaring any new module complete.

---

### [ ] 1. Module Implementation (`src/fldataprofiler/modules/<name>.py`)
- [ ] Class implements `name`, `description`, and `run(feature_csv, label_csv, output_dir, join_key, targets) -> ModuleResult`.
- [ ] Returns `ModuleResult(report_dir=run_dir, artifacts=artifacts)`.
- [ ] Uses `_sample_rows(merged, max_rows=MAX_ROWS, random_state=RANDOM_STATE)` for `--full` compatibility.
- [ ] Uses `with ModuleProgress(self.name, total=4) as progress_bar:` with informative `.step(...)` messages.

### [ ] 2. Artifact Output Completeness (`reports/<name>/`)
- [ ] Generates valid `report.md` (Markdown tables, summary, and relative image links).
- [ ] Generates interactive `report.html` (self-contained CSS, KPI cards, table rows).
- [ ] Generates `summary.json` with metadata and top ranked results.
- [ ] Generates `*.csv` primary score table and breakdown details.
- [ ] Generates `*.png` visual distribution / heatmap chart with DPI $\ge 150$.

### [ ] 3. Configuration in `config.default.json`
- [ ] Default configuration entry added under `"<name>": { ... }`.
- [ ] All parameters have sane default fallback constants in the Python class.

### [ ] 4. Registry & Aliases in `src/fldataprofiler/registry.py`
- [ ] Class imported in `registry.py`.
- [ ] Standard snake_case name registered.
- [ ] Compact and common alias forms registered (e.g. no underscores, short abbreviation).

### [ ] 5. Dedicated Documentation (`docs/<name>.md`)
- [ ] Clear Markdown documentation explaining mathematical foundations & metrics.
- [ ] Explains configuration parameters.
- [ ] Shows CLI commands for both full and alias names.
- [ ] Lists and describes all 6 output artifacts.

### [ ] 6. Master Docs & Structure Sync
- [ ] Updated `docs/README.md` (Taxonomy section & Lookup Table).
- [ ] Updated `docs/.ai/STRUCTURE.md` (Docs list, modules count & list, tests list).
- [ ] Updated `docs/.ai/MODULES.md` (Catalog bullet point).
- [ ] Rebuilt agent files via `bash scripts/generate_agents_markdown.sh`.

### [ ] 7. Automated Unit Tests (`tests/test_<name>.py`)
- [ ] Tests registry lookup and alias resolution.
- [ ] Unit tests for core calculation functions on synthetic DataFrames.
- [ ] End-to-end `module.run(...)` verifying all artifacts are created on disk.
- [ ] Verified via `uv run pytest tests/test_<name>.py -q` (100% pass).

### [ ] 8. End-to-End CLI Sanity Check
- [ ] Executed `uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv --module <name> --limit 1000`.
- [ ] Verified CLI exits with code 0 and reports are written to `reports/<name>/`.
