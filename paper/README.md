# Paper: Geometry-Conditioned, Repair-Guaranteed Zero-Cost NAS for Unseen Data

Self-contained technical report on the NAS Unseen-Data 2026 submission.

## Build

The source is a single self-contained `main.tex` that compiles with a vanilla
TeX Live / MacTeX install (no external `.sty` files, no special conference
class). The bibliography is inlined via `thebibliography`, so a single pass
chain suffices:

```bash
pdflatex main.tex
pdflatex main.tex      # second pass resolves \ref / \cite numbering
```

Output: `main.pdf`.

## Structure

- **§2 What Is Original Here** — explicitly separates reused prior art
  (AZ-NAS proxies, aging evolution, standard blocks) from the claimed
  contributions, leading with the system-level synthesis.
- **§3 Family Inference** — geometry → hard constraints (8 families).
- **§5 Spatial-Aware Repair** — the ceil/floor down-sampling subtlety and the
  11 ordered repair rules.
- **§6 Hardware-Aware Memory** — training-memory estimator + GPU-tied budget.
- **§7 AZ-NAS Adaptation** — gradient-retaining feature hook + graceful
  per-proxy degradation.
- **§8 Aging-Evolution Search**, **§9 Anytime Budgeting**,
  **§10 Three-Zone Early Stopping**.
- **§11 Robustness Engineering**, **§12 Determinism**.
- **§13 Experimental Protocol** — Table 4 (per-dataset accuracy) has
  placeholder cells to be filled from the official evaluation. No fabricated
  numbers are reported.

## Note on results

Per-dataset accuracy (Table 4) is intentionally left as `--` placeholders.
Fill these from the pipeline's own `predictions/<codename>_report.json` and the
official scorer; do not commit fabricated numbers.
