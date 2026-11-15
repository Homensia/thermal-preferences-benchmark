"""Regenerate the figures and tables of the B&E paper from the revision run outputs.

Usage (from the repository root, GPU hidden)::

    python -m analysis.paper_figures --run_dir rerun_2026-10-XX_v2 --out rerun_2026-10-XX_v2/PAPER_FIGURES
    python -m analysis.paper_figures --run_dir RUN --out NEW_DIR --only nn

See ``docs/PAPER_FIGURES.md`` for the mapping paper float -> item -> output files.
"""
