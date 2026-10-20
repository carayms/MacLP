"""Shared helpers for the MaC-LP experiments.

This package is independent of the ``maclp`` library: it contains the
reference implementations that were used to produce the results in the
paper (sampling, confidence intervals, LP / GE regions, weighted conformal
prediction and plotting).  Scripts in the experiment folders import it via

    from exputils import sampling, intervals, regions, conformal
"""

from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = EXPERIMENTS_ROOT / "results"
FIGURES_ROOT = EXPERIMENTS_ROOT / "figures"
DATA_ROOT = EXPERIMENTS_ROOT / "data"
