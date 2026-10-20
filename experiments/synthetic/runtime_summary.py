"""Tables S5 and S6 (Appendix K.4) from the results of ``runtime_scalability.py``.

The first repetition (warm-up) is dropped, leaving 50.

* Table S5: mean and standard deviation (ddof = 0) of the wall-clock seconds
  of Omega_LP and Omega_GE.
* Table S6: for each repetition, log2 of the mean ratio of GE to LP interval
  lengths, log2(K^-1 sum_k l_k,GE / l_k,LP); then mean and standard deviation
  (ddof = 1) over the repetitions.  This is how the numbers in Table S6 were
  computed.

Example:
    python synthetic/runtime_summary.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import RESULTS_ROOT
from exputils.plotting import generate_stats_markdown_table


def load_results(results_dir):
    results = {}
    for path in Path(results_dir).glob("runtime_K.*.pkl"):
        with open(path, "rb") as f:
            res = pickle.load(f)
        results[res["K"]] = res
    return dict(sorted(results.items()))


def log2_length_ratio(res, drop_first=True):
    lengths = {key: np.diff(box, axis=-1)[..., 0] for key, box in res["boxes"].items()}
    ratio = np.log2((lengths["ge"] / np.maximum(lengths["lp"], 1e-12)).mean(axis=1))
    return ratio[1:] if drop_first else ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT / "runtime"))
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        sys.exit(f"No result files in {args.results_dir}; run runtime_scalability.py first.")
    Ks = list(results)
    headers = ["K"] + [str(K) for K in Ks]

    rows, names = [], []
    for key, label in [("lp", "LP"), ("ge", "GE")]:
        times = [results[K]["runtime"][key][1:] for K in Ks]
        rows += [[t.mean() for t in times], [t.std() for t in times]]
        names += [f"{label} mean", f"{label} std"]
    solvers = {results[K]["lp_solver"] for K in Ks}
    print(f"Table S5: runtime in seconds per run (LP solver: {', '.join(sorted(solvers))})\n")
    print(generate_stats_markdown_table(np.array(rows), None, headers, names, digits=5))

    ratios = [log2_length_ratio(results[K]) for K in Ks]
    table = np.array([[r.mean() for r in ratios], [r.std(ddof=1) for r in ratios]])
    print("\nTable S6: log2 length ratio of GE to LP\n")
    print(generate_stats_markdown_table(table, None, headers, ["mean", "std"], digits=2))


if __name__ == "__main__":
    main()
