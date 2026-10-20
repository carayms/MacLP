"""Table S7 (Appendix K.5): target label proportions under the Dirichlet shift
design of the benchmark experiments.

For a in {-3, ..., 3}, draw 100 vectors q ~ Dir(10^a * 1_K) with K = 10, sort
each draw in increasing order, and report the mean and standard deviation
(ddof = 1) of each sorted position.

Example:
    python synthetic/dirichlet_shift_table.py
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--n-repts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=199)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    exponents = np.arange(-3, 4)
    draws = np.array([[np.sort(rng.dirichlet(10.0 ** a * np.ones(args.K))) for _ in range(args.n_repts)]
                      for a in exponents])                      # (n_a, n_repts, K)
    means, stds = draws.mean(axis=1), draws.std(axis=1, ddof=1)

    print("| a | | " + " | ".join(str(k) for k in range(args.K)) + " |")
    print("|" + " --- |" * (args.K + 2))
    for i, a in enumerate(exponents):
        print(f"| {a} | mean | " + " | ".join(f"{v:.3f}" for v in means[i]) + " |")
        print("| | std | " + " | ".join(f"{v:.3f}" for v in stds[i]) + " |")


if __name__ == "__main__":
    main()
