# From matrix inversion to constraints:

`maclp` computes confidence regions for importance weights under label shift. From the confusion matrix of a classifier on labeled source data and its predicted-label counts on target data, it returns

* `BBSE`: the point estimate of the weights ω_k = q(k) / p(k);
* `LP`: the box Ω_LP, the tightest axis-aligned box around the confidence region Ω, computed by linear programs in C++;
* `GE`: the interval Gaussian elimination box Ω_GE, a simpler baseline.

For the method and its guarantees, see the paper:

> Mushan Li, Kihyun Han and Yanyuan Ma. *From Matrix Inversion to Constraints: Provably Tighter Confidence Regions for Importance Weights in Label Shift.* NeurIPS 2026.

The code for the experiments in the paper is in `experiments/`.

## Installation

The LP solver is written in C++ (HiGHS, called through pybind11 and parallelized with OpenMP), so the package needs the HiGHS library and a C++ compiler at build time. The simplest way to get HiGHS is conda-forge:

```bash
# run from the root folder of this repository (after cloning or downloading it)
conda create -n maclp -c conda-forge python=3.12 highs pybind11 numpy scipy scikit-learn numba
conda activate maclp
conda install -c conda-forge llvm-openmp      # macOS only
pip install .
```

`setup.py` looks for the HiGHS headers and library in `$HIGHS_PREFIX`, else in the active conda environment (`$CONDA_PREFIX`), else in the Python prefix. If you built HiGHS yourself, point to its install folder:

```bash
HIGHS_PREFIX=/path/to/highs/install pip install .
```

The library path is recorded in the compiled module, so `LD_LIBRARY_PATH` does not need to be set afterwards. Check the installation from a folder other than the repository root (otherwise Python imports the uncompiled source folder):

```bash
cd /tmp && python -c "import maclp; print(maclp.LP)"
```

> **Note (Linux).** Do not import `highspy` (the HiGHS Python package from PyPI) in the same Python process as `maclp`. Both ship a library called `libhighs.so.1`, and whichever is loaded first is used by both, which can fail with an undefined-symbol error. `maclp` itself does not need `highspy`.

## Quick start

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import maclp

rng = np.random.default_rng(0)
K = 3
mus = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
Sigma = np.eye(2)
p = np.array([0.5, 0.3, 0.2])          # source label proportions
q = np.array([0.2, 0.3, 0.5])          # target label proportions (unknown in practice)
M = N = 2000

XY_src = maclp.toy_data_generating(mus, Sigma, M, p, rng=rng)   # labeled source sample
XY_tgt = maclp.toy_data_generating(mus, Sigma, N, q, rng=rng)   # target sample (labels unused)

# any classifier g(x); here the Bayes classifier of the source distribution
pred_src = maclp.bayes_classifier(XY_src[:, :-1], mus, Sigma, p, out_type="hard")
pred_tgt = maclp.bayes_classifier(XY_tgt[:, :-1], mus, Sigma, p, out_type="hard")

C_mtx = confusion_matrix(XY_src[:, -1].astype(int), pred_src, labels=np.arange(K)).T  # C[i, j] = #{pred i, true j}
q_dist = maclp.q_samp_dist(pred_tgt, K)                                              # #{target predicted as i}

w_hat = maclp.BBSE().fit(C_mtx, q_dist, M, N)
lp = maclp.LP()
box_lp = lp.fit_from_data(C_mtx, q_dist, M, N, delta=0.01)
box_ge = maclp.GE().fit_from_data(C_mtx, q_dist, M, N, delta=0.01)

print("true weights q/p :", np.round(q / p, 3))
print("BBSE estimate    :", np.round(w_hat, 3))
print("Omega_LP (lower, upper):\n", np.round(box_lp, 3))
print("Omega_GE (lower, upper):\n", np.round(box_ge, 3))
print("Omega nonempty   :", lp.is_feasible_)
```

Output:

```
true weights q/p : [0.4 1.  2.5]
BBSE estimate    : [0.419 1.056 2.448]
Omega_LP (lower, upper):
 [[0.    0.79 ]
 [0.54  1.571]
 [1.594 3.466]]
Omega_GE (lower, upper):
 [[0.    0.82 ]
 [0.469 1.613]
 [1.524 3.696]]
Omega nonempty   : True
```

Each row of a box is the interval [lower, upper] for one weight ω_k.

## Using your own classifier

All estimators work with two count summaries of a fitted classifier g:

| Input | Meaning |
| --- | --- |
| `C_mtx` (K × K) | `C_mtx[i, j]` = number of labeled source points with prediction i and true label j. With scikit-learn this is `confusion_matrix(y_true, y_pred, labels=range(K)).T` (note the transpose); `maclp.utils.confusion_mtx(y_true, y_pred, K)` gives the same. |
| `q_dist` (K,) | `q_dist[i]` = number of target points predicted as class i, e.g. `maclp.q_samp_dist(g_pred_target, K)`. |
| `M`, `N` | Number of source points used for `C_mtx` and of target points used for `q_dist`. |
| `delta` | Total error level of the intervals. Each of the K² entries of C and the K entries of q uses level δ / (K² + K) (Bonferroni), so all intervals hold jointly with probability at least 1 − δ. |

Labels are 0, …, K−1. Use source points that were not used to train g, so that `C_mtx` is an unbiased estimate of C\*.

## API

| Class / function | Description |
| --- | --- |
| `maclp.BBSE().fit(C_mtx, q_dist, M, N)` | BBSE point estimate pinv(C/M) (q/N), projected onto ω ≥ 0. Returns an array of shape (K,). |
| `maclp.LP(bound_single=(0, 500)).fit_from_data(C_mtx, q_dist, M, N, delta)` | Ω_LP. Returns an array of shape (K, 2) with the lower and upper bound of each weight. Upper bounds are capped at `bound_single[1]`. |
| `maclp.LP(...).fit(C_low, C_up, q_low, q_up)` | Ω_LP from interval bounds you computed yourself (C as proportions, q as proportions). |
| `maclp.GE(upper_limit=500).fit_from_data(...)` / `.fit(...)` | Ω_GE, the interval Gaussian elimination box (baseline). Same interface as `LP`. |
| `maclp.intervals.find_C_itvls`, `maclp.intervals.find_q_itvls` | The interval step on its own; return dictionaries with `C_low`, `C_up` and `q_low`, `q_up`. |
| `maclp.utils.validate_constraints(C_low, C_up, q_low, q_up, W)` | Indices of the columns of `W` (shape K × n) that lie in Ω. |
| `maclp.toy_data_generating`, `maclp.bayes_classifier` | Gaussian mixture data and its Bayes classifier, for examples and simulations. |

After `fit`, an `LP` object also has

* `is_feasible_`: whether Ω is nonempty. Ω can be empty when no nonnegative ω satisfies the constraints; the bounds are then clipped to `bound_single`, so a weight that is forced negative gets the interval [0, 0] (the BBSE projection onto ω ≥ 0), and a warning is raised.
* `slack_sum_`, `slack_vector_`, `possible_negative_dims_`: the result of the phase-1 LP that checks feasibility.

The 2K linear programs run in parallel with OpenMP; set `OMP_NUM_THREADS` to choose the number of threads.

### Points inside Ω

Ω itself is a polytope, not a box. Uniform points of Ω can be drawn by rejection from Ω_LP:

```python
from maclp.intervals import find_C_itvls, find_q_itvls
from maclp.utils import validate_constraints

C_rgn = find_C_itvls(C_mtx, M, K, alpha_matrix=0.01)
q_rgn = find_q_itvls(q_dist, N, K, alpha_vec=0.01)
box = maclp.LP().fit(C_rgn["C_low"], C_rgn["C_up"], q_rgn["q_low"], q_rgn["q_up"])

candidates = rng.uniform(box[:, 0], box[:, 1], size=(10000, K))
inside = validate_constraints(C_rgn["C_low"], C_rgn["C_up"], q_rgn["q_low"], q_rgn["q_up"], candidates.T)
omega_points = candidates[inside]          # the fraction inside estimates vol(Omega) / vol(Omega_LP)
```

## Repository structure

```
.
├── README.md
├── setup.py                 build script of the C++ extension (looks for HiGHS)
├── pyproject.toml
├── test.py                  small usage examples, including an empty region
├── maclp/                   the Python package
│   ├── __init__.py
│   ├── estimators.py        BBSE, LP (Ω_LP) and GE (Ω_GE)
│   ├── intervals.py         Clopper–Pearson intervals for C and q
│   └── utils.py             confusion matrix and target counts, constraint check, toy data, Bayes classifier
├── src/                     C++ solver (module maclp._fast_aabb_cpp)
│   ├── aabb_solver.cpp      the 2K linear programs with HiGHS and OpenMP
│   └── bindings.cpp         pybind11 bindings
└── experiments/             code for the experiments in the paper (independent of the package)
    ├── exputils/            shared code: sampling, intervals, regions, conformal prediction, I/O, plotting
    ├── classifiers/         training and prediction for MNIST, CIFAR-10, AG News, N24News
    ├── benchmarks/          Dirichlet label shift experiment on the four benchmarks
    ├── nuimages/            Singapore -> Boston experiment on nuImages
    ├── synthetic/           Gaussian mixture studies
    ├── diagnostics/         classifier accuracy and calibration
    ├── tools/               synthetic prediction files for testing
    ├── requirements.txt
    └── requirements-classifiers.txt
```

## Citation

```bibtex
@inproceedings{li2026maclp,
  title     = {From Matrix Inversion to Constraints: Provably Tighter Confidence Regions for Importance Weights in Label Shift},
  author    = {Li, Mushan and Han, Kihyun and Ma, Yanyuan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
