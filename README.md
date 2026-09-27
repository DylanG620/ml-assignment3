# Training Neural Networks with SGD, SCG and LeapFrog

Code for Assignment 3 (Option 2) of Machine Learning 441/741 at Stellenbosch University, by Dylan Gagiano.

A neural network with one hidden layer is trained by three algorithms on three classification
and three regression problems:

- **SGD**: stochastic gradient descent with momentum
- **SCG**: scaled conjugate gradient (Møller, 1993)
- **LeapFrog**: the LeapFrog optimiser LFOP1(b) (Snyman, 1983)

The network and all three algorithms are written from scratch in NumPy.

## Where to start

| Notebook | What it does |
|---|---|
| [`1_main_experiment.ipynb`](1_main_experiment.ipynb) | **The main experiment of the report.** The network, the three algorithms, Step 1 (hidden units), Step 2 (grid search), Step 3 (five-fold evaluation), the Friedman and Nemenyi tests, the convergence curves, the box plots and the LeapFrog spike analysis. Makes every figure of the report. |
| [`2_bayesian_optimisation.ipynb`](2_bayesian_optimisation.ipynb) | Tunes the control parameters again with Bayesian optimisation (a Gaussian process with expected improvement) and compares the result with the grid search. |
| [`3_pytorch_version.ipynb`](3_pytorch_version.ipynb) | Extra: the same network and algorithms written in PyTorch. Not used in the report and saved without outputs. |

The notebooks are saved with their outputs, so the tables and figures show on GitHub without running anything.

## Repository layout

```
.
├── 1_main_experiment.ipynb
├── 2_bayesian_optimisation.ipynb
├── 3_pytorch_version.ipynb
├── requirements.txt
└── results/
    ├── arch_sweep.csv        Step 1: validation errors for every number of hidden units
    ├── tuning.csv            Step 2: validation errors for every grid setting
    ├── final.csv             Step 3: test errors, test losses and times (five folds)
    ├── convergence.csv       training and held-out loss after every gradient calculation
    ├── leapfrog_trace.csv    LeapFrog time steps and gradient norms (diabetes and airfoil)
    ├── figures/              the figures of the report (PDF)
    └── bayesian/
        ├── trials.csv        every Bayesian optimisation trial
        ├── final.csv         five-fold results of the grid and the Bayesian settings
        └── search_progress.png
```

## The problems

| Problem | Task | Patterns | Inputs | Outputs | Hidden units |
|---|---|---:|---:|---:|---:|
| Wine | classification | 178 | 13 | 3 | 2 |
| Breast cancer | classification | 569 | 30 | 2 | 2 |
| Digits | classification | 1797 | 64 | 10 | 32 |
| Diabetes | regression | 442 | 10 | 1 | 2 |
| Airfoil | regression | 1503 | 5 | 1 | 16 |
| California housing | regression | 20640 | 8 | 1 | 32 |

All six datasets come from scikit-learn. Airfoil (OpenML 44957) and California housing are
downloaded the first time they are used.

## The experiment

1. **Step 1: hidden units.** Fourteen sizes from 1 to 128 are trained on five folds with the
   default control parameters. The one-standard-error rule picks the smallest size whose mean validation
   error is within one standard error of the lowest mean. Diabetes uses two units instead of one.
2. **Step 2: control parameters.** A grid search on five new folds: 49 settings for SGD, 9 for
   SCG and 36 for LeapFrog. Notebook 2 repeats this step with Bayesian optimisation.
3. **Step 3: evaluation.** Five-fold cross-validation on the complete datasets. The three
   algorithms share the folds and the starting weights.
4. **Statistics.** A Friedman test over the six problems, followed by Nemenyi comparisons.

Every training run gets 200 gradient calculations on the full training set (200 epochs for SGD).
Steps 1 and 2 use at most 5000 patterns per problem.

## Main findings

- SGD achieved significantly lower test errors and test losses than SCG.
- SCG needed the least tuning and trained significantly faster than SGD.
- The LeapFrog loss spiked when the time step exceeded the stability limit of the leapfrog scheme.
- Bayesian optimisation did not consistently improve on grid search.

## Running the code

```bash
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks from this folder, so that the paths to `results/` work. The slow parts save
their results as CSV files in `results/`. When a file exists, the notebook reads it instead of
training again:

| Part | Time without the saved file |
|---|---|
| Notebook 1, Steps 1 and 2 | about an hour |
| Notebook 1, Step 3, convergence curves and LeapFrog trace | several minutes |
| Notebook 2, searches and final runs | about half an hour |

Delete a CSV file to repeat that part. The results were produced with Python 3.12, NumPy 2.5,
pandas 3.0, SciPy 1.18 and scikit-learn 1.9, with one BLAS thread (the two-unit diabetes runs of
notebook 2 with Python 3.11 and NumPy 2.4). Floating-point differences
between computers can change a few validation errors (mostly for SCG), and with them a few
selected settings.

## References

- M. F. Møller, "A scaled conjugate gradient algorithm for fast supervised learning," *Neural Networks*, vol. 6, no. 4, pp. 525–533, 1993.
- J. A. Snyman, "A new and dynamic method for unconstrained minimization," *Applied Mathematical Modelling*, vol. 6, no. 6, pp. 449–462, 1982.
- J. A. Snyman, "An improved version of the original leap-frog dynamic method for unconstrained minimization: LFOP1(b)," *Applied Mathematical Modelling*, vol. 7, no. 3, pp. 216–218, 1983.
- J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *Journal of Machine Learning Research*, vol. 7, pp. 1–30, 2006.
- J. Snoek, H. Larochelle and R. P. Adams, "Practical Bayesian optimization of machine learning algorithms," in *Advances in Neural Information Processing Systems*, vol. 25, pp. 2951–2959, 2012.
