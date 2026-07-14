# ML Class Project

Two small, standalone Python scripts that train and compare classic scikit-learn classifiers (KNN, Naive Bayes, SVM, Logistic Regression, Decision Tree) on two well-known toy datasets: the Iris flower dataset and the Pima Indians Diabetes dataset. Written as coursework / self-study exercises, following the classic "Machine Learning Mastery" tutorial pattern (load data → summarize → cross-validate several models → plot comparison → fit and evaluate one model on a held-out split).

## Status: Archived

This project is **archived and no longer maintained**. It was a one-off class exercise, not a maintained tool or library. It has been minimally updated (see below) to keep it runnable, but no further development is planned.

## Tech stack

- Python 3 (tested with 3.12)
- [pandas](https://pandas.pydata.org/) — data loading/summary
- [scikit-learn](https://scikit-learn.org/) — models, cross-validation, metrics
- [matplotlib](https://matplotlib.org/) — histograms / boxplots

Both scripts pull their dataset directly from the internet at run time (no local data files):

- `iris.py` → [UCI Iris dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- `mlcl.py` → [Pima Indians Diabetes dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) (mirrored by Jason Brownlee)

An internet connection is required to run either script.

## Setup & run (verified 2026-07-14)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python iris.py
python mlcl.py
```

Both scripts were run end-to-end and confirmed working with these versions: `pandas 3.0.3`, `scikit-learn 1.9.0`, `matplotlib 3.11.0`, `numpy 2.5.1`. `requirements.txt` is unpinned to allow future compatible releases; if something breaks, try pinning to the versions above.

Each script opens matplotlib windows (`plt.show()`) for a histogram and a boxplot before printing results to the console — this requires a display. To run headlessly (e.g. in CI or over SSH), set `MPLBACKEND=Agg` first:

```bash
MPLBACKEND=Agg python iris.py
```

### What was fixed to get this running again

The scripts were written years ago against an older scikit-learn API and hadn't been run since. To get them working on a current scikit-learn (1.9.x):

- **`iris.py` / `mlcl.py`** — `model_selection.KFold(n_splits=10, random_state=seed)` now raises `ValueError: Setting a random_state has no effect since shuffle is False`. Added `shuffle=True` to match modern scikit-learn's requirement for using `random_state` with `KFold`. This is a behavior-preserving fix (the original intent was clearly a seeded, reproducible split); no results-affecting logic was otherwise changed.
- **`mlcl.py`** — the shebang line pointed to the original author's local machine path (`#!/home/littlegaintl/Codespace/MLclass/env python`), which doesn't exist on any other system. Replaced with `#!/usr/bin/env python3`. Scripts are normally invoked as `python iris.py` / `python mlcl.py`, so this had no effect on the runs above, but it's now safe to run as `./mlcl.py` too.
- Added `requirements.txt` (none existed previously).

Everything else is unchanged from the original coursework code.

### Known cosmetic warning

`mlcl.py`'s Logistic Regression model prints a `ConvergenceWarning` (`lbfgs failed to converge after 100 iteration(s)`) during cross-validation. This is harmless — the script still completes and prints results — and was left as-is since fixing it (e.g. increasing `max_iter` or scaling features) would change behavior beyond "make it run as originally intended."

## Structure

- [`iris.py`](iris.py) — loads the Iris dataset, plots histograms, cross-validates KNN/Naive Bayes/SVM, plots a comparison boxplot, then fits a KNN model on a train/validation split and prints accuracy, a confusion matrix, and a classification report.
- [`mlcl.py`](mlcl.py) — same pattern applied to the Pima Indians Diabetes dataset, comparing Logistic Regression, KNN, Decision Tree (CART), Naive Bayes, and SVM via 10-fold cross-validation.

There is no shared code between the two scripts, no tests, and no CI configuration.

## License

GPL-3.0 — see [LICENSE](LICENSE).
