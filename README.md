# AI business intelligence and decision quality

Reproducible retrospective study for DBA 802. It tests whether probability calibration improves resource-constrained screening decisions on two real UCI business datasets. This is a research draft, not evidence of journal acceptance, organizational readiness, or causal business profit.

## Run

Tested on Python 3.12.14, Linux x86_64. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python research.py --self-test --run
```

The full run fits 40 models (four methods, five splits, two datasets), calibrates each on separate data, applies five paired test-corruption masks at three nonzero missingness levels, and regenerates all outputs. It runs on a CPU; the reference run limits numerical threads to two. Exact timing depends on hardware. Windows users can activate `.venv\Scripts\activate` instead.

To regenerate tables and figures from the included predictions without refitting:

```bash
python research.py --report
```

The program verifies raw archive SHA-256 checksums before reading. If an archive is missing it downloads the exact UCI source; if its content differs it stops. No synthetic records or labels are used. The complete archives are included for offline reproduction after dependency installation.

## Minimal layout

- `research.py`: data preparation, training, calibration, decisions, bootstrap, tests and plotting.
- `requirements.txt`: exact package versions used.
- `data/`: two original UCI ZIP archives.
- `results/`: two prediction archives, provenance, numerical analyses, two CSV tables, two research figures and a separate graphical abstract.

Each scientific figure is available as vector PDF and 600 dpi PNG. The graphical abstract is a separate 3000 by 1200 pixel PNG and vector PDF. It is drawn with Matplotlib, without an image-generation model.

## Protocol and interpretation

Bank Marketing uses five expanding chronological splits. Online Shoppers uses five stratified outer folds with an internal calibration split. Training, calibration and test records do not overlap within a split. Test records appear in only one evaluation window/fold. Temporal windows have overlapping training histories, so they are not independent replications. There are no customer identifiers with which to verify entity-level independence.

The model choices and primary cost/capacity were fixed before the first benchmark run, without external preregistration or test-set hyperparameter tuning. Baselines are smoothed descriptive BI, logistic regression, random forest and histogram gradient boosting. All receive the same benefit-cost threshold and capacity rule in the main comparison. The four ablations hold gradient boosting fixed and vary monotone Platt calibration and the decision threshold. Calibration alone is not a new learning algorithm.

The primary benchmark uses benefit 1, action cost 0.10 and capacity at most 20% per test fold. Utility is `1000 * sum(action * (observed_label - cost)) / N`. This is an assumed screening score, not measured revenue, treatment effect or ROI. The bank outcome follows historical contact; the shopping data contain session-level aggregates. Do not interpret either result as the causal benefit of intervening or as validated real-time deployment performance.

Exact duplicate source rows are removed before splitting. Bank call duration and shopping PageValues are excluded to reduce outcome-proximal information. This does not establish that every other variable would be available at a proposed real-world decision time. Excluded predictors, class proportions, split sizes and calibration parameters are recorded in `manifest.json`.

Test-only missingness rates are 0%, 15%, 30% and 50%, with five paired masks. The clean predictions are repeated in storage for a uniform array shape; they are not five independent observations. Training and calibration remain clean. The stress test is MCAR cell masking, not an empirical model of operational missingness.

Intervals use 2,000 paired conditional bootstrap samples: circular blocks of 100 rows within bank windows and rows within shopper folds. Fitted models, calibration maps and action selections remain fixed. Thus these intervals exclude model-retraining, split-selection, cost-estimation and corruption-mask uncertainty. They are descriptive and not adjusted for multiple comparisons.

## Key results

At the primary operating point, calibrated cost-aware logistic regression has the highest bank utility (34.15 per 1,000 cases), ahead of gradient boosting (17.93). Gradient boosting has the highest shopping utility (53.67), but its paired difference from random forest includes zero. The difference between calibrated and raw cost-aware gradient boosting is exactly zero in shopping because the same cases are selected. Do not report universal superiority or a causal return.

The bank final test window has a much higher positive rate than its calibration window. Several fitted monotone calibration slopes reach the lower bound; inspect the manifest. This is evidence of difficult temporal transport, not evidence that calibration is reliable under arbitrary shift. Better probability scoring and better resource allocation are separate evaluation questions.

## Output schema

`bank_predictions.npz` and `shop_predictions.npz` each contain:

- `raw`, `calibrated`: shape `(4 models, 4 missingness levels, 5 masks, N test records)`.
- `y`, `row_id`, `fold`: true binary outcome, zero-based original source row and fold index.

`analysis.json` contains paired contrasts, 768 sensitivity scenarios (2 datasets × 4 methods × 4 missingness levels × 4 costs × 6 capacities), and diagnostic action-set invariance checks. Those checks are not independent experiments. `table1.csv` and `table2.csv` retain more precision than the rounded manuscript tables.

Self-tests verify policy optimality against exhaustive small-case enumeration, stable tie breaking, zero-gain exclusion, monotone ranking invariance, prohibited-feature exclusion and disjoint partitions. The full experiment is also rerun to compare numerical outputs for reproducibility; see the accompanying audit for the result.

## Data attribution and terms

1. Moro, S., Rita, P., and Cortez, P. (2014). **Bank Marketing**. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306. Source: https://archive.ics.uci.edu/dataset/222/bank+marketing.
2. Sakar, C. O., and Kastro, Y. (2018). **Online Shoppers Purchasing Intention Dataset**. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q. Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset.

UCI lists both datasets under Creative Commons Attribution 4.0 (https://creativecommons.org/licenses/by/4.0/). Original archives are unmodified; row deduplication and feature exclusions occur only in memory. Derived predictions, aggregate tables and figures are produced by this code. See dataset citations above for required attribution. A software license has not been selected by the author; absence of a license is not a grant of unrestricted reuse.

## Research status

The professor's reference manuscript and the student's original draft are not included in this repository. Their text, figures and experimental contributions were not copied. Numerical results come from executing this Python program on the cited datasets. Human author review, required submission declarations, journal requirements and a public archival release remain necessary before submission. No journal submission has been performed.
