# Calibration, capacity and business decision quality

Reproducible companion to **When calibration changes business decisions: A knowledge based audit of AI, capacity and data quality**.

The experiment evaluates observed-label screening decisions on three public business datasets. It compares four predictors and seven calibration/threshold policies, with separate training, calibration-fit, calibration-selection and test data. It is an empirical audit of established methods, not a claim to introduce a new calibration algorithm.

## Reproduce

Use Python 3.12 and the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python research.py --self-test
python research.py --run
python research.py --report
python research.py --verify-refit bank
python research.py --verify-refit shop
python research.py --verify-refit churn
```

`--run` fits every model, saves predictions and creates the tables and figures. `--report` requires those prediction arrays and regenerates the analyses without fitting. The submission archive includes saved predictions; this compact repository does not version the large prediction arrays or PNG previews. Vector figure PDFs are versioned. Checkpoint files under `results/.cache/` are ignored by Git; completed checkpoints are reused only when the source/protocol fingerprint agrees.

The self-test checks source checksums, split disjointness, temporal ordering, feature exclusions, map monotonicity, plateau tie handling, capacity-constrained expected-value optimality and direct threshold optimization. Independent refit checks compare the first original-fit fold of each dataset at numerical tolerances; they are not a claim of independent third-party replication.

## Experimental design

- **Data:** Bank Marketing, Online Shoppers and Iranian Churn. Exact source-record duplicates are removed before splitting. Source archives and provenance are retained.
- **Splits:** five expanding chronological bank test windows; five stratified shopping/churn folds. Calibration-fit and calibration-selection records are separate. Hyperparameters are chosen on a holdout within training only.
- **Predictors:** descriptive BI frequency, logistic regression, random forest and histogram gradient boosting; bounded three-candidate tuning for each machine-learning predictor.
- **Calibration:** identity, monotone logit-Platt, monotone beta and isotonic. Two further policies choose the map using validation Brier loss or validation utility. A direct threshold baseline optimizes screening utility on calibration-fit records.
- **Primary decision:** assumed unit positive reward, cost 0.10, capacity 20%. All monotone-map decisions preserve raw-score order and stable source-row tie breaking. Isotonic plateaus do not reorder cases.
- **Stress:** independent predictor-cell missingness and complete predictor-record outages, each at 15%, 30% and 50%, with three paired masks. Training/calibration remain clean; observed labels never change.
- **Fitting stability:** original fit plus two bootstrap fitting perturbations, repeating tuning. These three values form a descriptive range, not a confidence interval.
- **Inference:** paired 2,000-draw test bootstrap, conditional on fitted models and actions. Bank uses circular within-window blocks of 100, with 25/250 sensitivity; other datasets use within-fold row resampling. No multiple-comparison correction or population-wide significance claim is made.
- **Sensitivity:** all 24 cost-capacity combinations are retained, without selecting favourable scenarios.

The protocol in `protocol.json` was specified after inspecting an earlier two-dataset benchmark and before running this expanded revision. It is not externally preregistered. The manifest records a report-only update after fitting (floating-point arithmetic compatibility, within-fold ranking summaries and figure formatting); the fitting and decision implementation was unchanged and refit checks accompany the outputs.

## Results and interpretation

There are **35,643 unique held-out records**. At the primary operating point, Brier-selected calibration changes utility, averaged equally over the four predictor specifications, by **-1.57** per 1,000 bank records, **0.00** for shopping and **0.00** for churn. The conditional bank interval is **[-2.60, -0.68]**, while the three-fit bank range is **[-1.57, 1.13]**. These summaries measure different sources of variation.

Shopping calibration policies select the same cases at the primary operating point. Churn's zero average reflects offsetting changes and does not establish identical decisions. The prefix identity passes in **4,320** clean model-fold-map scenarios; **2,047** have both raw and calibrated policies filling capacity and therefore identical actions. These scenarios reuse records and are not independent observations.

Utility is a normalized retrospective screening score based on assumed rewards and costs. It is not causal revenue, treatment benefit, return on investment, digital-transformation readiness, entrepreneurship success or measured strategic decision quality. Bank temporal shift and missing-data degradation limit transport. The current study does not claim universal superiority or first discovery of calibration-ranking divergence.

## Files

The repository has one implementation script, one protocol, one dependency file and only `data/` and `results/` directories.

| Output | Meaning |
|---|---|
| `table1.csv` | Dataset accounting and test samples |
| `table2.csv` | Individual-model predictive and utility comparisons |
| `table3.csv` | Seven policies, paired intervals, fitting ranges and stress values |
| `fold_results.csv` | All model-fold-refit summaries and selection choices |
| `sensitivity_results.csv` | All 504 dataset-policy-cost-capacity summaries |
| `stress_results.csv` | All 147 dataset-policy-stress summaries |
| `decision_diagnostics.csv` | All 4,320 prefix comparisons |
| `manifest.json`, `analysis.json` | Provenance, fitted maps/settings and numerical checks |
| `verification_*.json` | Independent same-environment refit checks |
| `figure1.pdf` | Cost-capacity utility differences |
| `figure2.pdf` | Brier-loss improvement versus utility change |
| `figure3.pdf` | Two missing-data mechanisms and fitting ranges |
| `figure4.pdf` | Bank prevalence shift and fitting stability |
| `graphical_abstract.pdf` | Programmatic workflow summary |

Research figures are available as vector PDF and generated 600-dpi PNG. All plots and the graphical abstract use Matplotlib. Main-table AUC/AP are weighted within-fold averages; pooled values are additionally retained in the CSV.

## Source data and attribution

UCI lists these datasets under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original archives are redistributed unchanged. The script applies deduplication, feature exclusions and test-only missingness as documented transformations.

- S. Moro, P. Rita and P. Cortez. **Bank Marketing** (2014). [DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306). Excludes call duration; 41,176 records retained.
- C. O. Sakar and Y. Kastro. **Online Shoppers Purchasing Intention Dataset** (2018). [DOI: 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Excludes PageValues; 12,205 records retained.
- **Iranian Churn** (2020). [DOI: 10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z). Excludes Status and Customer Value conservatively; 2,850 records retained. UCI documents predictor aggregation over nine months and churn at month twelve.

## Research status

The numerical results are generated by the executable implementation. Author review, required submission declarations, current journal requirements and reviewer access to this private repository remain necessary before formal submission. No journal submission or university-portal upload has been performed.
