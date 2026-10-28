# Experiment protocol

## Reproduce

1. Install the project in a fresh Python 3.11 environment with `python -m pip install -e ".[dev]"`.
2. Run `python -m sms_spam train` from the repository root.
3. Inspect `reports/metrics.json` and the generated figures. The report records seed, Python version, installed library versions and dataset source information.
4. Run `python -m sms_spam predict --input examples/sample.json` to exercise the trained model.
5. Run `make check test` to verify the contracts.

## Selection rule

Casefolded, whitespace-normalized messages define duplicates. Conflicting duplicate-label groups are discarded; one representative of every remaining normalized message is retained. Seeded stratified splits allocate 60%/20%/20% to train/validation/test. Vectorizers learn vocabulary and IDF only on training text. Each candidate gets the validation threshold with maximum F1 subject to precision ≥ 0.98; the model with highest feasible validation F1 is selected. With no feasible threshold, the model predicts no spam.

## Reading the result

The selected word-plus-character logistic model reaches 97.56% test precision and 93.75% test recall. Its 98% validation precision target is not a guarantee on new messages: three legitimate messages are false positives on the test set. The prior baseline predicts no spam under this policy and has F1 zero. Scores are model outputs, not independently calibrated risk estimates.

The committed results are one seeded benchmark run. They do not establish confidence intervals, statistical significance, robustness across random seeds or production behavior. CV standard deviations summarize folds, not uncertainty on future production metrics. Do not tune settings against the final test set; use new validation experiments and reserve a fresh holdout if making repeated test-driven changes.

## Notebook

`notebooks/01_review_results.ipynb` reads committed reports and displays the evaluation figure. It is executed and can be viewed directly on GitHub without training. To rerun it locally, install `python -m pip install -e ".[notebook]"` and select that environment in your notebook editor. Training logic remains in the package so notebook execution cannot silently diverge from the command-line experiment.

## Resource boundaries

All examples run on CPU without cloud accounts or paid APIs. The default dataset archives are small, and raw data is cached locally. The CNN uses two CPU threads; the wine forest uses two workers. Reports and small images are committed, while model binaries, raw archives and environment folders are ignored. CI has a 15-minute execution limit.
