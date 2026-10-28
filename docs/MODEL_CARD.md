# Model card — SMS Spam Detector

Version: 1.0.0 · Experiment seed: 42 · Selected model: `word_char_logistic`

## Intended use

Classify short English SMS messages as ham or spam. Blocking a legitimate message has a different cost from missing spam, so this project chooses the decision threshold against an explicit validation precision target and reports the tradeoff on held-out messages.

The intended audience is engineers reviewing a reproducible ML portfolio and learners inspecting model selection, evaluation and inference. The project has no claimed deployment history or customer results.

## Training and evaluation

Casefolded, whitespace-normalized messages define duplicates. Conflicting duplicate-label groups are discarded; one representative of every remaining normalized message is retained. Seeded stratified splits allocate 60%/20%/20% to train/validation/test. Vectorizers learn vocabulary and IDF only on training text. Each candidate gets the validation threshold with maximum F1 subject to precision ≥ 0.98; the model with highest feasible validation F1 is selected. With no feasible threshold, the model predicts no spam.

| Model | average precision | precision | recall | f1 |
| --- | ---: | ---: | ---: | ---: |
| `prior_baseline` | 0.1240 | 0.0000 | 0.0000 | 0.0000 |
| `word_naive_bayes` | 0.9669 | 0.9741 | 0.8828 | 0.9262 |
| `word_char_logistic` | 0.9877 | 0.9756 | 0.9375 | 0.9562 |


## Interpretation

The selected word-plus-character logistic model reaches 97.56% test precision and 93.75% test recall. Its 98% validation precision target is not a guarantee on new messages: three legitimate messages are false positives on the test set. The prior baseline predicts no spam under this policy and has F1 zero. Scores are model outputs, not independently calibrated risk estimates.

## Limitations

Near-duplicate templates can survive exact normalized deduplication. Sender IDs, campaign IDs and reliable chronology are unavailable, so this is not an unseen-campaign or temporal deployment estimate. Non-English text, modern phishing campaigns and adversarial obfuscation require fresh evaluation. Do not automatically delete user messages based on this demonstration.

## Artifacts and maintenance

Run the training command to rebuild local artifacts and reports. Load only your own trusted local artifacts. For any data or model change, rerun training, compare with the baseline under the same split protocol, and inspect failure cases. A changed dataset hash or split constitutes a different experiment and must be documented as such. Fresh labeled data is required to estimate real deployment performance.
