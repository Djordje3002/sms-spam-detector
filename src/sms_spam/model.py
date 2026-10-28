"""Text deduplication and validation-only threshold selection."""

import io
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, make_pipeline

from .common import archive, figure, save_report


def normalize(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def clean(frame):
    frame = frame.assign(normalized=frame.text.map(normalize))
    conflicts = frame.groupby("normalized").label.nunique()
    frame = frame[~frame.normalized.isin(conflicts[conflicts > 1].index)]
    return frame.drop_duplicates("normalized").reset_index(drop=True)


def choose_threshold(y, probability, minimum_precision=0.98):
    candidates = np.unique(np.r_[probability, 1.000001])
    best = (0.0, 1.000001)
    for threshold in candidates:
        prediction = probability >= threshold
        precision = precision_score(y, prediction, zero_division=0)
        score = f1_score(y, prediction, zero_division=0)
        if precision >= minimum_precision and score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def metrics(y, probability, threshold):
    prediction = probability >= threshold
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, prediction, labels=[0, 1])
        .ravel()
        .tolist(),
        "threshold": float(threshold),
    }


def train(root):
    with archive(root) as source:
        frame = pd.read_csv(
            io.BytesIO(source.read("SMSSpamCollection")),
            sep="\t",
            header=None,
            names=["label", "text"],
            quoting=3,
        )
    raw = len(frame)
    frame = clean(frame)
    y = (frame.label == "spam").astype(int).to_numpy()
    fit, remainder = train_test_split(
        np.arange(len(frame)), test_size=0.4, stratify=y, random_state=42
    )
    valid, test = train_test_split(remainder, test_size=0.5, stratify=y[remainder], random_state=42)
    candidates = {
        "prior_baseline": make_pipeline(TfidfVectorizer(), DummyClassifier(strategy="prior")),
        "word_naive_bayes": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2), MultinomialNB(alpha=0.2)
        ),
        "word_char_logistic": make_pipeline(
            FeatureUnion(
                [
                    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
                    (
                        "char",
                        TfidfVectorizer(
                            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000
                        ),
                    ),
                ]
            ),
            LogisticRegression(C=4, class_weight="balanced", max_iter=1500, random_state=42),
        ),
    }
    validation, thresholds = {}, {}
    for name, model in candidates.items():
        model.fit(frame.text.iloc[fit], y[fit])
        probability = model.predict_proba(frame.text.iloc[valid])[:, 1]
        thresholds[name] = choose_threshold(y[valid], probability)
        validation[name] = metrics(y[valid], probability, thresholds[name])
    selected = max(validation, key=lambda name: validation[name]["f1"])
    results = {}
    _, ax = plt.subplots(figsize=(7, 5))
    for name, model in candidates.items():
        probability = model.predict_proba(frame.text.iloc[test])[:, 1]
        results[name] = metrics(y[test], probability, thresholds[name])
        PrecisionRecallDisplay.from_predictions(y[test], probability, name=name, ax=ax)
    plt.title("Spam detection: held-out precision–recall")
    figure(root, "evaluation.png")
    result = save_report(
        root,
        {
            "raw_rows": raw,
            "unique_unambiguous_messages": len(frame),
            "split": {"train": len(fit), "validation": len(valid), "test": len(test)},
            "selected": selected,
            "validation_precision_target": 0.98,
            "validation": validation,
            "test": results,
        },
    )
    joblib.dump(
        {"model": candidates[selected], "threshold": thresholds[selected]},
        root / "artifacts/model.joblib",
    )
    return result


def predict(root, payload):
    text = payload.get("text")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 5000:
        raise ValueError("text must contain 1–5000 characters")
    artifact = joblib.load(root / "artifacts/model.joblib")
    probability = float(artifact["model"].predict_proba([text])[0, 1])
    return {
        "label": "spam" if probability >= artifact["threshold"] else "ham",
        "spam_score": probability,
        "threshold": artifact["threshold"],
    }
