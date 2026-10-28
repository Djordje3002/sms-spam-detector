import numpy as np
import pandas as pd
import pytest

from sms_spam.model import choose_threshold, clean, metrics, normalize, predict


def test_normalized_duplicates_and_conflicts_are_removed():
    frame = pd.DataFrame(
        {
            "label": ["ham", "ham", "ham", "spam", "spam"],
            "text": ["Hello  there", "HELLO there ", "mixed", "MIXED", "prize"],
        }
    )
    result = clean(frame)
    assert result.text.tolist() == ["Hello  there", "prize"]
    assert result.normalized.is_unique


def test_threshold_respects_validation_precision_constraint():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.7, 0.6, 0.9])
    threshold = choose_threshold(y, p, minimum_precision=1.0)
    result = metrics(y, p, threshold)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_infeasible_target_abstains_from_spam_predictions():
    threshold = choose_threshold(np.array([0, 1]), np.array([0.9, 0.1]))
    assert threshold > 1


def test_casefold_and_whitespace_are_consistent():
    assert normalize("  Some\n TEXT ") == "some text"


@pytest.mark.parametrize("text", [None, "", " " * 3, "x" * 5001, 17])
def test_invalid_text_fails_before_model_loading(tmp_path, text):
    with pytest.raises(ValueError):
        predict(tmp_path, {"text": text})
