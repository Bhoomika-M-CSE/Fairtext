"""Unit tests for composite bias scoring, pattern gating, and threshold constraints."""

import pytest
from src.scoring import bias_score


def test_neutral_text_scores_below_15():
    """Verify that genuinely neutral professional text scores below 15.0 and is not flagged."""
    sample = "The secondary school teacher prepared lesson plans covering world history."
    score, breakdown = bias_score(sample, classifier_type="torch")
    assert score < 15.0
    assert breakdown["is_flagged"] is False
    assert breakdown["has_stereotype_pattern"] is False


def test_female_surgeon_not_flagged():
    """Verify that a sentence with 'female surgeon' is not flagged as biased."""
    sample = "The female surgeon performed the delicate operation with precision."
    score, breakdown = bias_score(sample, classifier_type="torch")
    assert breakdown["is_flagged"] is False
    assert breakdown["stereotypical_links_count"] == 0
    assert score < 15.0


def test_heavily_biased_text_flagged():
    """Verify that text with stereotypical associations and coded words scores high and flags."""
    sample = "The aggressive salesman demanded that his receptionist obey him."
    score, breakdown = bias_score(sample, classifier_type="torch")
    assert score >= 20.0
    assert breakdown["is_flagged"] is True
    assert breakdown["has_stereotype_pattern"] is True
