"""Unit tests for gender lexicon counting and name gender estimation."""

import pytest
from src.gender_lexicon import (
    count_gendered_terms,
    estimate_person_genders,
    gender_coded_share,
    load_lexicons,
)


def test_lexicons_loading():
    """Verify that all lexicons load non-empty collections."""
    male_terms, female_terms, masc_coded, fem_coded, neutral_map = load_lexicons()
    assert len(male_terms) > 20
    assert len(female_terms) > 20
    assert len(masc_coded) > 10
    assert len(fem_coded) > 10
    assert len(neutral_map) > 20


def test_gendered_terms_counts_and_ratio():
    """Verify male/female term counts and ratio calculation."""
    text_male = "He is a king and a father."
    m_cnt, f_cnt, ratio = count_gendered_terms(text_male)
    assert m_cnt == 3
    assert f_cnt == 0
    assert ratio == 1.0

    text_female = "She is a queen and a mother."
    m_cnt, f_cnt, ratio = count_gendered_terms(text_female)
    assert m_cnt == 0
    assert f_cnt == 3
    assert ratio == 0.0

    text_balanced = "He spoke with her."
    m_cnt, f_cnt, ratio = count_gendered_terms(text_balanced)
    assert m_cnt == 1
    assert f_cnt == 1
    assert ratio == 0.5

    text_neutral = "The team deployed the software service."
    m_cnt, f_cnt, ratio = count_gendered_terms(text_neutral)
    assert m_cnt == 0
    assert f_cnt == 0
    assert ratio == 0.5


def test_coded_language_share():
    """Verify detection of masculine-coded and feminine-coded words."""
    sample = "We need an ambitious, aggressive, and decisive leader."
    res = gender_coded_share(sample)
    assert res["masculine_count"] >= 3
    assert res["feminine_count"] == 0
    assert res["masculine_share"] > 0.0


def test_name_gender_estimation():
    """Verify gender estimation using NLTK names corpus."""
    entities = [
        {"label": "PERSON", "text": "John Smith"},
        {"label": "PERSON", "text": "Mary Johnson"},
        {"label": "PERSON", "text": "Alex Taylor"},
    ]
    results = estimate_person_genders(entities)
    assert len(results) == 3
    assert results[0]["estimated_gender"] == "male"
    assert results[1]["estimated_gender"] == "female"
    assert results[2]["estimated_gender"] in {"unknown", "male", "female"}
