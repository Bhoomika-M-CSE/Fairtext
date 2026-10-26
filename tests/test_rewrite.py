"""Unit tests for text neutralization, verb agreement, and capitalization preservation."""

import pytest
from src.rewrite import neutralize_text, suggest_rewrites


def test_chairman_to_chairperson_capitalization_preserved():
    """Verify that 'chairman' becomes 'chairperson' preserving title/lower/upper cases."""
    # Titlecase
    t1 = "The Chairman opened the session."
    r1 = neutralize_text(t1)
    assert "Chairperson" in r1
    assert "Chairman" not in r1

    # Lowercase
    t2 = "The chairman opened the session."
    r2 = neutralize_text(t2)
    assert "chairperson" in r2
    assert "chairman" not in r2

    # Uppercase
    t3 = "THE CHAIRMAN WAS PRESENT."
    r3 = neutralize_text(t3)
    assert "CHAIRPERSON" in r3


def test_pronoun_and_verb_agreement():
    """Verify that 'he was' converts to 'they were' and 'she is' to 'they are'."""
    t1 = "He was excited about the project."
    r1 = neutralize_text(t1)
    assert "They were" in r1

    t2 = "She is an experienced architect."
    r2 = neutralize_text(t2)
    assert "They are" in r2


def test_multiple_gendered_terms():
    """Verify simultaneous replacement of nouns and pronouns."""
    text = "The policeman spoke to the waitress after he finished his patrol."
    rewritten = neutralize_text(text)
    assert "police officer" in rewritten
    assert "server" in rewritten
    assert "they" in rewritten
    assert "their" in rewritten
