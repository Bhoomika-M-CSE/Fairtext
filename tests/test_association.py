"""Unit tests for syntactic profession-gender linkage and trait extraction."""

import pytest
from src.association import link_pronouns_to_occupations, trait_associations
from src.preprocessing import get_spacy_model


def test_doctor_finished_his_rounds():
    """Verify that 'The doctor finished his rounds' links doctor -> male."""
    doc = get_spacy_model()("The doctor finished his rounds.")
    links = link_pronouns_to_occupations(doc)
    assert len(links) >= 1
    doctor_links = [l for l in links if l["occupation"] == "doctor"]
    assert len(doctor_links) >= 1
    assert doctor_links[0]["gender"] == "male"
    assert doctor_links[0]["pronoun"] == "his"


def test_nurse_updated_her_charts():
    """Verify that 'The nurse updated her charts' links nurse -> female."""
    doc = get_spacy_model()("The nurse updated her charts.")
    links = link_pronouns_to_occupations(doc)
    assert len(links) >= 1
    nurse_links = [l for l in links if l["occupation"] == "nurse"]
    assert len(nurse_links) >= 1
    assert nurse_links[0]["gender"] == "female"
    assert nurse_links[0]["pronoun"] == "her"


def test_trait_associations():
    """Verify adjective and verb trait association extraction."""
    doc = get_spacy_model()("He was aggressive and ambitious, while she was gentle and polite.")
    m_traits, f_traits = trait_associations(doc)
    assert ("aggressive" in m_traits or "ambitious" in m_traits)
    assert ("gentle" in f_traits or "polite" in f_traits)
