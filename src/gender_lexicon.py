"""Gender lexicon and coded language analyzer for FairText.

Provides term matching against male/female lexicons, Gaucher et al. (2011)
gender-coded language detection, NLTK name gender estimation, and composite
document gender profiling.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nltk
from nltk.corpus import names

from src import config
from src.preprocessing import analyze

_MALE_TERMS: Optional[Set[str]] = None
_FEMALE_TERMS: Optional[Set[str]] = None
_MASCULINE_CODED: Optional[Set[str]] = None
_FEMININE_CODED: Optional[Set[str]] = None
_NEUTRAL_MAP: Optional[Dict[str, str]] = None
_NAME_GENDER_CACHE: Optional[Dict[str, str]] = None


def load_lexicons() -> Tuple[Set[str], Set[str], Set[str], Set[str], Dict[str, str]]:
    """Load and cache all gender and coded language lexicons."""
    global _MALE_TERMS, _FEMALE_TERMS, _MASCULINE_CODED, _FEMININE_CODED, _NEUTRAL_MAP

    if _MALE_TERMS is None:
        with open(config.MALE_TERMS_PATH, "r", encoding="utf-8") as f:
            _MALE_TERMS = {line.strip().lower() for line in f if line.strip()}

    if _FEMALE_TERMS is None:
        with open(config.FEMALE_TERMS_PATH, "r", encoding="utf-8") as f:
            _FEMALE_TERMS = {line.strip().lower() for line in f if line.strip()}

    if _MASCULINE_CODED is None:
        with open(config.MASCULINE_CODED_PATH, "r", encoding="utf-8") as f:
            _MASCULINE_CODED = {line.strip().lower() for line in f if line.strip()}

    if _FEMININE_CODED is None:
        with open(config.FEMININE_CODED_PATH, "r", encoding="utf-8") as f:
            _FEMININE_CODED = {line.strip().lower() for line in f if line.strip()}

    if _NEUTRAL_MAP is None:
        with open(config.NEUTRAL_ALTERNATIVES_PATH, "r", encoding="utf-8") as f:
            _NEUTRAL_MAP = json.load(f)

    return _MALE_TERMS, _FEMALE_TERMS, _MASCULINE_CODED, _FEMININE_CODED, _NEUTRAL_MAP


def count_gendered_terms(text: str) -> Tuple[int, int, float]:
    """Count explicit male and female terms and return (male_count, female_count, male_ratio).

    Ratio is male / (male + female), default 0.5 when no gendered terms appear.
    """
    male_terms, female_terms, _, _, _ = load_lexicons()
    parsed = analyze(text)
    words = [t.lower() for t in parsed["tokens"] if t.isalpha()]

    male_count = sum(1 for w in words if w in male_terms)
    female_count = sum(1 for w in words if w in female_terms)
    total = male_count + female_count
    ratio = (male_count / total) if total > 0 else 0.5

    return male_count, female_count, round(ratio, 4)


def gender_coded_share(text: str) -> Dict[str, Any]:
    """Compute masculine- and feminine-coded word frequencies and proportions."""
    _, _, masc_coded, fem_coded, _ = load_lexicons()
    parsed = analyze(text)
    words = [t.lower() for t in parsed["tokens"] if t.isalpha()]
    total_words = len(words) or 1

    masc_matches = [w for w in words if w in masc_coded]
    fem_matches = [w for w in words if w in fem_coded]

    masc_count = len(masc_matches)
    fem_count = len(fem_matches)

    return {
        "masculine_count": masc_count,
        "feminine_count": fem_count,
        "masculine_matches": masc_matches,
        "feminine_matches": fem_matches,
        "masculine_share": round(masc_count / total_words, 4),
        "feminine_share": round(fem_count / total_words, 4),
        "total_words": total_words,
    }


def _get_name_gender_dict() -> Dict[str, str]:
    """Build NLTK name gender lookup dictionary."""
    global _NAME_GENDER_CACHE
    if _NAME_GENDER_CACHE is None:
        try:
            male_names = Counter(name.lower() for name in names.words("male.txt"))
            female_names = Counter(name.lower() for name in names.words("female.txt"))
            all_names = set(male_names.keys()) | set(female_names.keys())
            cache: Dict[str, str] = {}
            for n in all_names:
                m_cnt = male_names.get(n, 0)
                f_cnt = female_names.get(n, 0)
                tot = m_cnt + f_cnt
                if tot == 0:
                    cache[n] = "unknown"
                elif (m_cnt / tot) > 0.65:
                    cache[n] = "male"
                elif (f_cnt / tot) > 0.65:
                    cache[n] = "female"
                else:
                    cache[n] = "unknown"
            _NAME_GENDER_CACHE = cache
        except Exception:
            _NAME_GENDER_CACHE = {}
    return _NAME_GENDER_CACHE


def estimate_person_genders(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Estimate genders of PERSON entities using NLTK names corpus."""
    name_lookup = _get_name_gender_dict()
    results: List[Dict[str, Any]] = []

    for ent in entities:
        if ent.get("label") == "PERSON":
            first_name = ent.get("text", "").strip().split()[0].lower() if ent.get("text") else ""
            gender = name_lookup.get(first_name, "unknown")
            results.append({
                "entity": ent["text"],
                "first_name": first_name,
                "estimated_gender": gender,
            })
    return results


def document_gender_profile(text: str) -> Dict[str, Any]:
    """Generate comprehensive gender profile combining term counts, coded words, and entities."""
    parsed = analyze(text)
    male_count, female_count, ratio = count_gendered_terms(text)
    coded = gender_coded_share(text)
    persons = estimate_person_genders(parsed["entities"])

    return {
        "text": parsed["text"],
        "male_terms_count": male_count,
        "female_terms_count": female_count,
        "gender_term_ratio": ratio,
        "coded_language": coded,
        "person_entities": persons,
        "sentiment": parsed["sentiment"],
    }


if __name__ == "__main__":
    sample = "The competitive businessman said he admired the empathetic nurse."
    profile = document_gender_profile(sample)
    print(f"Profile: {profile}")
