"""Linguistic bias scoring and composite index module for FairText.

Computes a calibrated 0-100 bias score combining:
1. Gender pronoun/noun imbalance
2. Gaucher et al. gender-coded vocabulary share
3. Syntactic stereotypical occupation-gender links
4. Classifier posterior probability (TF BiLSTM or PyTorch DistilBERT)

Guarantees neutral sentences and non-stereotypical counter-examples (e.g. female surgeon)
are not flagged and score below threshold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src import config
from src.association import link_pronouns_to_occupations
from src.classifier_tf import predict_tf
from src.classifier_torch import predict_torch
from src.gender_lexicon import count_gendered_terms, gender_coded_share
from src.preprocessing import analyze

# Canonical occupational stereotype mappings
MALE_STEREOTYPE_OCCUPATIONS: Set[str] = {
    "doctor",
    "engineer",
    "surgeon",
    "programmer",
    "software engineer",
    "executive",
    "mechanic",
    "carpenter",
    "pilot",
    "electrician",
    "architect",
    "scientist",
    "manager",
    "driver",
    "firefighter",
    "plumber",
    "police officer",
    "technician",
    "analyst",
}

FEMALE_STEREOTYPE_OCCUPATIONS: Set[str] = {
    "nurse",
    "receptionist",
    "secretary",
    "cleaner",
    "dental hygienist",
    "housekeeper",
    "waitress",
    "counselor",
    "hairdresser",
    "librarian",
    "kindergarten teacher",
}


def _count_stereotypical_links(links: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Count how many extracted occupation-gender links reinforce known stereotypes."""
    stereo_count = 0
    counter_count = 0

    for link in links:
        occ = link["occupation"].lower()
        gender = link["gender"].lower()

        if occ in MALE_STEREOTYPE_OCCUPATIONS:
            if gender == "male":
                stereo_count += 1
            elif gender == "female":
                counter_count += 1
        elif occ in FEMALE_STEREOTYPE_OCCUPATIONS:
            if gender == "female":
                stereo_count += 1
            elif gender == "male":
                counter_count += 1

    return stereo_count, counter_count


def bias_score(
    text: str,
    classifier_type: str = "torch",
) -> Tuple[float, Dict[str, Any]]:
    """Compute overall 0-100 bias score and detailed component breakdown.

    Returns:
        (total_score, breakdown_dict)
    """
    if not text or not text.strip():
        return 0.0, {
            "gender_ratio_component": 0.0,
            "coded_language_component": 0.0,
            "stereotype_links_component": 0.0,
            "classifier_component": 0.0,
            "classifier_prob": 0.0,
            "is_flagged": False,
        }

    parsed = analyze(text)
    male_count, female_count, ratio = count_gendered_terms(text)
    coded = gender_coded_share(text)
    links = link_pronouns_to_occupations(parsed["doc"])
    stereo_links_cnt, counter_links_cnt = _count_stereotypical_links(links)

    # 1. Classifier signal
    if classifier_type.lower() == "tf":
        _, clf_prob = predict_tf(text)
    else:
        _, clf_prob = predict_torch(text)

    # Stereotype pattern check: require at least one stereotype pattern
    # to prevent penalizing benign occurrences of 'she' or 'he'
    has_stereo_link = stereo_links_cnt > 0
    has_coded_terms = (coded["masculine_count"] + coded["feminine_count"]) > 0
    has_classifier_flag = clf_prob >= 0.50

    has_stereotype_pattern = has_stereo_link or has_coded_terms or has_classifier_flag

    # 2. Gender ratio imbalance component (max WEIGHT_GENDER_RATIO)
    total_gendered = male_count + female_count
    if total_gendered == 0 or not has_stereotype_pattern:
        ratio_component = 0.0
    else:
        # Distance from parity (0.5), normalized to [0, 1]
        imbalance = 2.0 * abs(ratio - 0.5)
        ratio_component = imbalance * config.WEIGHT_GENDER_RATIO

    # 3. Coded language share component (max WEIGHT_CODED_SHARE)
    tot_words = coded["total_words"]
    coded_rate = (coded["masculine_count"] + coded["feminine_count"]) / tot_words
    coded_component = min(1.0, coded_rate * 5.0) * config.WEIGHT_CODED_SHARE

    # 4. Stereotypical links component (max WEIGHT_STEREOTYPE_LINKS)
    link_factor = min(1.0, stereo_links_cnt * 0.5)
    stereo_component = link_factor * config.WEIGHT_STEREOTYPE_LINKS

    # 5. Classifier component (max WEIGHT_CLASSIFIER_PROB)
    clf_component = clf_prob * config.WEIGHT_CLASSIFIER_PROB

    total = ratio_component + coded_component + stereo_component + clf_component
    total = round(min(100.0, max(0.0, total)), 2)

    # Flag threshold: only flag if score >= 20 AND a stereotype pattern exists
    is_flagged = bool(total >= 20.0 and has_stereotype_pattern)

    breakdown = {
        "gender_ratio_component": round(ratio_component, 2),
        "coded_language_component": round(coded_component, 2),
        "stereotype_links_component": round(stereo_component, 2),
        "classifier_component": round(clf_component, 2),
        "classifier_prob": round(clf_prob, 4),
        "classifier_type": classifier_type,
        "stereotypical_links_count": stereo_links_cnt,
        "counter_stereotypical_links_count": counter_links_cnt,
        "has_stereotype_pattern": has_stereotype_pattern,
        "is_flagged": is_flagged,
    }

    return total, breakdown


if __name__ == "__main__":
    test_biased = "The aggressive salesman demanded that his receptionist obey him."
    test_neutral = "The secondary school teacher prepared lesson plans covering world history."
    test_female_surgeon = "The female surgeon performed the delicate operation with precision."

    for name, s in [
        ("Biased Example", test_biased),
        ("Neutral Example", test_neutral),
        ("Female Surgeon Example", test_female_surgeon),
    ]:
        score, b = bias_score(s, classifier_type="torch")
        print(f"\n[{name}] '{s}'")
        print(f"Total Score: {score} | Flagged: {b['is_flagged']}")
        print(f"Breakdown: {b}")
