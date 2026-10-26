"""Profession and trait association analysis module for FairText.

Extracts syntactic and coreference links between gendered pronouns and
professions, computes profession-gender distributions, identifies gendered
trait associations, and produces visual reports.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Counter as CounterType, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from spacy.tokens import Doc, Token

from src import config
from src.preprocessing import analyze, get_spacy_model

_OCCUPATIONS: Optional[List[str]] = None
MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}


def load_occupations() -> List[str]:
    """Load canonical list of occupations from data/lexicons/occupations.txt."""
    global _OCCUPATIONS
    if _OCCUPATIONS is None:
        with open(config.OCCUPATIONS_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip().lower() for line in f if line.strip()]
            # Sort by descending length so multi-word titles match first
            _OCCUPATIONS = sorted(lines, key=len, reverse=True)
    return _OCCUPATIONS


def _find_occupations_in_doc(doc: Doc) -> List[Tuple[str, Token]]:
    """Locate occupation occurrences in a spaCy Doc."""
    occupations = load_occupations()
    found: List[Tuple[str, Token]] = []
    text_lower = doc.text.lower()

    for occ in occupations:
        if occ in text_lower:
            for token in doc:
                # Direct match or compound head
                if token.text.lower() == occ or token.lemma_.lower() == occ:
                    found.append((occ, token))
                elif " " in occ:
                    span_text = doc[token.i : token.i + len(occ.split())].text.lower()
                    if span_text == occ:
                        found.append((occ, token))
    # Deduplicate while preserving order
    unique_found: List[Tuple[str, Token]] = []
    seen = set()
    for occ, tok in found:
        if (occ, tok.i) not in seen:
            seen.add((occ, tok.i))
            unique_found.append((occ, tok))
    return unique_found


def _extract_dep_links(doc: Doc, occ_name: str, occ_token: Token) -> List[Tuple[str, str, str]]:
    """Extract pronoun links for an occupation using syntactic dependencies."""
    links: List[Tuple[str, str, str]] = []

    # 1. Direct possessive modifier (e.g. "his doctor", "her nurse")
    for child in occ_token.children:
        if child.dep_ == "poss" and child.text.lower() in (MALE_PRONOUNS | FEMALE_PRONOUNS):
            gender = "male" if child.text.lower() in MALE_PRONOUNS else "female"
            links.append((occ_name, child.text.lower(), gender))

    # 2. Possessive in verb object (e.g. "The doctor finished his rounds")
    verb = occ_token.head
    if occ_token.dep_ in {"nsubj", "nsubjpass"} and verb.pos_ == "VERB":
        for child in verb.children:
            if child.dep_ in {"dobj", "attr", "prep", "pobj"}:
                for subchild in child.subtree:
                    if subchild.dep_ == "poss" and subchild.text.lower() in (MALE_PRONOUNS | FEMALE_PRONOUNS):
                        gender = "male" if subchild.text.lower() in MALE_PRONOUNS else "female"
                        links.append((occ_name, subchild.text.lower(), gender))

    # 3. Complement clause subject (e.g. "The doctor said that he was ready")
    if occ_token.dep_ in {"nsubj", "nsubjpass"} and verb.pos_ in {"VERB", "AUX"}:
        for sibling in verb.children:
            if sibling.dep_ in {"ccomp", "xcomp", "advcl"}:
                for subchild in sibling.children:
                    if subchild.dep_ in {"nsubj", "nsubjpass"} and subchild.text.lower() in (MALE_PRONOUNS | FEMALE_PRONOUNS):
                        gender = "male" if subchild.text.lower() in MALE_PRONOUNS else "female"
                        links.append((occ_name, subchild.text.lower(), gender))

    return links


def link_pronouns_to_occupations(doc: Union[Doc, str]) -> List[Dict[str, Any]]:
    """Link gendered pronouns to occupation nouns using syntactic dependencies."""
    if isinstance(doc, str):
        doc = get_spacy_model()(doc)

    occupations_in_doc = _find_occupations_in_doc(doc)
    extracted: List[Dict[str, Any]] = []

    for occ_name, occ_tok in occupations_in_doc:
        links = _extract_dep_links(doc, occ_name, occ_tok)

        # Fallback: nearest antecedent rule within the same sentence
        if not links:
            sent = occ_tok.sent
            for token in sent:
                word = token.text.lower()
                if word in (MALE_PRONOUNS | FEMALE_PRONOUNS):
                    gender = "male" if word in MALE_PRONOUNS else "female"
                    links.append((occ_name, word, gender))
                    break

        for occ, pronoun, gender in links:
            extracted.append({
                "occupation": occ,
                "pronoun": pronoun,
                "gender": gender,
                "sentence": occ_tok.sent.text.strip(),
            })

    return extracted


def profession_gender_matrix(
    corpus: Union[str, List[str], pd.DataFrame]
) -> pd.DataFrame:
    """Compute profession-gender association counts and percentages across a corpus."""
    nlp = get_spacy_model()
    sentences: List[str] = []

    if isinstance(corpus, str):
        sentences = [s.strip() for s in corpus.split("\n") if s.strip()]
    elif isinstance(corpus, pd.DataFrame):
        text_col = "text" if "text" in corpus.columns else corpus.columns[0]
        sentences = corpus[text_col].dropna().astype(str).tolist()
    else:
        sentences = list(corpus)

    counts: Dict[str, Dict[str, int]] = {}
    for sent in sentences:
        doc = nlp(sent)
        links = link_pronouns_to_occupations(doc)
        for link in links:
            occ = link["occupation"]
            g = link["gender"]
            if occ not in counts:
                counts[occ] = {"male_links": 0, "female_links": 0}
            counts[occ][f"{g}_links"] += 1

    records: List[Dict[str, Any]] = []
    for occ, data in counts.items():
        total = data["male_links"] + data["female_links"]
        pct_male = (data["male_links"] / total * 100.0) if total > 0 else 50.0
        records.append({
            "occupation": occ,
            "male_links": data["male_links"],
            "female_links": data["female_links"],
            "total_links": total,
            "pct_male": round(pct_male, 2),
        })

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["occupation", "male_links", "female_links", "total_links", "pct_male"])
    return df.sort_values(by="total_links", ascending=False).reset_index(drop=True)


def trait_associations(doc: Union[Doc, str]) -> Tuple[CounterType[str], CounterType[str]]:
    """Extract descriptive adjectives and verbs associated with male vs female subjects."""
    if isinstance(doc, str):
        doc = get_spacy_model()(doc)

    male_traits: CounterType[str] = Counter()
    female_traits: CounterType[str] = Counter()

    for token in doc:
        # Check adjectival modifiers directly attached to gendered nouns/pronouns
        word = token.text.lower()
        if word in MALE_PRONOUNS or word in {"man", "boy", "father", "son", "brother", "husband"}:
            for child in token.children:
                if child.pos_ in {"ADJ", "VERB"} and child.dep_ in {"amod", "acl"}:
                    male_traits[child.lemma_.lower()] += 1
            if token.dep_ in {"nsubj", "nsubjpass"}:
                verb = token.head
                if verb.pos_ == "VERB":
                    male_traits[verb.lemma_.lower()] += 1
                for child in verb.children:
                    if child.dep_ in {"acomp", "attr"} and child.pos_ == "ADJ":
                        male_traits[child.lemma_.lower()] += 1

        elif word in FEMALE_PRONOUNS or word in {"woman", "girl", "mother", "daughter", "sister", "wife"}:
            for child in token.children:
                if child.pos_ in {"ADJ", "VERB"} and child.dep_ in {"amod", "acl"}:
                    female_traits[child.lemma_.lower()] += 1
            if token.dep_ in {"nsubj", "nsubjpass"}:
                verb = token.head
                if verb.pos_ == "VERB":
                    female_traits[verb.lemma_.lower()] += 1
                for child in verb.children:
                    if child.dep_ in {"acomp", "attr"} and child.pos_ == "ADJ":
                        female_traits[child.lemma_.lower()] += 1

    return male_traits, female_traits


def plot_profession_gender(
    matrix: pd.DataFrame, save_path: Optional[Path] = None
) -> Path:
    """Generate and save stacked bar chart visualization of profession-gender counts."""
    out_path = save_path or config.PROFESSION_GENDER_PLOT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if matrix.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No Profession-Gender Links Found", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    top_df = matrix.head(15).copy()
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = range(len(top_df))
    ax.barh(y_positions, top_df["male_links"], label="Male Links", color="#4C72B0")
    ax.barh(
        y_positions,
        top_df["female_links"],
        left=top_df["male_links"],
        label="Female Links",
        color="#DD8452",
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(top_df["occupation"].str.title(), fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Syntactic / Co-reference Links", fontsize=11)
    ax.set_title("Profession-Gender Co-occurrence & Syntactic Links", fontsize=13, weight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def validate_on_manual_sample(csv_path: Optional[Path] = None) -> float:
    """Validate extraction precision against labeled benchmark examples."""
    target_csv = csv_path or config.RAW_WINOBIAS_PATH
    df = pd.read_csv(target_csv)

    correct = 0
    total = 0
    nlp = get_spacy_model()

    for _, row in df.iterrows():
        doc = nlp(str(row["text"]))
        links = link_pronouns_to_occupations(doc)
        expected_occ = str(row["occupation"]).lower()
        expected_pronoun = str(row["pronoun"]).lower()
        expected_gender = "male" if expected_pronoun in MALE_PRONOUNS else "female"

        # Check if expected link is present among detected links
        matched = any(
            link["occupation"] == expected_occ and link["gender"] == expected_gender
            for link in links
        )
        if matched:
            correct += 1
        total += 1

    precision = (correct / total) if total > 0 else 0.0
    print(f"Validation on {target_csv.name}: {correct}/{total} correct ({precision:.2%})")
    return round(precision, 4)


if __name__ == "__main__":
    test_sent1 = "The doctor finished his rounds."
    test_sent2 = "The nurse updated her charts."
    print("Test 1 links:", link_pronouns_to_occupations(test_sent1))
    print("Test 2 links:", link_pronouns_to_occupations(test_sent2))
    validate_on_manual_sample()
