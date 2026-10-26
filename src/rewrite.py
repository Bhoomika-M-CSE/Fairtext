"""Text rewriting and debiasing suggestion module for FairText.

Provides rule- and lexicon-based gender neutralization, verb agreement adjustment
(e.g., 'he was' -> 'they were'), suggestion generation, and character span tagging
for visual highlighting in the Streamlit UI.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from spacy.tokens import Doc, Token

from src import config
from src.association import link_pronouns_to_occupations
from src.gender_lexicon import load_lexicons
from src.preprocessing import analyze, get_spacy_model

# Singular-to-plural verb agreement adjustments for 'they' substitutions
VERB_AGREEMENT_MAP: Dict[str, str] = {
    "is": "are",
    "was": "were",
    "has": "have",
    "does": "do",
    "isn't": "aren't",
    "wasn't": "weren't",
    "hasn't": "haven't",
    "doesn't": "don't",
}


def _match_case(source: str, target: str) -> str:
    """Transfer casing pattern from source string to target string."""
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.title()
    return target.lower()


def neutralize_text(text: str) -> str:
    """Rewrite text into gender-neutral phrasing using lexicon mapping and verb agreement."""
    if not text or not text.strip():
        return ""

    _, _, _, _, neutral_map = load_lexicons()
    nlp = get_spacy_model()
    doc = nlp(text)
    out_tokens: List[str] = []

    i = 0
    while i < len(doc):
        token: Token = doc[i]
        token_lower = token.text.lower()
        ws = token.whitespace_

        # Check multi-word compound replacements (e.g. 'police officer')
        replacement: Optional[str] = None
        if token_lower in neutral_map:
            replacement = neutral_map[token_lower]

        if replacement is not None:
            cased_replacement = _match_case(token.text, replacement)
            out_tokens.append(cased_replacement + ws)

            # Check next token for verb agreement if replaced 'he'/'she' with 'they'
            if token_lower in {"he", "she"} and i + 1 < len(doc):
                next_tok = doc[i + 1]
                next_lower = next_tok.text.lower()
                if next_lower in VERB_AGREEMENT_MAP:
                    adj_verb = _match_case(next_tok.text, VERB_AGREEMENT_MAP[next_lower])
                    out_tokens.append(adj_verb + next_tok.whitespace_)
                    i += 2
                    continue
                # 3rd person singular present verb (e.g. "works" -> "work")
                elif next_tok.tag_ == "VBZ" and next_tok.lemma_:
                    adj_verb = _match_case(next_tok.text, next_tok.lemma_)
                    out_tokens.append(adj_verb + next_tok.whitespace_)
                    i += 2
                    continue
            i += 1
        else:
            out_tokens.append(token.text + ws)
            i += 1

    return "".join(out_tokens).strip()


def suggest_rewrites(text: str) -> List[Tuple[str, str, str]]:
    """Generate list of actionable rewrite suggestions: (original_span, suggestion, reason)."""
    _, _, masc_coded, fem_coded, neutral_map = load_lexicons()
    nlp = get_spacy_model()
    doc = nlp(text)
    suggestions: List[Tuple[str, str, str]] = []

    for token in doc:
        w_lower = token.text.lower()
        if w_lower in neutral_map:
            repl = _match_case(token.text, neutral_map[w_lower])
            suggestions.append((
                token.text,
                repl,
                f"Gender-specific terminology. Replace with gender-inclusive '{repl}'."
            ))
        elif w_lower in masc_coded:
            suggestions.append((
                token.text,
                "[consider neutral phrasing]",
                f"Masculine-coded language (Gaucher et al.). May discourage diverse candidates."
            ))
        elif w_lower in fem_coded:
            suggestions.append((
                token.text,
                "[consider neutral phrasing]",
                f"Feminine-coded language (Gaucher et al.). May signal gendered role expectations."
            ))

    return suggestions


def highlight_spans(text: str) -> List[Dict[str, Any]]:
    """Detect character spans categorized for visual UI highlighting."""
    _, _, masc_coded, fem_coded, neutral_map = load_lexicons()
    nlp = get_spacy_model()
    doc = nlp(text)
    spans: List[Dict[str, Any]] = []

    # Identify stereotype profession-pronoun links
    stereo_links = link_pronouns_to_occupations(doc)
    stereo_words = {l["pronoun"].lower() for l in stereo_links} | {l["occupation"].lower() for l in stereo_links}

    for token in doc:
        w_lower = token.text.lower()
        category: Optional[str] = None
        suggestion: Optional[str] = None

        if w_lower in stereo_words:
            category = "stereotype_flag"
            suggestion = "Stereotype association link detected between occupation and gender pronoun"
        elif w_lower in neutral_map:
            category = "gender_term"
            suggestion = f"Gender-specific term. Alternative: {neutral_map[w_lower]}"
        elif w_lower in masc_coded:
            category = "male_coded"
            suggestion = "Masculine-coded descriptor (Gaucher et al. 2011)"
        elif w_lower in fem_coded:
            category = "female_coded"
            suggestion = "Feminine-coded descriptor (Gaucher et al. 2011)"

        if category:
            spans.append({
                "start": token.idx,
                "end": token.idx + len(token.text),
                "text": token.text,
                "category": category,
                "suggestion": suggestion,
            })

    return spans


if __name__ == "__main__":
    sample = "The Chairman stated that he was ready, while the waitress was polite."
    print("Neutralized:", neutralize_text(sample))
    print("Suggestions:", suggest_rewrites(sample))
    print("Spans:", highlight_spans(sample))
