"""Text preprocessing module for FairText.

Provides NLTK and spaCy NLP utilities: tokenization, lemmatization,
stopword filtering, sentiment analysis, NER, and dependency parsing.
Uses a singleton spaCy model instance to ensure optimal memory and speed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from src import config

_SPACY_NLP: Optional[Language] = None
_VADER_ANALYZER: Optional[SentimentIntensityAnalyzer] = None
_WORDNET_LEMMATIZER: Optional[WordNetLemmatizer] = None
_STOPWORDS_SET: Optional[set[str]] = None


def get_spacy_model() -> Language:
    """Return the cached singleton spaCy en_core_web_md language model."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = spacy.load(config.SPACY_MODEL_NAME)
    return _SPACY_NLP


def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Return the cached singleton VADER sentiment intensity analyzer."""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


def get_lemmatizer() -> WordNetLemmatizer:
    """Return the cached singleton WordNet lemmatizer."""
    global _WORDNET_LEMMATIZER
    if _WORDNET_LEMMATIZER is None:
        _WORDNET_LEMMATIZER = WordNetLemmatizer()
    return _WORDNET_LEMMATIZER


def get_stopwords() -> set[str]:
    """Return standard English stopwords set."""
    global _STOPWORDS_SET
    if _STOPWORDS_SET is None:
        _STOPWORDS_SET = set(stopwords.words("english"))
    return _STOPWORDS_SET


def clean_text(text: str) -> str:
    """Clean raw text by normalizing whitespace, unicode, and control characters."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"[\r\n\t]+", " ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def analyze(text: str) -> Dict[str, Any]:
    """Perform comprehensive linguistic analysis using spaCy and NLTK.

    Returns:
        dict containing sentences, tokens, lemmas, pos_tags, entities,
        dependencies, sentiment scores, and the cached spaCy Doc.
    """
    cleaned = clean_text(text)
    nlp = get_spacy_model()
    doc: Doc = nlp(cleaned)

    sentences: List[str] = [sent.text.strip() for sent in doc.sents]
    if not sentences and cleaned:
        sentences = sent_tokenize(cleaned)

    tokens: List[str] = [token.text for token in doc]
    lemmas: List[str] = [token.lemma_ for token in doc]
    pos_tags: List[Tuple[str, str]] = [(token.text, token.pos_) for token in doc]

    entities: List[Dict[str, Any]] = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]

    dependencies: List[Dict[str, Any]] = [
        {
            "text": token.text,
            "dep": token.dep_,
            "head": token.head.text,
            "head_pos": token.head.pos_,
            "pos": token.pos_,
        }
        for token in doc
    ]

    vader = get_vader_analyzer()
    sentiment: Dict[str, float] = vader.polarity_scores(cleaned)

    return {
        "text": cleaned,
        "doc": doc,
        "sentences": sentences,
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "entities": entities,
        "dependencies": dependencies,
        "sentiment": sentiment,
    }


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out English stopwords while preserving meaningful terms."""
    stops = get_stopwords()
    return [tok for tok in tokens if tok.lower() not in stops and tok.isalpha()]


if __name__ == "__main__":
    sample = "The doctor finished his rounds while the nurse organized her charts."
    res = analyze(sample)
    print(f"Tokens: {res['tokens']}")
    print(f"Entities: {res['entities']}")
    print(f"Sentiment: {res['sentiment']}")
