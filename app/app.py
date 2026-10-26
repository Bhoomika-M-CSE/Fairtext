"""FairText Streamlit Dashboard Application.

Interactive linguistic bias and stereotype auditor featuring:
- Real-time text analysis and file upload (.txt)
- 0-100 bias score gauge and component breakdown
- Color-coded span highlighting (blue: male-coded, pink: female-coded, orange: stereotype, purple: gender terms)
- Side-by-side neutral rewrite comparison
- Profession-gender syntactic association and trait tables
- Dual text comparison tab
- Pre-trained vs debiased word embedding audit tab
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="FairText - Linguistic Bias Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src import config
from src.association import (
    link_pronouns_to_occupations,
    plot_profession_gender,
    profession_gender_matrix,
    trait_associations,
)
from src.classifier_tf import load_tf_classifier, load_tf_vocab, predict_tf
from src.classifier_torch import load_torch_classifier, predict_torch
from src.debias import before_after_report
from src.gender_lexicon import count_gendered_terms, gender_coded_share
from src.preprocessing import analyze
from src.rewrite import highlight_spans, neutralize_text, suggest_rewrites
from src.scoring import bias_score


@st.cache_resource
def get_torch_models():
    """Cache PyTorch DistilBERT model and tokenizer."""
    return load_torch_classifier()


@st.cache_resource
def get_tf_models():
    """Cache TensorFlow BiLSTM model and vocab."""
    return load_tf_classifier(), load_tf_vocab()


def render_highlighted_html(text: str, spans: List[Dict[str, Any]]) -> str:
    """Generate color-coded HTML string representing annotated linguistic spans."""
    if not spans:
        return f"<div style='line-height:1.8; font-size:16px;'>{html.escape(text)}</div>"

    # Color palette
    category_colors = {
        "male_coded": ("#D0E1FD", "#1A56DB", "Masculine-Coded"),
        "female_coded": ("#FCE8F3", "#C71F7E", "Feminine-Coded"),
        "stereotype_flag": ("#FEEBC8", "#C05621", "Stereotype Association"),
        "gender_term": ("#E9D8FD", "#6B46C1", "Gendered Term"),
    }

    # Sort spans by start offset
    sorted_spans = sorted(spans, key=lambda s: s["start"])
    html_parts: List[str] = []
    last_idx = 0

    for span in sorted_spans:
        start = span["start"]
        end = span["end"]
        if start < last_idx:
            continue
        # Plain text segment before the span
        html_parts.append(html.escape(text[last_idx:start]))
        bg, text_c, label = category_colors.get(span["category"], ("#EDF2F7", "#4A5568", "Flag"))
        tip = html.escape(span.get("suggestion", ""))
        span_html = (
            f"<mark style='background-color:{bg}; color:{text_c}; padding:2px 6px; "
            f"border-radius:4px; font-weight:600; border:1px solid {text_c};' "
            f"title='{tip}'>{html.escape(text[start:end])} "
            f"<span style='font-size:10px; text-transform:uppercase;'>[{label}]</span></mark>"
        )
        html_parts.append(span_html)
        last_idx = end

    html_parts.append(html.escape(text[last_idx:]))
    return f"<div style='line-height:2.0; font-size:16px; background:#fafafa; padding:15px; border-radius:8px; border:1px solid #e2e8f0;'>{''.join(html_parts)}</div>"


def render_score_badge(score: float, is_flagged: bool) -> None:
    """Render overall bias score card with dynamic status colors."""
    if score < 20:
        status_color = "#38A169"
        status_text = "Low Bias / Neutral"
    elif score < 50:
        status_color = "#DD6B20"
        status_text = "Moderate Linguistic Bias"
    else:
        status_color = "#E53E3E"
        status_text = "High Linguistic / Stereotype Bias"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"""
            <div style="background-color:#F7FAFC; border:2px solid {status_color}; border-radius:12px; padding:20px; text-align:center;">
                <div style="font-size:14px; text-transform:uppercase; color:#718096; font-weight:700;">Overall Bias Score</div>
                <div style="font-size:48px; font-weight:800; color:{status_color}; margin:8px 0;">{score:.1f}<span style="font-size:22px; color:#A0AEC0;">/100</span></div>
                <div style="display:inline-block; padding:4px 12px; border-radius:20px; background-color:{status_color}; color:white; font-size:13px; font-weight:600;">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Streamlit application UI entrypoint."""
    st.title("⚖️ FairText: Linguistic Bias & Stereotype Auditor")
    st.markdown("An NLP system auditing gender bias, syntactic stereotypes, and embedding representations.")

    # -----------------------------------------------------------------------
    # Sidebar Controls
    # -----------------------------------------------------------------------
    st.sidebar.header("Configuration")
    clf_choice = st.sidebar.selectbox(
        "Active Bias Classifier",
        ["PyTorch DistilBERT", "TensorFlow BiLSTM"],
        index=0,
    )
    clf_key = "torch" if "DistilBERT" in clf_choice else "tf"

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Sections")
    show_spans = st.sidebar.checkbox("Highlighted Spans", value=True)
    show_breakdown = st.sidebar.checkbox("Score Component Breakdown", value=True)
    show_rewrites = st.sidebar.checkbox("Neutral Rewrite Suggestions", value=True)
    show_associations = st.sidebar.checkbox("Profession-Gender Links", value=True)
    show_traits = st.sidebar.checkbox("Gendered Trait Associations", value=True)

    tab_audit, tab_compare, tab_embeddings = st.tabs([
        "🔍 Document Auditor", "⚖️ Compare Texts", "📊 Embedding Audit (WEAT)"
    ])

    # Pre-load cached models
    if clf_key == "torch":
        get_torch_models()
    else:
        get_tf_models()

    # -----------------------------------------------------------------------
    # Tab 1: Single Document Auditor
    # -----------------------------------------------------------------------
    with tab_audit:
        st.subheader("Input Text for Linguistic Audit")
        col_input1, col_input2 = st.columns([3, 1])

        with col_input1:
            default_sample = (
                "The competitive chairman demanded that his polite receptionist follow his exact orders. "
                "Meanwhile, the doctor finished his rounds while the nurse organized her patient charts."
            )
            input_text = st.text_area(
                "Enter text or job description to audit:",
                value=default_sample,
                height=130,
            )

        with col_input2:
            uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
            if uploaded_file is not None:
                input_text = uploaded_file.read().decode("utf-8")

        if st.button("Run Audit", type="primary") and input_text.strip():
            score, breakdown = bias_score(input_text, classifier_type=clf_key)
            render_score_badge(score, breakdown["is_flagged"])

            if show_breakdown:
                st.markdown("### Component Breakdown")
                b_cols = st.columns(4)
                b_cols[0].metric("Gender Ratio Imbalance", f"{breakdown['gender_ratio_component']:.1f} / 20")
                b_cols[1].metric("Coded Word Share", f"{breakdown['coded_language_component']:.1f} / 25")
                b_cols[2].metric("Stereotype Links", f"{breakdown['stereotype_links_component']:.1f} / 30")
                b_cols[3].metric("Classifier Prob", f"{breakdown['classifier_prob']:.2%}")

            if show_spans:
                st.markdown("### Annotated Text Highlighting")
                spans = highlight_spans(input_text)
                st.markdown(render_highlighted_html(input_text, spans), unsafe_allow_html=True)

            if show_rewrites:
                st.markdown("### Suggested Neutral Phrasing")
                neutral = neutralize_text(input_text)
                col_r1, col_r2 = st.columns(2)
                col_r1.text_area("Original Text", value=input_text, height=110, disabled=True)
                col_r2.text_area("Debiased Neutral Rewrite", value=neutral, height=110, disabled=True)

                suggestions = suggest_rewrites(input_text)
                if suggestions:
                    st.markdown("**Actionable Edits:**")
                    s_df = pd.DataFrame(suggestions, columns=["Original Term", "Suggested Alternative", "Linguistic Rationale"])
                    st.dataframe(s_df, use_container_width=True)

            if show_associations:
                st.markdown("### Profession-Gender Syntactic Associations")
                parsed_doc = analyze(input_text)["doc"]
                links = link_pronouns_to_occupations(parsed_doc)
                if links:
                    matrix = profession_gender_matrix(input_text)
                    st.dataframe(matrix, use_container_width=True)
                    plot_path = plot_profession_gender(matrix)
                    st.image(str(plot_path), use_container_width=True)
                else:
                    st.info("No syntactic pronoun-profession links detected in the input text.")

            if show_traits:
                st.markdown("### Trait Associations (Adjectives & Verbs)")
                parsed_doc = analyze(input_text)["doc"]
                m_traits, f_traits = trait_associations(parsed_doc)
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("**Male Subject Descriptors / Actions:**")
                    st.write(dict(m_traits.most_common(10)) if m_traits else "None detected")
                with col_t2:
                    st.markdown("**Female Subject Descriptors / Actions:**")
                    st.write(dict(f_traits.most_common(10)) if f_traits else "None detected")

    # -----------------------------------------------------------------------
    # Tab 2: Compare Two Texts
    # -----------------------------------------------------------------------
    with tab_compare:
        st.subheader("Comparative Document Audit")
        c1, c2 = st.columns(2)

        with c1:
            text_a = st.text_area(
                "Document A (e.g. Masculine-Coded Job Description):",
                value="We are seeking an aggressive and dominant leader who can fearlessly outperform rivals.",
                height=150,
            )
        with c2:
            text_b = st.text_area(
                "Document B (e.g. Inclusive Job Description):",
                value="We are seeking a collaborative software engineer who values transparent teamwork.",
                height=150,
            )

        if st.button("Compare Texts", type="secondary"):
            score_a, b_a = bias_score(text_a, classifier_type=clf_key)
            score_b, b_b = bias_score(text_b, classifier_type=clf_key)

            res_cols = st.columns(2)
            with res_cols[0]:
                st.markdown(f"#### Document A Score: **{score_a:.1f}/100**")
                st.metric("Classifier Probability", f"{b_a['classifier_prob']:.2%}")
                st.metric("Coded Words Component", f"{b_a['coded_language_component']:.1f}")
                st.metric("Stereotype Links", b_a["stereotypical_links_count"])

            with res_cols[1]:
                st.markdown(f"#### Document B Score: **{score_b:.1f}/100**")
                st.metric("Classifier Probability", f"{b_b['classifier_prob']:.2%}")
                st.metric("Coded Words Component", f"{b_b['coded_language_component']:.1f}")
                st.metric("Stereotype Links", b_b["stereotypical_links_count"])

    # -----------------------------------------------------------------------
    # Tab 3: Embedding Audit (WEAT & Hard Debiasing)
    # -----------------------------------------------------------------------
    with tab_embeddings:
        st.subheader("Word Embedding Association Test (WEAT) & Hard Debiasing Audit")
        st.markdown(
            "Audits gender bias in **GloVe 100d** word vectors and validates mitigation via **Bolukbasi et al. (2016)** hard debiasing."
        )

        if config.DEBIAS_REPORT_PATH.exists():
            report_df = pd.read_csv(config.DEBIAS_REPORT_PATH)
            st.markdown("### Before vs. After Debiasing Report")
            st.dataframe(report_df, use_container_width=True)
        else:
            if st.button("Generate WEAT Audit Report"):
                with st.spinner("Calculating WEAT effect sizes and Bolukbasi debiasing..."):
                    report_df = before_after_report()
                    st.dataframe(report_df, use_container_width=True)

        if config.PROJECTION_PLOT_PATH.exists():
            st.markdown("### Profession Projections on Gender Subspace")
            st.image(str(config.PROJECTION_PLOT_PATH), use_container_width=True)


if __name__ == "__main__":
    main()
