# FairText Installation and Run Commands

## 1. Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'names', 'vader_lexicon', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'])"
```

## 2. Dataset Preparation

```bash
python -m src.data_loader
```

## 3. Linguistic Profiling & Syntactic Association Tests

```bash
python -m src.preprocessing
python -m src.gender_lexicon
python -m src.association
```

## 4. Embedding Bias Audit & Bolukbasi Hard Debiasing

```bash
python -m src.embeddings_audit
python -m src.debias
```

## 5. Model Training (TF BiLSTM & PyTorch DistilBERT)

```bash
python -m src.classifier_tf
python -m src.classifier_torch
```

## 6. Text Rewriting & Bias Scoring Verification

```bash
python -m src.rewrite
python -m src.scoring
```

## 7. Full Benchmark Evaluation & Error Analysis

```bash
python -m src.evaluate
```

## 8. Launch Streamlit Web Application

```bash
streamlit run app/app.py
```

## 9. Run Pytest Test Suite

```bash
pytest tests/ -v
```
