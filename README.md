# Baseline NLP Text Classifier (20 Newsgroups)

A small experiment in classical NLP topic classification using a real labeled dataset.

This project implements a traditional machine learning baseline:
**TF-IDF feature extraction + Linear Support Vector Machine (SVM)**.

The goal is to evaluate how well a simple, interpretable model can separate different domains of text.

---

## Dataset

Uses the **20 Newsgroups** dataset (scikit-learn).

The model classifies posts into four categories:

- `talk.politics.misc`
- `rec.sport.hockey`
- `sci.space`
- `comp.graphics`

These categories were selected to compare distinct language styles:
- Political discussion (argumentative, abstract)
- Sports discussion (event-driven vocabulary)
- Space/science discourse (technical terminology)
- Graphics/computing topics (engineering vocabulary)

---

## Method

Pipeline:

- **TF-IDF Vectorization**
  - English stopword removal
  - Unigrams + bigrams
  - Rare-term filtering (`min_df=2`)
- **Linear SVM (LinearSVC)**
  - Strong classical baseline for high-dimensional sparse text data

Evaluation metrics:
- Accuracy
- Precision / Recall / F1-score
- Top indicative terms per class (for interpretability)

---

## Results

Accuracy: **0.907**

Observations:
- `comp.graphics` and `rec.sport.hockey` are the most separable classes.
- `sci.space` shows slightly lower precision, likely due to overlapping technical vocabulary.
- Political discussions contain broader language, increasing classification ambiguity.

The model distinguishes domain-specific vocabulary effectively despite its simplicity.

---

## Sample Output

Below is a sample run of the classifier including metrics and top indicative terms.

![Model Results](images/results-output.png)

---

## Run

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python news_classifier.py