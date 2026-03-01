# Baseline NLP Text Classifier (20 Newsgroups)

Tiny project w/ real dataset, surprisingly solid results.

This repo is a quick baseline experiment in **topic classification** using the classic combo:
**TF-IDF → Linear SVM**.
 
A *simple* NLP model that trains fast and is easy to inspect.

## What it does
Loads a real labeled dataset: **20 Newsgroups**
Classifies posts into 4 topics:
    `talk.politics.misc` (politics)
    `rec.sport.hockey` (hockey)
    `comp.ai` (AI)
    `comp.graphics` (graphics)
Trains a baseline model using:
    **TF-IDF** (with unigrams + bigrams)
    **Linear SVM**
Prints:
    accuracy
    precision / recall / F1
    top “most telling” words per category (fun + interpretability)

## Why these categories?
I picked them because the writing styles are different:
    politics can be abstract and argumentative
    hockey is event + team + player language
    AI is technical and research-y
    graphics includes engineering/CS vocabulary

It’s a diverse range, to see how a simple model separates domains.

## Results

Accuracy: 0.907

Observations:
- comp.graphics and hockey are easiest to separate
- sci.space shows slightly lower precision, likely due to overlapping technical vocabulary
- Political category contains broader and more abstract language, increasing ambiguity

Model appears to distinguish domain-specific vocabulary effectively.

## Run it
```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    pip install scikit-learn
    python news_classifier.py