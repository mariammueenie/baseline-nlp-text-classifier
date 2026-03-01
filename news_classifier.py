"""
Real Dataset NLP Mini Project: 20 Newsgroups Text Classification (Baseline)

What this does:
- Loads a real labeled text dataset (20 Newsgroups)
- Trains a classic NLP baseline (TF-IDF + Linear SVM)
- Evaluates performance (accuracy + precision/recall/F1)
- Prints “top indicative terms” per class for interpretability
- Runs a couple sample predictions

Why this matters:
TF-IDF + Linear SVM is a strong traditional baseline for topic classification.
It's fast, surprisingly accurate, and easy to interpret.
"""

# ---------------------------
# Imports (what + why)
# ---------------------------

# Real labeled text dataset (topic-tagged posts)
from sklearn.datasets import fetch_20newsgroups

# Standard utility for splitting data into train/test sets
from sklearn.model_selection import train_test_split

# Turns raw text into numeric vectors using TF-IDF (classic NLP feature extraction)
from sklearn.feature_extraction.text import TfidfVectorizer

# Linear SVM classifier: great baseline for sparse high-dimensional text features
from sklearn.svm import LinearSVC

# Pipeline glues preprocessing + model into one reproducible workflow
from sklearn.pipeline import Pipeline

# Metrics: accuracy + detailed per-class report
from sklearn.metrics import accuracy_score, classification_report


def main():
    # ---------------------------
    # Step 1: Pick valid dataset categories
    # ---------------------------
    # You can’t invent new categories in 20 Newsgroups; you must choose from the built-in list.
    # These are chosen to compare different “styles” of language:
    #   politics (argumentative, abstract)
    #   hockey (event + names + team talk)
    #   AI (technical discussion)
    #   graphics (engineering/CS vocabulary)
    categories = [
        "talk.politics.misc",
        "rec.sport.hockey",
        "sci.space",
        "comp.graphics"
    ]

    # ---------------------------
    # Step 2: Load the dataset
    # ---------------------------
    # remove=("headers","footers","quotes") reduces easy shortcuts (like email headers)
    # so the model learns topic words instead of formatting.
    data = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes")
    )

    # ---------------------------
    # Step 3: Split into train/test
    # ---------------------------
    # stratify keeps class proportions consistent across splits.
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target
    )

    # ---------------------------
    # Step 4: Build the baseline pipeline
    # ---------------------------
    # TF-IDF:
    #   stop_words="english" removes common filler words
    #   ngram_range=(1,2) captures single words + 2-word phrases (more context)
    #   min_df=2 ignores super-rare tokens (less noise)
    #
    # LinearSVC:
    #   strong classic baseline for topic classification on sparse text features
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )),
        ("svm", LinearSVC())
    ])

    # ---------------------------
    # Step 5: Train
    # ---------------------------
    clf.fit(X_train, y_train)

    # ---------------------------
    # Step 6: Predict + Evaluate
    # ---------------------------
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, preds, target_names=data.target_names))

    # ---------------------------
    # Step 7: “What words is it using?” (Interpretability)
    # ---------------------------
    # Linear models expose coefficients per feature; we can show the strongest terms per class.
    tfidf = clf.named_steps["tfidf"]
    svm = clf.named_steps["svm"]
    feature_names = tfidf.get_feature_names_out()

    print("\nTop indicative terms per class:")
    for i, label in enumerate(data.target_names):
        # argsort gives indices of features sorted by weight
        top10 = svm.coef_[i].argsort()[-10:][::-1]
        terms = ", ".join(feature_names[j] for j in top10)
        print(f"- {label}: {terms}")

    # ---------------------------
    # Step 8: Quick demo predictions
    # ---------------------------
    samples = [
        # politics
        "The election debate focused on policy, government spending, and media coverage.",
        # hockey
        "The goalie had an unreal save and the team won in overtime after a tough third period.",
        # space
        "The spacecraft entered orbit after a successful launch and deployed its satellite payload."
        # graphics
        "The rendering pipeline uses shaders, texture mapping, and anti-aliasing to improve output quality."
    ]

    sample_preds = clf.predict(samples)

    print("\nSample predictions:")
    for text, p in zip(samples, sample_preds):
        print(f"- Predicted: {data.target_names[p]} | Text: {text}")


if __name__ == "__main__":
    main()