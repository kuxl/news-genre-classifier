"""Streamlit interface for the restaurant review model in the analysis notebook.

Run with: ``streamlit run restaurent.py``
Keep ``output.csv`` in the same folder.
"""
import re
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


DATA_PATH = Path(__file__).parent / "output.csv"


def load_nltk_resources() -> tuple[set[str], WordNetLemmatizer]:
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        if not nltk.download("stopwords", quiet=True):
            raise RuntimeError("Could not download the NLTK stopwords corpus.")
        stop_words = set(stopwords.words("english"))

    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("reviews")
    except LookupError:
        if not nltk.download("wordnet", quiet=True):
            raise RuntimeError("Could not download the NLTK wordnet corpus.")
        lemmatizer.lemmatize("reviews")

    return stop_words, lemmatizer


def preprocess_review(
    review: str, stop_words: set[str], lemmatizer: WordNetLemmatizer
) -> str:
    text = re.sub(r"[^\w\s]", "", str(review).lower())
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )


@st.cache_resource
def load_model():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Keep output.csv beside restaurent.py."
        )

    data = pd.read_csv(DATA_PATH)
    columns = {str(column).strip().lower(): column for column in data.columns}
    if not {"review", "liked"}.issubset(columns):
        raise ValueError("output.csv must contain Review and Liked columns.")

    data = data.rename(
        columns={columns["review"]: "Review", columns["liked"]: "Liked"}
    ).dropna(subset=["Review", "Liked"])
    labels = pd.to_numeric(data["Liked"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Liked must contain only binary labels 0 and 1.")
    if labels.nunique() != 2:
        raise ValueError("Training data must include both sentiment classes.")

    stop_words, lemmatizer = load_nltk_resources()
    reviews = [
        preprocess_review(review, stop_words, lemmatizer)
        for review in data["Review"]
    ]
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(reviews)
    X_train, _, y_train, _ = train_test_split(
        features,
        labels.astype(int).to_numpy(),
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    model = BernoulliNB()
    model.fit(X_train, y_train)
    return vectorizer, model, stop_words, lemmatizer


def main() -> None:
    st.set_page_config(
        page_title="Review Sentiment Classifier",
        page_icon="🍽️",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 560px; padding-top: 5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🍽️ Review Sentiment Classifier")
    review = st.text_area("Enter your review:", height=100)

    if st.button("Predict"):
        if not review.strip():
            st.warning("Enter a review first.")
            return

        try:
            vectorizer, model, stop_words, lemmatizer = load_model()
            cleaned_review = preprocess_review(review, stop_words, lemmatizer)
            prediction = model.predict(vectorizer.transform([cleaned_review]))[0]
            sentiment = "Positive" if prediction == 1 else "Negative"
            show_result = st.success if prediction == 1 else st.error
            show_result(f"Sentiment: **{sentiment}**")
        except (
            FileNotFoundError,
            pd.errors.ParserError,
            RuntimeError,
            ValueError,
        ) as error:
            st.error(f"Could not predict sentiment: {error}")


if __name__ == "__main__":
    main()
