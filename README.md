# Naive Bayes Classifiers

This repository contains two Streamlit apps:

- `restaurent.py` classifies restaurant reviews as positive or negative using
  the `output.csv` dataset.
- `app.py` classifies news articles into five genres using the BBC News dataset.

Both apps use the dependencies in `requirements.txt`.

## Run locally

```bash
python -m venv .venv
```

Activate the virtual environment, then install the dependencies and start the
app you want:

```bash
python -m pip install -r requirements.txt
python -m streamlit run restaurent.py
```

For the news classifier, run `python -m streamlit run app.py` instead.
The restaurant review app requires `output.csv` in the repository root. It
downloads the NLTK English stopwords and WordNet data when first needed, then
trains its model from the CSV.

The news classifier downloads the BBC News dataset and NLTK English stopwords
when needed. It trains and saves its model in `models/`; these generated files
are excluded from Git and recreated automatically when missing.

## Deploy to Streamlit Community Cloud

To deploy the restaurant review app:

1. Push `restaurent.py`, `output.csv`, `requirements.txt`, and the other project
   files to your GitHub repository. Make sure `output.csv` is included; the app
   cannot run without it.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub and choose **Create app**.
3. Select this repository, the branch you pushed, and `restaurent.py` as the
   **Main file path**.
4. Choose **Deploy**. Streamlit installs the packages from `requirements.txt`.

The first prediction downloads the NLTK resources and trains the classifier.
To deploy the news app instead, select `app.py` as the **Main file path**.
