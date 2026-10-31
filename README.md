# Machine Learning Streamlit Apps

This repository contains three Streamlit apps:

- `restaurent.py` classifies restaurant reviews as positive or negative using
  the `output.csv` dataset.
- `app.py` classifies news articles into five genres using the BBC News dataset.
- `daimond.py` predicts diamond prices with a K-Nearest Neighbors regression
  pipeline using `diamonds.csv`.

The apps use the dependencies in `requirements.txt`.

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
For diamond price prediction, place `diamonds.csv` in the repository root and
run `python -m streamlit run daimond.py`. The app uses `price` as its target,
preprocesses numeric and categorical features, and provides a single-diamond
price estimator. The fitted pipeline is saved to
`models/diamond_knn_pipeline.pkl`.
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
For diamond price prediction, select `daimond.py` and include `diamonds.csv`
in the repository root.
