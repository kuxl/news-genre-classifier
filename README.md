# News Genre Classifier

A Streamlit app that classifies news text into business, entertainment, politics,
sport, or technology using Naive Bayes models trained on the BBC News dataset.
It compares Multinomial Naive Bayes and Gaussian Naive Bayes, then uses the
better-performing model.

## Run locally

```bash
python -m venv .venv
```

Activate the virtual environment, then install the dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app downloads the BBC News dataset and NLTK English stopwords when needed.
On its first start, it trains both models and saves the selected model and
metrics in `models/`. Those generated files are excluded from Git; they are
recreated automatically when the app starts without them.

## Deploy to Streamlit Community Cloud

This repository is ready to deploy from
[Streamlit Community Cloud](https://share.streamlit.io/):

1. Push the project to the `main` branch of
   [`kuxl/news-genre-classifier`](https://github.com/kuxl/news-genre-classifier).
2. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
3. Select repository `kuxl/news-genre-classifier`, branch `main`, and main file
   `app.py`.
4. Choose **Deploy**. Streamlit installs the packages from `requirements.txt`.

The first launch downloads the BBC News dataset and NLTK English stopwords,
then trains and saves the models in `models/`. These generated model files are
not committed; if the deployment starts without them, it trains them again.
