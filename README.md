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

1. Push this project to a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app
   from that repository and select the branch and `app.py` as the main file.
3. Deploy. Streamlit Cloud installs the packages listed in `requirements.txt`.
