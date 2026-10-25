"""
News Genre Classifier - Naive Bayes text classification on the BBC News dataset.

Run:
    pip install streamlit pandas numpy scikit-learn joblib nltk
    streamlit run app.py

On the first run the app trains Multinomial NB and Gaussian NB, compares their test
performance, selects the better model and saves it (pickle and joblib) in a "models"
folder. Subsequent runs load the saved model.
"""
import html
import json
import pickle
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/ronaldleonardo/Data_Transformer_Datasets/"
    "refs/heads/main/bbc_text_cls.csv"
)
MODEL_DIR = Path(__file__).parent / "models"

COLOR = {
    "business": "#1d4ed8",
    "entertainment": "#be185d",
    "politics": "#6d28d9",
    "sport": "#15803d",
    "tech": "#c2410c",
}
ICON = {
    "business": "💼",
    "entertainment": "🎬",
    "politics": "🏛️",
    "sport": "⚽",
    "tech": "💻",
}
EXAMPLES = {
    "sport": "Chelsea beat Arsenal 2-1 in the league match on Saturday, with the coach praising his players after the victory. England captain scored in the second half and the club moved to the top of the table as the season reaches its final stage.",
    "business": "Shares in the bank rose after it reported higher quarterly profits, with analysts saying strong sales and growth in the economy boosted the firm. Oil prices and the dollar fell, while investors watched interest rates and exports from China.",
    "entertainment": "The film won best actor and best actress at the awards ceremony in Hollywood, and the director thanked the stars of the movie. The singer's new album went to number one in the chart, and the band will play a concert at the festival.",
    "politics": "The prime minister told MPs that the government will bring in new immigration rules, as the Conservatives and Labour clashed in the Commons. The chancellor said the party would not raise tax before the election.",
    "tech": "Microsoft has released new software for Windows users, with the company saying the programs will protect computers from spam and viruses. Broadband and mobile phone users can also download the update online.",
}
NEEDED_KEYS = {"best_model", "accuracy", "classes", "n_train", "n_test", "class_counts",
               "avg_words", "top_words", "total", "report", "confusion"}


# ----------------------------------------------------------------------------
# 1. Text preprocessing
# ----------------------------------------------------------------------------
@st.cache_resource
def get_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


sw = get_stopwords()


def clean(text):
    """Lowercase, strip non-alphabetic characters, remove stop words."""
    text = text.lower()
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return " ".join(w for w in text.split() if w not in sw)


# ----------------------------------------------------------------------------
# 2. Training: compare models, select the best, persist with pickle + joblib
# ----------------------------------------------------------------------------
def train_and_save():
    df = pd.read_csv(DATA_URL).dropna(subset=["text", "labels"])
    df["cleaned_text"] = df["text"].apply(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["labels"], test_size=0.2, random_state=42, stratify=df["labels"]
    )

    tfidf = TfidfVectorizer()
    X_train_t = tfidf.fit_transform(X_train).toarray()
    X_test_t = tfidf.transform(X_test).toarray()

    mnb = MultinomialNB().fit(X_train_t, y_train)
    gnb = GaussianNB().fit(X_train_t, y_train)

    accuracy = {
        "MultinomialNB": float(accuracy_score(y_test, mnb.predict(X_test_t))),
        "GaussianNB": float(accuracy_score(y_test, gnb.predict(X_test_t))),
    }
    best_name = max(accuracy, key=accuracy.get)
    best_model = mnb if best_name == "MultinomialNB" else gnb
    classes = [str(c) for c in best_model.classes_]

    y_pred = best_model.predict(X_test_t)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report = {k: {m: float(v) for m, v in vals.items()}
              for k, vals in report.items() if isinstance(vals, dict)}
    confusion = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    names = np.array(tfidf.get_feature_names_out())
    overall = X_train_t.mean(axis=0)
    top_words = {}
    for c in classes:
        diff = X_train_t[(y_train == c).values].mean(axis=0) - overall
        top_words[c] = [str(w) for w in names[np.argsort(diff)[::-1][:18]]]

    df["n_words"] = df["text"].str.split().str.len()
    info = {
        "best_model": best_name,
        "accuracy": accuracy,
        "classes": classes,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "total": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["labels"].value_counts().items()},
        "avg_words": {str(k): float(v) for k, v in df.groupby("labels")["n_words"].mean().round(0).items()},
        "top_words": top_words,
        "report": report,
        "confusion": confusion,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    joblib.dump(best_model, MODEL_DIR / "model.joblib")
    joblib.dump(tfidf, MODEL_DIR / "tfidf.joblib")
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)


@st.cache_resource
def load_model():
    info_path = MODEL_DIR / "model_info.json"
    ok = info_path.exists() and (MODEL_DIR / "model.joblib").exists() and (MODEL_DIR / "tfidf.joblib").exists()
    if ok:
        with open(info_path) as f:
            ok = NEEDED_KEYS.issubset(json.load(f))  # outdated files -> retrain
    if not ok:
        with st.spinner("⏳ Training models. This takes about a minute on first run..."):
            train_and_save()
    model = joblib.load(MODEL_DIR / "model.joblib")
    tfidf = joblib.load(MODEL_DIR / "tfidf.joblib")
    with open(info_path) as f:
        info = json.load(f)
    return model, tfidf, info


# ----------------------------------------------------------------------------
# 3. Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="News Genre Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none;}
    .block-container {padding-top: 2rem; max-width: 1200px;}

    .header {padding: 2rem 2.2rem; border-radius: 14px; color: #f8fafc;
             background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
             box-shadow: 0 6px 20px rgba(15,23,42,.25);}
    .header h1 {margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.015em;}
    .header p {margin: .5rem 0 0 0; color: #cbd5e1; font-size: 1rem; max-width: 780px; line-height: 1.55;}
    .badges {margin-top: 1rem;}
    .badge {display: inline-block; padding: .22rem .75rem; margin: 0 .4rem .3rem 0; border-radius: 999px;
            font-size: .8rem; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);}

    .kpi {padding: 1rem 1.2rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.28);
          background: rgba(128,128,128,.05); box-shadow: 0 1px 3px rgba(0,0,0,.05);}
    .kpi .label {font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65;}
    .kpi .value {font-size: 1.5rem; font-weight: 700; margin-top: .15rem;}

    .section-title {font-size: .82rem; text-transform: uppercase; letter-spacing: .07em;
                    opacity: .75; font-weight: 700; margin: 1.3rem 0 .5rem 0;}

    .result {padding: 1.2rem 1.5rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.28);
             border-left-width: 7px; background: rgba(128,128,128,.05); margin: .7rem 0 1rem 0;
             box-shadow: 0 1px 4px rgba(0,0,0,.06);}
    .result .caption {font-size: .75rem; text-transform: uppercase; letter-spacing: .07em; opacity: .65;}
    .result .genre {font-size: 1.9rem; font-weight: 700; margin: .15rem 0;}
    .result .meta {font-size: .92rem; opacity: .85;}

    .chip {display: inline-block; padding: .25rem .8rem; margin: .18rem .25rem .18rem 0;
           border-radius: 999px; font-size: .88rem; border: 1px solid;}
    .articlebox {padding: 1.1rem 1.3rem; border-radius: 12px; line-height: 1.7;
                 border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.06);}
    .footer {text-align: center; opacity: .55; font-size: .8rem; margin-top: 2.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    model, tfidf, info = load_model()
except Exception as e:
    st.error(f"❌ Could not load or train the model: {e}")
    st.stop()

best = info["best_model"]
classes = info["classes"]


# ----------------------------------------------------------------------------
# 4. Helpers
# ----------------------------------------------------------------------------
def pretty(c):
    return c.title()


def labeled(c):
    """Category name with its emoji."""
    return f"{ICON.get(c, '📰')} {c.title()}"


def kpi(col, label, value):
    col.markdown(
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div></div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def predict(texts):
    X = tfidf.transform([clean(t) for t in texts]).toarray()
    return model.predict_proba(X)


def word_scores(text, class_index, top_n=10):
    """Words in the text that pushed the prediction towards the chosen genre."""
    if not hasattr(model, "feature_log_prob_"):
        return {}
    lp = model.feature_log_prob_
    diff = lp[class_index] - np.delete(lp, class_index, axis=0).mean(axis=0)
    X = tfidf.transform([clean(text)])
    names = tfidf.get_feature_names_out()
    scores = {str(names[i]): float(v * diff[i]) for i, v in zip(X.indices, X.data)}
    scores = {w: s for w, s in scores.items() if s > 0}
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n])


def highlight(text, words, color):
    out, last = [], 0
    for mt in re.finditer(r"[A-Za-z]+", text):
        out.append(html.escape(text[last:mt.start()]))
        w = mt.group(0)
        if w.lower() in words:
            out.append(
                f'<mark style="background:{color}2e;color:inherit;border-bottom:2px solid {color};'
                f'padding:0 3px;border-radius:3px">{html.escape(w)}</mark>'
            )
        else:
            out.append(html.escape(w))
        last = mt.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


def confidence_label(conf):
    if conf >= 0.85:
        return "🟢 High confidence"
    if conf >= 0.6:
        return "🟡 Moderate confidence"
    return "🔴 Low confidence"


# ----------------------------------------------------------------------------
# 5. Session state and callbacks
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "text" not in st.session_state:
    st.session_state.text = ""


def set_example(genre):
    st.session_state.text = EXAMPLES[genre]


def clear_text():
    st.session_state.text = ""


# ----------------------------------------------------------------------------
# 6. Layout
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header"><h1>📰 News Genre Classifier</h1>
    <p>Automated categorisation of news text into business, entertainment, politics,
    sport and technology using a Naive Bayes model trained on the BBC News dataset.</p>
    <div class="badges">
      <span class="badge">💼 Business</span><span class="badge">🎬 Entertainment</span>
      <span class="badge">🏛️ Politics</span><span class="badge">⚽ Sport</span>
      <span class="badge">💻 Tech</span>
    </div></div>
    """,
    unsafe_allow_html=True,
)
st.write("")

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "🤖 Active model", best.replace("NB", " NB"))
kpi(k2, "🎯 Test accuracy", f"{info['accuracy'][best]:.2%}")
kpi(k3, "🗂️ Categories", len(classes))
kpi(k4, "📚 Training articles", f"{info['n_train']:,}")

st.write("")
tab1, tab2, tab3 = st.tabs(["🔍 Classify", "📦 Batch analysis", "📊 Genre insights"])

# ---------------- Classify ----------------
with tab1:
    section("✨ Sample inputs")
    cols = st.columns(len(classes))
    for col, g in zip(cols, classes):
        col.button(labeled(g), key=f"ex_{g}", on_click=set_example, args=(g,), use_container_width=True)

    text = st.text_area("📝 Article text", key="text", height=180,
                        placeholder="Paste a news headline or article...")
    b1, b2, _ = st.columns([1, 1, 6])
    go = b1.button("🚀 Classify", type="primary", key="predict_btn", use_container_width=True)
    b2.button("🧹 Clear", key="clear_btn", on_click=clear_text, use_container_width=True)

    if go:
        if len(text.split()) < 3:
            st.warning("⚠️ Please enter at least a few words.")
        else:
            proba = predict([text])[0]
            order = np.argsort(proba)[::-1]
            i = int(order[0])
            label, conf = classes[i], float(proba[i])
            color = COLOR.get(label, "#334155")

            st.markdown(
                f"""<div class="result" style="border-left-color:{color}">
                <div class="caption">🏷️ Predicted category</div>
                <div class="genre" style="color:{color}">{html.escape(labeled(label))}</div>
                <div class="meta">Confidence {conf:.1%} &nbsp;|&nbsp; {confidence_label(conf)}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            s1, s2, s3 = st.columns(3)
            kpi(s1, "🔤 Words", len(text.split()))
            kpi(s2, "🔡 Characters", len(text))
            kpi(s3, "⏱️ Est. reading time", f"{max(1, round(len(text.split()) / 200 * 60))} sec")

            left, right = st.columns(2)
            with left:
                section("🏆 Top 3 categories")
                for j in order[:3]:
                    st.progress(float(proba[j]), text=f"{labeled(classes[j])}: {proba[j]:.1%}")
                section("📈 Probability distribution")
                st.bar_chart(pd.DataFrame({"Probability": proba}, index=[pretty(c) for c in classes]))
            with right:
                scores = word_scores(text, i)
                section(f"🔑 Key terms supporting {pretty(label)}")
                if scores:
                    st.bar_chart(pd.Series(scores, name="Influence"))
                else:
                    st.caption("No category-specific terms were found in this text.")

            section("🖍️ Annotated text")
            st.markdown(f'<div class="articlebox">{highlight(text, set(scores), color)}</div>',
                        unsafe_allow_html=True)

            st.session_state.history.insert(
                0, {"Text": text[:80] + ("..." if len(text) > 80 else ""),
                    "Prediction": label, "Confidence": f"{conf:.1%}"})

    if st.session_state.history:
        st.divider()
        section("🕘 Prediction history")
        h = pd.DataFrame(st.session_state.history)
        hc1, hc2 = st.columns([3, 2])
        with hc1:
            st.dataframe(h, use_container_width=True, hide_index=True)
        with hc2:
            st.bar_chart(h["Prediction"].value_counts())
        d1, d2, _ = st.columns([2, 2, 6])
        d1.download_button("📥 Export history (CSV)", h.to_csv(index=False), "history.csv", "text/csv")
        if d2.button("🗑️ Clear history", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

# ---------------- Batch analysis ----------------
with tab2:
    section("📦 Bulk classification")
    mode = st.radio("Input method", ["✍️ Paste text (one article per line)", "📄 Upload CSV"],
                    horizontal=True)
    texts = []
    if mode == "📄 Upload CSV":
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            d = pd.read_csv(up)
            col = st.selectbox("Text column", d.columns)
            texts = d[col].fillna("").astype(str).tolist()
            st.caption(f"✅ {len(texts):,} rows loaded.")
    else:
        raw = st.text_area("Articles or headlines (one per line)", height=160)
        texts = [t for t in raw.splitlines() if t.strip()]

    if st.button("🚀 Run classification", type="primary", disabled=not texts, key="batch_btn"):
        proba = predict(texts)
        idx = proba.argmax(axis=1)
        out = pd.DataFrame({
            "Text": [t[:100] for t in texts],
            "Prediction": [classes[i] for i in idx],
            "Confidence": [round(float(proba[r, i]), 3) for r, i in enumerate(idx)],
        })
        r1, r2 = st.columns([3, 2])
        with r1:
            st.dataframe(out, use_container_width=True, hide_index=True)
        with r2:
            section("🗂️ Category distribution")
            st.bar_chart(out["Prediction"].value_counts())
        st.download_button("📥 Export results (CSV)", out.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------- Genre insights ----------------
with tab3:
    section("🧭 Category profile")
    g = st.radio("Category", classes, format_func=labeled, horizontal=True, key="explore_genre",
                 label_visibility="collapsed")
    gc = COLOR.get(g, "#334155")
    e1, e2, e3 = st.columns(3)
    kpi(e1, "📰 Articles in dataset", f"{info['class_counts'].get(g, 0):,}")
    kpi(e2, "📏 Average length", f"{int(info['avg_words'].get(g, 0))} words")
    kpi(e3, "🥧 Share of dataset", f"{info['class_counts'].get(g, 0) / info['total']:.0%}")

    section(f"🔑 Most distinctive terms for {pretty(g)}")
    chips = "".join(
        f'<span class="chip" style="border-color:{gc};background:{gc}18;color:{gc}">{html.escape(w)}</span>'
        for w in info["top_words"][g]
    )
    st.markdown(chips, unsafe_allow_html=True)

    section("🧪 Interactive test")
    picks = st.multiselect("Select terms to see how the model responds", info["top_words"][g],
                           key=f"picks_{g}")
    if picks:
        p = predict([" ".join(picks)])[0]
        j = int(p.argmax())
        st.success(f"✅ Based on these terms alone, the model predicts **{labeled(classes[j])}** ({p[j]:.1%}).")
        st.bar_chart(pd.DataFrame({"Probability": p}, index=[pretty(c) for c in classes]))

st.markdown('<div class="footer">Built with Streamlit and scikit-learn &nbsp;|&nbsp; BBC News dataset</div>',
            unsafe_allow_html=True)