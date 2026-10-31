"""Streamlit app for diamond price prediction with a KNN regression pipeline."""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "diamonds.csv"
MODEL_PATH = ROOT / "models" / "diamond_knn_pipeline.pkl"
TARGET = "price"
K_NEIGHBORS = 7
REQUIRED_FEATURES = (
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "x",
    "y",
    "z",
)

st.set_page_config(
    page_title="Diamond Price Studio",
    page_icon="💎",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {color-scheme: dark;}
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #0e1117;
        color: #f1f5f9;
    }
    [data-testid="stHeader"] {background: #0e1117;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    h1, h2, h3, p, label,
    [data-testid="stMarkdownContainer"] {
        color: #f1f5f9;
    }
    [data-testid="stMetric"] {
        background: #172433;
        border: 1px solid #2b3d50;
        padding: 1rem;
        border-radius: 0.75rem;
        color: #f1f5f9;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    [data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #f1f5f9 !important;
    }
    .hero {
        padding: 1.5rem 1.75rem;
        border: 1px solid #26384b;
        border-left: 5px solid #168477;
        border-radius: 0.75rem;
        background: linear-gradient(120deg, #172433, #1d3044);
        color: #f1f5f9;
        margin-bottom: 1.25rem;
    }
    .hero h1 {color: #f8fafc;}
    .hero p {color: #cbd5e1; margin-bottom: 0;}
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #ffffff;
        border-color: #cbd6e2;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] *,
    [data-testid="stNumberInput"] button {
        color: #243247;
    }
    [data-testid="stFormSubmitButton"] button {
        background: #376b8a;
        border-color: #376b8a;
        color: #ffffff;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #2b5873;
        border-color: #2b5873;
        color: #ffffff;
    }
    [data-testid="stAlert"] {
        background: #e8f4f1;
        border: 1px solid #b9ddd5;
        color: #18594f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the source CSV."""
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"The dataset must contain a '{TARGET}' column.")

    data = data.loc[:, ~data.columns.astype(str).str.match(r"^Unnamed:")]
    data = data.dropna(subset=[TARGET]).copy()
    if data.empty:
        raise ValueError("The dataset has no rows with a valid price.")
    if not all(column in data.columns for column in REQUIRED_FEATURES):
        missing = [column for column in REQUIRED_FEATURES if column not in data.columns]
        raise ValueError(f"The dataset is missing required features: {', '.join(missing)}.")
    return data


@st.cache_resource(show_spinner="Training the KNN model on diamonds.csv...")
def train_model(
    path: str, modified_at: float
) -> tuple[Pipeline, pd.DataFrame]:
    """Train the preprocessing + KNN pipeline, then save it."""
    del modified_at  # Included in the cache key so a changed CSV triggers retraining.
    data = load_dataset(path)
    features = data.drop(columns=[TARGET])
    target = data[TARGET]
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                KNeighborsRegressor(n_neighbors=K_NEIGHBORS, weights="distance"),
            ),
        ]
    )

    X_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.25, random_state=42
    )
    pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(pipeline, model_file)

    return pipeline, data


def main() -> None:
    if not DATA_PATH.is_file():
        st.error(f"Could not find the dataset: `{DATA_PATH}`")
        st.info("Place `diamonds.csv` in the same folder as `daimond.py`.")
        st.stop()

    try:
        data = load_dataset(str(DATA_PATH))
        pipeline, data = train_model(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to prepare the diamond model: {error}")
        st.stop()

    st.markdown(
        """
        <div class="hero">
            <h1>Diamond price, made clearer.</h1>
            <p>Estimate a diamond's market price from its physical characteristics
            using a trained KNN regression model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Price estimator")
    st.write("Enter the stone's measurements and quality grades to get an estimate.")

    with st.form("diamond_prediction_form"):
        first_row = st.columns(3)
        carat = first_row[0].number_input(
            "Carat", min_value=0.01, value=0.50, step=0.01, format="%.2f"
        )
        cut = first_row[1].selectbox(
            "Cut", options=sorted(data["cut"].dropna().unique().tolist())
        )
        color = first_row[2].selectbox(
            "Color", options=sorted(data["color"].dropna().unique().tolist())
        )

        second_row = st.columns(3)
        clarity = second_row[0].selectbox(
            "Clarity", options=sorted(data["clarity"].dropna().unique().tolist())
        )
        depth = second_row[1].number_input(
            "Depth (%)", min_value=0.0, value=61.5, step=0.1, format="%.1f"
        )
        table = second_row[2].number_input(
            "Table (%)", min_value=0.0, value=57.0, step=0.1, format="%.1f"
        )

        third_row = st.columns(3)
        x = third_row[0].number_input(
            "Length (x, mm)", min_value=0.0, value=5.0, step=0.01, format="%.2f"
        )
        y = third_row[1].number_input(
            "Width (y, mm)", min_value=0.0, value=5.0, step=0.01, format="%.2f"
        )
        z = third_row[2].number_input(
            "Height (z, mm)", min_value=0.0, value=3.1, step=0.01, format="%.2f"
        )
        submitted = st.form_submit_button(
            "Estimate price", type="primary", use_container_width=True
        )

    if submitted:
        sample = pd.DataFrame(
            [
                {
                    "carat": carat,
                    "cut": cut,
                    "color": color,
                    "clarity": clarity,
                    "depth": depth,
                    "table": table,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            ]
        )
        estimate = float(pipeline.predict(sample)[0])
        st.metric("Estimated price", f"${estimate:,.0f}")
        st.caption(
            "This is a model estimate based on the training data, not a formal appraisal."
        )


if __name__ == "__main__":
    main()