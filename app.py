"""
Week 3 - Streamlit Deployment
Amazon Reviews Sentiment Analyzer
Run: streamlit run app.py
"""

import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
from io import BytesIO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .sentiment-positive { background:#d4edda; color:#155724; border-radius:8px;
                          padding:8px 16px; font-weight:600; display:inline-block; }
    .sentiment-negative { background:#f8d7da; color:#721c24; border-radius:8px;
                          padding:8px 16px; font-weight:600; display:inline-block; }
    .sentiment-neutral  { background:#fff3cd; color:#856404; border-radius:8px;
                          padding:8px 16px; font-weight:600; display:inline-block; }
    .review-card { background:white; border-radius:10px; padding:20px;
                   box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-bottom:12px; }
    h1 { color: #2C3E50; }
    .metric-card { border-radius:10px; padding:16px; text-align:center; margin-bottom:8px; }
    .metric-card p { margin:0; font-size:13px; }
    .metric-card h2 { margin:4px 0 0 0; font-size:28px; }
    .metric-total    { background-color:#f0f2f6; }
    .metric-total p  { color:#666; }
    .metric-total h2 { color:#2C3E50; }
    .metric-positive    { background-color:#d4edda; }
    .metric-positive p  { color:#155724; }
    .metric-positive h2 { color:#155724; }
    .metric-neutral    { background-color:#fff3cd; }
    .metric-neutral p  { color:#856404; }
    .metric-neutral h2 { color:#856404; }
    .metric-negative    { background-color:#f8d7da; }
    .metric-negative p  { color:#721c24; }
    .metric-negative h2 { color:#721c24; }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_model.pkl")
    except FileNotFoundError:
        st.error("Model not found. Run the model building notebook first.")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_reviews.csv")

        if "vader_compound" not in df.columns:
            vader = SentimentIntensityAnalyzer()
            scores = df["clean"].fillna("").apply(vader.polarity_scores)
            df["vader_neg"]      = scores.apply(lambda x: x["neg"])
            df["vader_neu"]      = scores.apply(lambda x: x["neu"])
            df["vader_pos"]      = scores.apply(lambda x: x["pos"])
            df["vader_compound"] = scores.apply(lambda x: x["compound"])

        if "word_count" not in df.columns:
            df["word_count"] = df["clean"].fillna("").str.split().apply(len)

        if "char_count" not in df.columns:
            df["char_count"] = df["clean"].fillna("").str.len()

        return df
    except FileNotFoundError:
        return None

model    = load_model()
df       = load_data()
analyzer = SentimentIntensityAnalyzer()

# ── Helpers ──────────────────────────────────────────────────────────────────
STOPWORDS_EXTRA = set(STOPWORDS) | {"phone", "mobile", "samsung", "galaxy",
                                     "amazon", "product", "one", "also", "will",
                                     "buy", "bought", "use", "using"}

def clean_text(t):
    t = emoji.replace_emoji(str(t), replace="")
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def get_sentiment_badge(label):
    css = f"sentiment-{label}"
    emoji_map = {"positive": "😊 Positive", "negative": "😞 Negative", "neutral": "😐 Neutral"}
    return f'<span class="{css}">{emoji_map.get(label, label)}</span>'

def gauge_chart(score):
    color = "#2ECC71" if score > 0.05 else "#E74C3C" if score < -0.05 else "#F1C40F"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "VADER Compound Score", "font": {"size": 14}},
        gauge={
            "axis":    {"range": [-1, 1], "tickwidth": 1},
            "bar":     {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [-1, -0.05],  "color": "#FADBD8"},
                {"range": [-0.05, 0.05], "color": "#FEF9E7"},
                {"range": [0.05, 1],    "color": "#D5F5E3"},
            ],
            "threshold": {"line": {"color": "black", "width": 2},
                          "thickness": 0.75, "value": score},
        },
        number={"font": {"size": 28}, "valueformat": ".3f"},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)
    st.title("Settings")
    mode = st.radio("Mode", ["Single Review", "Bulk Analysis", "Dataset Insights"], index=0)
    st.markdown("---")
    st.caption("Amazon Review Sentiment Analyzer\nBuilt with Streamlit + scikit-learn")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 Amazon Review Sentiment Analyzer")
st.markdown("Predict whether a review is **Positive**, **Neutral**, or **Negative** using ML + VADER lexicon analysis.")
st.markdown("---")

# =============================================================================
# MODE 1: Single Review
# =============================================================================
if mode == "Single Review":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Enter Review")
        title        = st.text_input("Review Title", placeholder="e.g. Great phone for the price!")
        body         = st.text_area("Review Body", height=150, placeholder="Write your review here...")
        rating_input = st.slider("Star Rating (optional)", 1, 5, 3)
        analyze      = st.button("🔍 Analyze Sentiment", use_container_width=True)

    with col2:
        st.subheader("📌 Note")
        st.info("You can paste reviews in Hindi or Malayalam — the app handles multilingual text and emojis automatically.")

    if analyze and (title or body):
        full_text = f"{title} {body}"
        clean     = clean_text(full_text)
        vader_sc  = analyzer.polarity_scores(full_text)
        compound  = vader_sc["compound"]

        if model:
            ml_label = model.predict([clean])[0]
            proba    = model.predict_proba([clean])[0] if hasattr(model.named_steps["clf"], "predict_proba") else None
        else:
            ml_label = "unavailable"
            proba    = None

        vader_label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"

        st.markdown("---")
        st.subheader("🎯 Analysis Results")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="metric-card metric-total">
                <p>Your Rating</p>
                <h2>{rating_input}/5</h2>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card metric-{'positive' if ml_label=='positive' else 'negative' if ml_label=='negative' else 'neutral'}">
                <p>ML Prediction</p>
                <h2>{ml_label.capitalize()}</h2>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-card metric-{'positive' if vader_label=='positive' else 'negative' if vader_label=='negative' else 'neutral'}">
                <p>VADER Prediction</p>
                <h2>{vader_label.capitalize()}</h2>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**ML Model Sentiment**")
            st.markdown(get_sentiment_badge(ml_label), unsafe_allow_html=True)
            if proba is not None:
                labels  = model.classes_
                bar_fig = px.bar(
                    x=labels, y=proba,
                    color=labels,
                    color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#F1C40F"},
                    labels={"x": "Sentiment", "y": "Probability"},
                    title="Class Probabilities",
                )
                bar_fig.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(bar_fig, use_container_width=True)

        with col_b:
            st.plotly_chart(gauge_chart(compound), use_container_width=True)
            st.markdown("**VADER Component Scores**")
            vader_df = pd.DataFrame({
                "Component": ["Negative", "Neutral", "Positive", "Compound"],
                "Score":     [vader_sc["neg"], vader_sc["neu"], vader_sc["pos"], vader_sc["compound"]],
            })
            st.dataframe(vader_df.style.format({"Score": "{:.3f}"}), hide_index=True, use_container_width=True)

        ec = emoji.emoji_count(full_text)
        if ec:
            st.info(f"Detected {ec} emoji(s) in your review.")

# =============================================================================
# MODE 2: Bulk Analysis
# =============================================================================
elif mode == "Bulk Analysis":
    st.subheader("📂 Bulk Review Analysis")
    st.markdown("Upload a CSV or Excel file with columns: **title**, **rating**, **body**")

    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx"])

    if uploaded:
        user_df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
        st.success(f"Loaded {len(user_df)} reviews")
        st.dataframe(user_df.head(5), use_container_width=True)

        if st.button("🚀 Run Bulk Analysis"):
            with st.spinner("Analyzing reviews..."):
                user_df["clean"] = (user_df["title"].astype(str) + " " + user_df["body"].astype(str)).apply(clean_text)

                if model:
                    user_df["ml_sentiment"] = model.predict(user_df["clean"])

                user_df["vader_compound"] = user_df["clean"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
                user_df["vader_sentiment"] = user_df["vader_compound"].apply(
                    lambda c: "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
                )

            st.success("Analysis complete!")

            vc = user_df["ml_sentiment"].value_counts()

            b1, b2, b3 = st.columns(3)
            with b1:
                st.markdown(f"""
                <div class="metric-card metric-positive">
                    <p>Positive</p>
                    <h2>{int(vc.get("positive", 0))}</h2>
                </div>""", unsafe_allow_html=True)
            with b2:
                st.markdown(f"""
                <div class="metric-card metric-neutral">
                    <p>Neutral</p>
                    <h2>{int(vc.get("neutral", 0))}</h2>
                </div>""", unsafe_allow_html=True)
            with b3:
                st.markdown(f"""
                <div class="metric-card metric-negative">
                    <p>Negative</p>
                    <h2>{int(vc.get("negative", 0))}</h2>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            pie_fig = px.pie(
                values=vc.values, names=vc.index,
                color=vc.index,
                color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#F1C40F"},
                title="Sentiment Distribution",
            )
            st.plotly_chart(pie_fig, use_container_width=True)
            st.dataframe(user_df[["title", "rating", "ml_sentiment", "vader_compound", "vader_sentiment"]], use_container_width=True)

            csv_out = user_df.to_csv(index=False).encode()
            st.download_button("Download Results CSV", csv_out, "sentiment_results.csv", "text/csv")

# =============================================================================
# MODE 3: Dataset Insights
# =============================================================================
elif mode == "Dataset Insights":
    if df is None:
        st.warning("cleaned_reviews.csv not found. Run the EDA notebook first.")
    else:
        st.subheader("📊 Dataset Insights")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card metric-total">
                <p>Total Reviews</p>
                <h2>{len(df)}</h2>
            </div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card metric-positive">
                <p>Positive</p>
                <h2>{int((df["sentiment"] == "positive").sum())}</h2>
            </div>""", unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card metric-neutral">
                <p>Neutral</p>
                <h2>{int((df["sentiment"] == "neutral").sum())}</h2>
            </div>""", unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card metric-negative">
                <p>Negative</p>
                <h2>{int((df["sentiment"] == "negative").sum())}</h2>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Distributions", "☁️ Word Clouds", "🔥 Correlations"])

        with tab1:
            c1, c2 = st.columns(2)

            with c1:
                cnt = df["rating"].value_counts().sort_index()
                fig = px.bar(
                    x=cnt.index, y=cnt.values,
                    labels={"x": "Rating", "y": "Count"},
                    color=cnt.index,
                    color_continuous_scale="RdYlGn",
                    title="Rating Distribution",
                )
                fig.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                sent_cnt = df["sentiment"].value_counts()
                fig2 = px.pie(
                    values=sent_cnt.values,
                    names=sent_cnt.index,
                    color=sent_cnt.index,
                    color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#F1C40F"},
                    title="Sentiment Breakdown",
                )
                st.plotly_chart(fig2, use_container_width=True)

            if "vader_compound" in df.columns:
                fig3 = px.violin(
                    df, x="sentiment", y="vader_compound",
                    color="sentiment",
                    color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#F1C40F"},
                    box=True, points="outliers",
                    title="VADER Compound Score by Sentiment",
                )
                st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            sentiment_filter = st.selectbox("Select Sentiment", ["positive", "negative", "neutral"])
            text = " ".join(df[df["sentiment"] == sentiment_filter]["clean"].fillna("").tolist())

            if text.strip():
                if sentiment_filter == "positive":
                    cmap = "Greens"
                elif sentiment_filter == "negative":
                    cmap = "Reds"
                else:
                    cmap = "Blues"

                wc = WordCloud(
                    width=800, height=400,
                    background_color="white",
                    stopwords=STOPWORDS_EXTRA,
                    max_words=100,
                    colormap=cmap,
                ).generate(text)

                fig_wc, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(f"Word Cloud - {sentiment_filter.capitalize()} Reviews", fontsize=13, fontweight="bold")
                st.pyplot(fig_wc)
            else:
                st.warning("No text available for selected sentiment.")

        with tab3:
            wanted = ["rating", "word_count", "char_count",
                      "vader_neg", "vader_neu", "vader_pos", "vader_compound"]
            numeric_cols = [c for c in wanted if c in df.columns]

            if len(numeric_cols) < 2:
                st.warning("Not enough numeric columns. Run the EDA notebook first.")
            else:
                corr = df[numeric_cols].corr()
                fig_h = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlation Heatmap",
                )
                st.plotly_chart(fig_h, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Amazon Sentiment Analyzer · Week 3 Deployment · Built with Streamlit")
