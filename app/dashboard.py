import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Financial Sentiment Dashboard", layout="wide")

st.title("📊 Financial Sentiment Analysis Dashboard")

# -----------------------
# LOAD SENTIMENT DATA
# -----------------------
df = pd.read_csv("data/news_sentiment.csv")

st.subheader("Dataset Overview")
st.write(df.head())
st.write("Total Articles:", len(df))

# -----------------------
# SENTIMENT DISTRIBUTION
# -----------------------
st.subheader("Sentiment Distribution")

fig1 = px.histogram(
    df,
    x="sentiment",
    color="sentiment",
    title="Sentiment Breakdown"
)

st.plotly_chart(fig1, width="stretch")

# -----------------------
# CONFIDENCE SCORE
# -----------------------
st.subheader("Confidence Score Distribution")

fig2 = px.histogram(
    df,
    x="confidence_score",
    nbins=20,
    title="Model Confidence"
)

st.plotly_chart(fig2, width="stretch")

# -----------------------
# TIME SERIES SENTIMENT
# -----------------------
st.subheader("Sentiment Over Time")

df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
df = df.dropna(subset=["published_at"])

df_grouped = df.groupby(
    [df["published_at"].dt.date, "sentiment"]
).size().reset_index(name="count")

fig3 = px.line(
    df_grouped,
    x="published_at",
    y="count",
    color="sentiment",
    title="Daily Sentiment Trend"
)

st.plotly_chart(fig3, width="stretch")

# -----------------------
# FILTER SECTION
# -----------------------
st.subheader("Filter News")

sentiment_filter = st.selectbox(
    "Select Sentiment",
    ["All"] + list(df["sentiment"].unique())
)

if sentiment_filter != "All":
    filtered_df = df[df["sentiment"] == sentiment_filter]
else:
    filtered_df = df

st.dataframe(filtered_df, use_container_width=True)

# -----------------------
# STOCK PRICE SECTION
# -----------------------
st.subheader("📈 Stock Price Analysis")

ticker = st.selectbox(
    "Select Stock",
    ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN"]
)

stock = yf.download(ticker, period="1mo", interval="1d")
stock = stock.reset_index()

# FIX: flatten multi-index columns
stock.columns = [
    col[0] if isinstance(col, tuple) else col
    for col in stock.columns
]

fig4 = px.line(
    stock,
    x="Date",
    y="Close",
    title=f"{ticker} Stock Price Trend"
)

st.plotly_chart(fig4, width="stretch")

# -----------------------
# SENTIMENT vs STOCK CORRELATION
# -----------------------
st.subheader("📊 Sentiment vs {ticker} Stock Correlation")

# Stock preprocessing
stock_corr = stock[["Date", "Close"]].copy()
stock_corr["Date"] = pd.to_datetime(stock_corr["Date"]).dt.date
stock_corr.rename(columns={"Close": "close_price"}, inplace=True)

# Sentiment preprocessing
df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date

sentiment_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

df["sentiment_score"] = df["sentiment"].map(sentiment_map)

sent_daily = df.groupby("published_at")["sentiment_score"].mean().reset_index()
sent_daily.rename(columns={"published_at": "Date"}, inplace=True)

# Merge
merged = pd.merge(stock_corr, sent_daily, on="Date", how="inner")

st.write("Merged Dataset")
st.dataframe(merged)

if not merged.empty:
    fig_corr = px.scatter(
        merged,
        x="sentiment_score",
        y="close_price",
        trendline="ols",
        title="Sentiment vs Stock Price Correlation"
    )

    st.plotly_chart(fig_corr, width="stretch")
else:
    st.warning("Not enough overlapping data between stock and sentiment dates.")