import pandas as pd
from transformers import pipeline

# Load FinBERT model (finance-trained sentiment model)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def get_sentiment(text):
    """
    Returns sentiment label and confidence score
    """
    if text is None or str(text).strip() == "":
        return "neutral", 0.0

    try:
        result = sentiment_pipeline(str(text)[:512])[0]
        return result["label"], result["score"]
    except Exception as e:
        print("Error processing text:", e)
        return "neutral", 0.0


def run_sentiment(input_file, output_file):
    """
    Reads news CSV, applies sentiment analysis, saves output CSV
    """

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return

    print(f"✅ Loaded dataset with {len(df)} rows")

    if df.empty:
        print("❌ Dataset is empty")
        return

    sentiments = []
    scores = []

    for i, text in enumerate(df["content"]):
        print(f"Processing {i + 1}/{len(df)}")

        label, score = get_sentiment(text)

        sentiments.append(label)
        scores.append(score)

    df["sentiment"] = sentiments
    df["confidence_score"] = scores

    df.to_csv(output_file, index=False)

    print("\n✅ Sentiment analysis completed successfully!")
    print(f"📁 Output saved to: {output_file}")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    run_sentiment(
        input_file="data/news.csv",
        output_file="data/news_sentiment.csv"
    )