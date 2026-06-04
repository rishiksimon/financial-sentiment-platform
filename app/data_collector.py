import requests
import pandas as pd

API_KEY = "9423983e684641e1b745f684ade51c56"

def fetch_news(query="stock market OR finance OR economy"):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=50&apiKey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    if "articles" not in data:
        print("Error:", data)
        return pd.DataFrame()

    articles = []

    for article in data["articles"]:
        articles.append({
            "title": article["title"],
            "description": article["description"],
            "content": article["content"],
            "source": article["source"]["name"],
            "published_at": article["publishedAt"]
        })

    return pd.DataFrame(articles)


if __name__ == "__main__":
    df = fetch_news()
    print(df.head())
    df.to_csv("data/news.csv", index=False)