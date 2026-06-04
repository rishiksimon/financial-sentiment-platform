import yfinance as yf
import pandas as pd

def get_stock_data(ticker="AAPL", period="1mo", interval="1d"):
    """
    Fetch stock data using Yahoo Finance
    """

    stock = yf.download(ticker, period=period, interval=interval)

    if stock.empty:
        print("❌ No stock data found")
        return pd.DataFrame()

    stock = stock.reset_index()

    # FIX: flatten MultiIndex columns if they exist
    stock.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in stock.columns
    ]

    # Standardize column names (important for dashboard consistency)
    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }

    stock.rename(columns=rename_map, inplace=True)

    return stock


def get_multiple_stocks(tickers=["AAPL", "TSLA"]):
    """
    Optional: compare multiple stocks
    """
    all_data = []

    for ticker in tickers:
        df = yf.download(ticker, period="1mo", interval="1d")

        if df.empty:
            continue

        df = df.reset_index()

        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

        df["ticker"] = ticker
        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    df = get_stock_data("AAPL")
    print(df.head())