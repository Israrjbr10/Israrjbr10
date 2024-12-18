
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime

# Define Yahoo Finance URL for the index (e.g., S&P 500)
YAHOO_FINANCE_URL = "https://finance.yahoo.com/markets/stocks/most-active/"

# Function to fetch data from Yahoo Finance
def fetch_index_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the necessary data using CSS selectors
    try:
        # Get the current price of the index
        price = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'}).text
        # Get the change and percentage change
        change = soup.find('fin-streamer', {'data-field': 'regularMarketChange'}).text
        percent_change = soup.find('fin-streamer', {'data-field': 'regularMarketChangePercent'}).text
    except AttributeError:
        print("Error fetching data. Yahoo Finance might have changed its structure.")
        return None

    # Return the data as a dictionary
    return {
        'price': price,
        'change': change,
        'percent_change': percent_change,
        'timestamp': datetime.now()
    }

# Function to create a database and a table if they don't exist
def setup_database():
    conn = sqlite3.connect('market_index.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS index_data (
                        id INTEGER PRIMARY KEY,
                        price TEXT,
                        change TEXT,
                        percent_change TEXT,
                        timestamp DATETIME)''')
    conn.commit()
    conn.close()

# Function to insert data into the database
def insert_data(data):
    conn = sqlite3.connect('market_index.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO index_data (price, change, percent_change, timestamp)
                      VALUES (?, ?, ?, ?)''',
                   (data['price'], data['change'], data['percent_change'], data['timestamp']))
    conn.commit()
    conn.close()

# Main script
def main():
    setup_database()
    data = fetch_index_data(YAHOO_FINANCE_URL)
    if data:
        insert_data(data)
        print("Data inserted into the database successfully:", data)
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    main()


