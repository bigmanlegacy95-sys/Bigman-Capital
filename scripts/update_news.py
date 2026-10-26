import json
import os

def update_market_data():
    # Add your news fetching or data generation logic here
    news_data = {
        "status": "updated",
        "message": "Market data refreshed successfully."
    }
    
    # Save to news.json in the root directory
    with open('news.json', 'w') as f:
        json.dump(news_data, f, indent=4)

if __name__ == "__main__":
    update_market_data()
